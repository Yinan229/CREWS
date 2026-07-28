from __future__ import annotations
import csv
import os
import random
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..config import ExperimentConfig
from ..models import (
    edge_jetson_model as EdgeJetsonModel,
    server_bottom_model as ServerBottomModel,
    server_top_model as ServerTopModel,
    edge_split_model as EdgeSplitModel,
)
from ..simulation import (
    RxClassCache,
    TestMaskPlanner,
    bernoulli_drop_rids,
    make_S1,
    make_S3
)
from ..epa import (
    epa_align_bottom_plus_edgeon,
    get_active_rids_for_epa_round,
)

def split_one_batch_to_client_input(batch_x, client_id):
    """Take the ``client_id``-th view input tensor from ``batch_x`` ([B, R, ...])."""
    return batch_x[:, client_id, ...]

class Trainer:
    def __init__(self, cfg: ExperimentConfig, train_loader, test_loader):
        self.cfg = cfg
        self.device = cfg.device
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.maxnum_of_clients = cfg.maxnum_of_clients
        self.num_classes = cfg.num_classes
        self.rxs = list(range(cfg.maxnum_of_clients))
        self.server_bottom_model = ServerBottomModel().to(self.device)
        self.edge_split_model = nn.ModuleDict({
            str(i): EdgeSplitModel(num_blocks=cfg.client_blocks[i]).to(self.device)
            for i in range(self.maxnum_of_clients)
        })
        self.server_top_model = ServerTopModel(num_classA=cfg.num_classes).to(self.device)
        self.edge_jetson_model = nn.ModuleDict({
            str(i): EdgeJetsonModel(num_blocks=cfg.client_blocks[i]).to(self.device)
            for i in range(self.maxnum_of_clients)
        })
        server_params = []
        for i in range(self.maxnum_of_clients):
            server_params += list(self.edge_split_model[str(i)].parameters())
        server_params += list(self.server_bottom_model.parameters())
        server_params += list(self.server_top_model.parameters())
        self.server_optimizer = torch.optim.Adam(server_params, lr=cfg.lr)
        self.bottom_optimizers = {
            str(i): torch.optim.Adam(self.edge_jetson_model[str(i)].parameters(), lr=cfg.lr)
            for i in range(self.maxnum_of_clients)
        }
        self.criterion = nn.CrossEntropyLoss()
        self.train_rng = random.Random(cfg.train_rng_seed)
        self.epa_rng = random.Random(cfg.epa_rng_seed)
        self.test_mask_planner = TestMaskPlanner(cfg.sim_seed_test, self.maxnum_of_clients)
        self.rx_cache = RxClassCache(
            self.device, self.rxs, cfg.num_classes,
            num_batches_keep=cfg.num_batches_keep, ts_global=1,
        )
        self.drop_probs_train, self.drop_probs_test = make_S3(cfg.s3_prob)
        self.drop_probs_epa = self.drop_probs_train

    def train_one_epoch(self, ts_global):
        cfg = self.cfg
        device = self.device
        for i in range(self.maxnum_of_clients):
            self.edge_jetson_model[str(i)].train()
            self.edge_split_model[str(i)].train()
        self.server_bottom_model.train()
        self.server_top_model.train()
        total_loss = 0.0
        total_loss_used = 0.0
        total_loss_stale = 0.0
        total_correct = 0
        total_seen = 0
        STALE_BUDGET = cfg.stale_budget
        LAMBDA_DECAY = cfg.lambda_decay
        for batch_idx, (batch_x, batch_y) in enumerate(self.train_loader):
            if batch_y.unique().numel() < self.num_classes:
                continue
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if ts_global == 0 and batch_idx == 0:
                present_rids = self.rxs
                missing_rids = []
            else:
                present_rids, missing_rids = bernoulli_drop_rids(
                    self.maxnum_of_clients, self.train_rng, self.drop_probs_train
                )
            self.rx_cache.record_fresh(present_rids)
            self.server_optimizer.zero_grad(set_to_none=True)
            for rid in self.rxs:
                self.bottom_optimizers[str(rid)].zero_grad(set_to_none=True)
            smashed = {}
            for rid in self.rxs:
                x_i = split_one_batch_to_client_input(batch_x, rid).to(device)
                smashed[rid] = self.edge_jetson_model[str(rid)](x_i)
            edge_feats = {}
            for rid in present_rids:
                edge_feats[rid] = self.edge_split_model[str(rid)](smashed[rid])
            y_sel = batch_y.clone()
            num_extra = min(STALE_BUDGET, self.maxnum_of_clients - len(present_rids))
            stale_feats, stale_rids, stale_age_Bs = self.rx_cache.sample_stale_views(
                present_rids=present_rids, labels_B=y_sel, num_extra=num_extra
            )
            feats_list = [edge_feats[rid] for rid in present_rids]
            if cfg.jitter:
                if len(feats_list) > 1 and self.train_rng.random() < cfg.jitter_swap_prob:
                    idx1, idx2 = self.train_rng.sample(range(len(feats_list)), 2)
                    feats_list[idx1], feats_list[idx2] = feats_list[idx2], feats_list[idx1]
            xA_used = torch.cat(feats_list, dim=3)
            have_stale = (stale_feats is not None) and (len(stale_feats) > 0)
            if have_stale:
                stale_list = [sf.detach() for sf in stale_feats]
                xA_stale = torch.cat(stale_list, dim=3)
            H_used = self.server_bottom_model(xA_used, xA_used.shape[3])
            logits_used = self.server_top_model(H_used)
            loss_used = self.criterion(logits_used, batch_y)
            loss_stale = 0
            if have_stale:
                H_stale = self.server_bottom_model(xA_stale, xA_stale.shape[3])
                logits_stale = self.server_top_model(H_stale)
                ce_stale = F.cross_entropy(logits_stale, batch_y, reduction="none")
                age_mat = torch.stack(
                    [a.to(device=ce_stale.device) for a in stale_age_Bs], dim=1
                )
                age_eff = age_mat.float().mean(dim=1)
                w = (LAMBDA_DECAY ** age_eff).to(dtype=ce_stale.dtype, device=ce_stale.device)
                w_sum = w.sum().clamp_min(1e-8)
                if cfg.normalize_stale_loss:
                    loss_stale = (w * ce_stale).sum() / w_sum
                else:
                    loss_stale = (w * ce_stale).mean()
            loss = loss_used + loss_stale
            loss.backward()
            self.server_optimizer.step()
            for rid in self.rxs:
                self.bottom_optimizers[str(rid)].step()
            with torch.no_grad():
                self.rx_cache.update_from_Rx(missing_rids)
                for rid in range(self.maxnum_of_clients):
                    if rid in present_rids:
                        self.rx_cache.update_from_batch(rid, edge_feats[rid], y_sel, True)
                    else:
                        self.rx_cache.update_from_batch(rid, None, y_sel, False)
            with torch.no_grad():
                pred = logits_used.argmax(dim=1)
                total_correct += (pred == batch_y).sum().item()
                total_seen += batch_y.size(0)
                bs = batch_y.size(0)
                total_loss += float(loss.item()) * bs
                total_loss_used += float(loss_used.item()) * bs
                total_loss_stale += float(loss_stale.item()) * bs if have_stale else 0.0
        avg_loss = total_loss / max(total_seen, 1)
        avg_loss_used = total_loss_used / max(total_seen, 1)
        avg_loss_stale = total_loss_stale / max(total_seen, 1)
        avg_acc = total_correct / max(total_seen, 1)
        return avg_loss, avg_acc, avg_loss_used, avg_loss_stale

    @torch.no_grad()
    def eval_one_epoch(self):
        cfg = self.cfg
        device = self.device
        for i in range(self.maxnum_of_clients):
            self.edge_jetson_model[str(i)].eval()
            self.edge_split_model[str(i)].eval()
        self.server_bottom_model.eval()
        self.server_top_model.eval()
        total_loss = 0.0
        total_correct = 0
        total_seen = 0
        for batch_idx, (batch_x, batch_y) in enumerate(self.test_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            if cfg.fix_test_missing:
                present_rids_test, test_shuffle_indices = self.test_mask_planner.get_present_rids(
                        batch_idx=batch_idx, drop_probs_test=self.drop_probs_test
                    )
            else:
                present_rids_test, _ = bernoulli_drop_rids(
                    self.maxnum_of_clients, random.Random(), self.drop_probs_test
                )
                test_shuffle_indices = list(range(len(present_rids_test)))
                if cfg.jitter:
                    random.shuffle(test_shuffle_indices)
            smashed = {}
            for rid in present_rids_test:
                x_i = split_one_batch_to_client_input(batch_x, rid).to(device)
                smashed[rid] = self.edge_jetson_model[str(rid)](x_i)
            edge_feats = {}
            for rid in present_rids_test:
                edge_feats[rid] = self.edge_split_model[str(rid)](smashed[rid])
            feats_list = [edge_feats[rid] for rid in present_rids_test]
            if cfg.jitter:
                feats_list = [feats_list[i] for i in test_shuffle_indices]
            xA_used_test = torch.cat(feats_list, dim=3)
            H = self.server_bottom_model(xA_used_test, len(present_rids_test))
            logits = self.server_top_model(H)
            loss = self.criterion(logits, batch_y)
            pred = logits.argmax(dim=1)
            total_correct += (pred == batch_y).sum().item()
            total_seen += batch_y.size(0)
            total_loss += loss.item() * batch_y.size(0)
        avg_loss = total_loss / max(total_seen, 1)
        avg_acc = total_correct / max(total_seen, 1)
        return avg_loss, avg_acc

    def fit(self):
        cfg = self.cfg
        result_dir = os.path.abspath(cfg.output_dir)
        ckpt_dir = os.path.join(result_dir, "checkpoints")
        os.makedirs(result_dir, exist_ok=True)
        os.makedirs(ckpt_dir, exist_ok=True)
        csv_path = os.path.join(result_dir, f"{cfg.experiment_name}_seed{cfg.seed}.csv")
        if not os.path.exists(csv_path):
            with open(csv_path, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "timestamp", "epoch",
                    "train_loss", "train_acc",
                    "train_loss_used", "train_loss_stale",
                    "test_loss", "test_acc",
                ])
        ts_global = 0
        for epoch in range(cfg.epochs):
            train_loss, train_acc, train_loss_used, train_loss_stale =  self.train_one_epoch(ts_global)
            active_rids_epoch = get_active_rids_for_epa_round(
                maxnum_of_clients=self.maxnum_of_clients,
                rng=self.epa_rng,
                drop_probs_epa=self.drop_probs_epa,
            )
            if cfg.k_align == 'all':
                active_rids_epoch = self.rxs
            if len(active_rids_epoch) >= 2:
                if ts_global % cfg.epa_interval == 0:
                    self.rx_cache.reset_cycle_counts()
                    self.rx_cache.clear_cache_after_fed(reset_counts=True, reset_slots=True)
                    epa_align_bottom_plus_edgeon(
                        self.edge_jetson_model, self.edge_split_model,
                        maxnum_of_clients=self.maxnum_of_clients,
                        rids=active_rids_epoch,
                        strict_load=False,
                        mu=cfg.epa_mu,
                    )
            else:
                print(f"[Epoch {epoch+1:03d}] Skip EPA: active clients < 2, "
                      f"active_rids={active_rids_epoch}")
            test_loss, test_acc = self.eval_one_epoch()
            ts_global = ts_global + 1
            print(f"[Epoch {epoch+1:03d}/{cfg.epochs}] "
                  f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f} | "
                  f"used={train_loss_used:.4f}, stale={train_loss_stale:.4f} | "
                  f"test_loss={test_loss:.4f}, test_acc={test_acc:.4f} | "
                  f"active_fed={len(active_rids_epoch)}, rids={active_rids_epoch}")
            with open(csv_path, "a", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    datetime.now().isoformat(timespec="seconds"),
                    epoch + 1,
                    f"{train_loss:.6f}", f"{train_acc:.6f}",
                    f"{train_loss_used:.6f}", f"{train_loss_stale:.6f}",
                    f"{test_loss:.6f}", f"{test_acc:.6f}",
                ])
        ckpt_path = os.path.join(ckpt_dir, f"{cfg.experiment_name}_seed{cfg.seed}_final.pt")
        torch.save({
            'epoch': cfg.epochs,
            'seed': cfg.seed,
            'edge_jetson_model': {str(i): self.edge_jetson_model[str(i)].state_dict()
                                  for i in range(self.maxnum_of_clients)},
            'edge_split_model': {str(i): self.edge_split_model[str(i)].state_dict()
                                 for i in range(self.maxnum_of_clients)},
            'server_bottom_model': self.server_bottom_model.state_dict(),
            'server_top_model': self.server_top_model.state_dict(),
        }, ckpt_path)
        print(f"[Checkpoint] Saved to {ckpt_path}")
        return csv_path, ckpt_path
