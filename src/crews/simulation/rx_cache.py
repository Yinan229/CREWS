"""Stale-feature ring-buffer cache (``RxClassCache``).
Stores per-timestep, per-client, per-class edge features so that missing clients
can be replaced by stale "views" during training. Also implements the figure-style
coverage-maximization sampling of extra (stale) client views.
"""
from __future__ import annotations
import math
from collections import deque, defaultdict
import torch

class RxClassCache:
    """Aligns cached views to the global sim_id for ``max_sample_mat``.
    - For every timestep, every rid and every class ``y`` we append once.
    - present: append a Tensor.
    - missing: append ``None`` as a placeholder.
    """

    def __init__(self, device, all_rids, num_classes, num_batches_keep=3, ts_global=0):
        self.device = device
        self.num_classes = int(num_classes)
        self.all_rids = [int(r) for r in all_rids]
        self.num_batches_keep = int(num_batches_keep)
        self.slots = deque(range(0, self.num_batches_keep), maxlen=self.num_batches_keep)
        self.store = {
            slot: {rid: {y: None for y in range(self.num_classes)} for rid in self.all_rids}
            for slot in list(self.slots)
        }
        self.drop_by_ts = {}
        self.cur_ts = self.slots[-1]
        self.fresh_count = defaultdict(int)
        self.stale_count = defaultdict(int)

    def reset_cycle_counts(self):
        """Call once at the start of each EPA cycle."""
        self.fresh_count = defaultdict(int)
        self.stale_count = defaultdict(int)

    def record_fresh(self, present_rids):
        """Record fresh rids present in each batch."""
        for rid in present_rids:
            self.fresh_count[int(rid)] += 1

    def has_bank_for_labels(self, rid, labels_B):
        rid = int(rid)
        labels_B = labels_B.view(-1).to(torch.long)
        for y in labels_B.unique().tolist():
            bank, _ = self.get_latest_nonempty_bank(rid, int(y))
            if bank is None:
                return False
        return True

    def get(self, rid, y, ts):
        return self.store[int(ts)][int(rid)][int(y)]

    def update_from_Rx(self, drop_rids):
        ts = self.slots.popleft()
        self.slots.append(ts)
        self.cur_ts = ts
        for rid in self.store[ts]:
            for y in self.store[ts][rid]:
                self.store[ts][rid][y] = None
        self.drop_by_ts[ts] = list(drop_rids)

    def clear_cache_after_fed(self, reset_counts=True, reset_slots=True):
        """Call after EPA / periodic encoder refresh.
        1) Clear all stale feature banks (``self.store``).
        2) Clear the per-slot dropout records (``self.drop_by_ts``).
        3) Optionally reset the ring-buffer slot order.
        4) Optionally reset fresh/stale participation counts.
        """
        for ts in self.store:
            for rid in self.store[ts]:
                for y in self.store[ts][rid]:
                    self.store[ts][rid][y] = None
        self.drop_by_ts = {}
        if reset_slots:
            self.slots = deque(range(0, self.num_batches_keep), maxlen=self.num_batches_keep)
        self.cur_ts = self.slots[-1]
        if reset_counts:
            self.reset_cycle_counts()

    @torch.no_grad()
    def update_from_batch(self, rid, feats_BCT1, labels_B, is_present):
        ts = self.cur_ts
        rid = int(rid)
        labels_B = labels_B.view(-1).to(torch.long)
        for y in labels_B.unique().tolist():
            y = int(y)
            if is_present:
                idxs = (labels_B == y).nonzero(as_tuple=True)[0]
                feat = feats_BCT1[idxs].detach()
            else:
                feat = None
            self.store[ts][rid][y] = feat

    def _slots_newest_first(self):
        return list(reversed(list(self.slots)))

    def get_latest_nonempty_bank(self, rid, y):
        """Return ``(bank, age)`` where age 0 is the newest slot."""
        rid = int(rid); y = int(y)
        slots = self._slots_newest_first()
        for age, ts in enumerate(slots):
            bank = self.get(rid, y, ts)
            if bank is not None:
                return bank, age
        return None, None

    @torch.no_grad()
    def sample_BCT1_for_rid(self, rid, labels_B):
        """Sample a "view" aligned to the current batch.
        Returns ``(out [B,C,T,1], age_B [B])``.
        """
        rid = int(rid)
        labels_B = labels_B.view(-1).to(torch.long)
        B = int(labels_B.numel())
        y0 = int(labels_B.unique()[0].item())
        bank0, _ = self.get_latest_nonempty_bank(rid, y0)
        if bank0 is None:
            return None, None
        out = torch.empty((B,) + bank0.shape[1:], device=self.device, dtype=bank0.dtype)
        age_B = torch.empty((B,), device=self.device, dtype=torch.long)
        for y in labels_B.unique().tolist():
            y = int(y)
            idxs = (labels_B == y).nonzero(as_tuple=True)[0]
            bank, age = self.get_latest_nonempty_bank(rid, y)
            Ny = int(bank.size(0))
            pick = torch.randint(low=0, high=Ny, size=(idxs.numel(),), device=self.device)
            out[idxs] = bank[pick]
            age_B[idxs] = age
        return out, age_B

    def _rid_age_for_labels(self, rid: int, labels_B: torch.Tensor) -> float:
        """rid-level ``age_t(k)`` proxy used during sampling."""
        rid = int(rid)
        labels_B = labels_B.view(-1).to(torch.long)
        ages = []
        for y in labels_B.unique().tolist():
            _, age = self.get_latest_nonempty_bank(rid, int(y))
            if age is None:
                return float("inf")
            ages.append(float(age))
        return float(sum(ages) / max(len(ages), 1))

    def pick_extra_rids_fig_sampling(
        self,
        present_rids,
        labels_B,
        num_extra,
        beta: float = 0.5,
        delta: float = 1.0,
        gamma: float = 1.0,
        eta: float = 1.0,
        alpha: float = 5.0,
        eps: float = 1e-6,
        fixed_size: bool = True,
    ):
        """Figure-style sampling of extra (stale) client views.
        r_t(k)  ~ (c_{t-1}(k)+delta)^(-beta) normalized
        a_hat   = age / sum(age)
        a_t(k)  = exp(-gamma * a_hat)
        u_t(k)  = log r + eta * sqrt(a) + eps
        p_t(k)  = sigmoid(alpha*(u - mean(u)))
        """
        present = set(int(r) for r in present_rids)
        candidates = [int(r) for r in self.all_rids if r not in present]
        avail = []
        age_list = []
        for r in candidates:
            if self.has_bank_for_labels(r, labels_B):
                a = self._rid_age_for_labels(r, labels_B)
                if math.isfinite(a):
                    avail.append(r)
                    age_list.append(a)
        if len(avail) == 0 or num_extra <= 0:
            return []
        counts = torch.tensor(
            [float(self.fresh_count[r]) for r in avail],
            device=self.device
        )
        r_unnorm = (counts + float(delta)).pow(-float(beta))
        r = r_unnorm / (r_unnorm.sum().clamp_min(1e-12))
        ages = torch.tensor(age_list, device=self.device, dtype=torch.float32).clamp_min(0.0)
        a_hat = ages / ages.sum().clamp_min(1e-12)
        a_t = torch.exp(-float(gamma) * a_hat)
        u = torch.log(r.clamp_min(1e-12)) + float(eta) * torch.sqrt(a_t.clamp_min(0.0)) + float(eps)
        u_bar = u.mean()
        p = torch.sigmoid(float(alpha) * (u - u_bar))
        if fixed_size == 'fixed':
            w = (p / p.sum().clamp_min(1e-12))
            idx = torch.multinomial(w, num_samples=min(int(num_extra), len(avail)), replacement=False)
            return [avail[i] for i in idx.tolist()]
        elif fixed_size == 'variable':
            keep = (torch.rand_like(p) < p).nonzero(as_tuple=True)[0].tolist()
            sel = [avail[i] for i in keep]
            if len(sel) > num_extra:
                sel = sel[:num_extra]
            elif len(sel) < num_extra:
                order = torch.argsort(p, descending=True).tolist()
                for i in order:
                    if avail[i] not in sel:
                        sel.append(avail[i])
                    if len(sel) >= num_extra:
                        break
            return sel
        else:
            keep = (torch.rand_like(p) < p).nonzero(as_tuple=True)[0].tolist()
            sel = [avail[i] for i in keep]
            return sel

    @torch.no_grad()
    def sample_stale_views(self, present_rids, labels_B, num_extra):
        """Fig-style: choose stale_rids by p_t(k), then sample cache features."""
        stale_rids = self.pick_extra_rids_fig_sampling(
            present_rids=present_rids,
            labels_B=labels_B,
            num_extra=num_extra,
            fixed_size='a',
        )
        stale_feats = []
        stale_age_Bs = []
        for rid in stale_rids:
            feat_BCT1, age_B = self.sample_BCT1_for_rid(rid, labels_B)
            if feat_BCT1 is None or age_B is None:
                continue
            stale_feats.append(feat_BCT1)
            stale_age_Bs.append(age_B)
            self.stale_count[int(rid)] += 1
        return stale_feats, stale_rids, stale_age_Bs
