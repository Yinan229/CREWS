"""Elastic Parameter Alignment (EPA) utilities for the split (edge + server-side) sub-models.
EPA periodically aligns each client's sub-model parameters toward a global average
via an elastic (EMA) blend, rather than a hard overwrite.
"""
from __future__ import annotations
import copy
import random
import torch
from ..simulation.dropout import bernoulli_drop_rids

def epa_aggregate(w):
    """Align a list of state-dicts positionally into the EPA global state."""
    epa_global = copy.deepcopy(w[0])
    temp = 0
    for k in epa_global.keys():
        for i in range(1, len(w)):
            k_c = list(w[i].keys())[temp]
            epa_global[k] += w[i][k_c]
        epa_global[k] = torch.div(epa_global[k], len(w))
        temp = temp + 1
    return epa_global

def get_active_rids_for_epa_round(maxnum_of_clients, rng: random.Random, drop_probs_epa):
    active_rids, _ = bernoulli_drop_rids(
        maxnum_of_clients=maxnum_of_clients,
        rng=rng,
        drop_probs=drop_probs_epa,
        ensure_at_least_one_present=True,
    )
    return active_rids

@torch.no_grad()
def _ema_blend(local_sd, global_sd, mu: float):
    blended = {}
    for k, g in global_sd.items():
        l = local_sd[k]
        if torch.is_floating_point(g):
            blended[k] = (1.0 - mu) * l + mu * g
        else:
            blended[k] = g.clone()
    return blended

@torch.no_grad()
def epa_align_bottom_plus_edgeon(
    edge_jetson_model,
    edge_split_model,
    maxnum_of_clients: int,
    rids=None,
    strict_load: bool = False,
    mu: float = 1.0,
):
    if rids is None:
        rids = list(range(maxnum_of_clients))
    else:
        rids = list(rids)
    if len(rids) <= 1:
        return None
    combined_list = []
    canonical_keys = None
    for rid in rids:
        edge_sd = edge_jetson_model[str(rid)].state_dict()
        server_sd = edge_split_model[str(rid)].state_dict()
        combined = {**edge_sd, **server_sd}
        if canonical_keys is None:
            canonical_keys = list(combined.keys())
        else:
            if set(combined.keys()) != set(canonical_keys):
                raise RuntimeError(
                    f"rid={rid} 的 edge+server key 集合与 canonical 不一致，无法做合并对齐。"
                )
        combined_list.append(combined)
    paired_state_dicts = []
    for combined in combined_list:
        ordered = {k: combined[k] for k in canonical_keys}
        paired_state_dicts.append(ordered)
    epa_global = epa_aggregate(paired_state_dicts)
    for rid in rids:
        edge_keys = list(edge_jetson_model[str(rid)].state_dict().keys())
        server_keys = list(edge_split_model[str(rid)].state_dict().keys())
        for k in edge_keys:
            if k not in epa_global:
                raise RuntimeError(f"epa_global 缺少 edge key: {k} (rid={rid})")
        for k in server_keys:
            if k not in epa_global:
                raise RuntimeError(f"epa_global 缺少 server key: {k} (rid={rid})")
        edge_global = {k: epa_global[k] for k in edge_keys}
        server_global = {k: epa_global[k] for k in server_keys}
        if mu >= 1.0:
            edge_new = edge_global
            server_new = server_global
        else:
            edge_local = edge_jetson_model[str(rid)].state_dict()
            server_local = edge_split_model[str(rid)].state_dict()
            edge_new = _ema_blend(edge_local, edge_global, mu)
            server_new = _ema_blend(server_local, server_global, mu)
        edge_jetson_model[str(rid)].load_state_dict(edge_new, strict=strict_load)
        edge_split_model[str(rid)].load_state_dict(server_new, strict=strict_load)
    return epa_global
