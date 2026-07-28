"""Experiment configuration handling.
Loads defaults from a YAML file (optional) and allows command-line overrides.
Every hard-coded hyper-parameter from the original monolithic script lives here so
that experiments are fully reproducible and configurable from the CLI.
"""
from __future__ import annotations
import argparse
import os
from dataclasses import dataclass, field, asdict
from typing import List, Optional
try:
    import yaml
except Exception:
    yaml = None

@dataclass
class ExperimentConfig:
    experiment_name: str = "Experiment_2_Jitter_New_2"
    data_root: str = "./data"
    output_dir: str = "outputs"
    environments: List[str] = field(default_factory=lambda: [
        "data_for_yn_V1",
        "data_for_ch_V1",
        "data_for_zj_V1",
        "data_for_wt_V1",
        "data_for_wq_V1",
        "data_for_yl_V1",
    ])
    device: str = "cuda:0"
    limit_num_threads: bool = True
    seed: int = 0
    sim_seed_test: int = 20250909
    split_seed: int = 2025
    train_rng_seed: int = 20251216
    epa_rng_seed: int = 20251217
    num_classes: int = 6
    maxnum_of_clients: int = 8
    batch_size: int = 120
    test_ratio: float = 0.3
    num_workers: int = 0
    client_blocks: List[int] = field(default_factory=lambda: [1, 1, 1, 1, 1, 1, 1, 1])
    lr: float = 1e-3
    epochs: int = 100
    s1_prob: float = 0.9
    s3_prob: float = 0.9
    fix_test_missing: bool = True
    num_batches_keep: int = 5
    stale_budget: int = 2
    lambda_decay: float = 0.4
    normalize_stale_loss: bool = True
    jitter: bool = False
    jitter_swap_prob: float = 0.3
    epa_interval: int = 4
    epa_mu: float = 1.0
    k_align: str = "all"

    def to_dict(self) -> dict:
        return asdict(self)

def _load_yaml(path: str) -> dict:
    if yaml is None:
        raise RuntimeError(
            "PyYAML is required to load config files. Install it via `pip install pyyaml`."
        )
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run CREWS Experiment_2_Jitter_New_2"
    )
    parser.add_argument("--config", type=str, default=None,
                        help="Path to a YAML config file with default values.")
    parser.add_argument("--experiment_name", type=str, default=None)
    parser.add_argument("--data_root", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None,
                        help="Global random seed (0-5)")
    parser.add_argument("--sim_seed_test", type=int, default=None,
                        help="Test mask & shuffle seed")
    parser.add_argument("--s1_prob", type=float, default=None,
                        help="Probability factor for make_S1 (0.1 to 0.9)")
    parser.add_argument("--s3_prob", type=float, default=None,
                        help="Probability factor for make_S3 (0.1 to 0.9)")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--epa_interval", type=int, default=None)
    parser.add_argument("--epa_mu", type=float, default=None,
                        help="Elastic (EMA) blend factor toward the global average (1.0 = full overwrite)")
    parser.add_argument("--k_align", type=str, default=None,
                        help="Client selection for EPA alignment ('all' = align every client)")
    parser.add_argument("--stale_budget", type=int, default=None)
    parser.add_argument("--lambda_decay", type=float, default=None)
    parser.add_argument("--normalize_stale_loss", type=lambda x: x.lower() in ("1", "true", "yes"),
                        default=None, help="Normalize stale loss by staleness weight sum")
    parser.add_argument("--jitter", type=lambda x: x.lower() in ("1", "true", "yes"),
                        default=None, help="Enable view-swap jitter augmentation")
    parser.add_argument("--jitter_swap_prob", type=float, default=None,
                        help="Probability of swapping two views when jitter is enabled")
    parser.add_argument("--num_batches_keep", type=int, default=None)
    return parser

def load_config(argv: Optional[List[str]] = None) -> ExperimentConfig:
    """Build an :class:`ExperimentConfig` from defaults, an optional YAML file and CLI overrides."""
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    values: dict = {}
    if args.config:
        values.update(_load_yaml(args.config))
    override_keys = [
        "experiment_name", "data_root", "output_dir", "device", "seed",
        "sim_seed_test", "s1_prob", "s3_prob", "epochs", "batch_size", "lr",
        "num_workers", "epa_interval", "epa_mu", "stale_budget",
        "lambda_decay", "normalize_stale_loss",
        "jitter", "jitter_swap_prob", "num_batches_keep", "k_align",
    ]
    for key in override_keys:
        val = getattr(args, key, None)
        if val is not None:
            values[key] = val
    valid_fields = set(ExperimentConfig().to_dict().keys())
    filtered = {k: v for k, v in values.items() if k in valid_fields}
    return ExperimentConfig(**filtered)
