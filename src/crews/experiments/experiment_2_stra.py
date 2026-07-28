from __future__ import annotations

import random
from typing import List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from ..config import ExperimentConfig, load_config
from ..data import build_train_test_datasets
from ..training import Trainer


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def run(cfg: ExperimentConfig) -> None:
    set_global_seed(cfg.seed)

    if cfg.limit_num_threads:
        num_threads = torch.get_num_threads()
        torch.set_num_threads(max(num_threads // 2, 1))

    print(
        f"Running experiment with make_S3({cfg.s3_prob}), "
        f"SEED={cfg.seed}, SIM_SEED_TEST={cfg.sim_seed_test}"
    )

    train_dataset, test_dataset = build_train_test_datasets(
        root=cfg.data_root,
        environments=cfg.environments,
        num_classes=cfg.num_classes,
        test_ratio=cfg.test_ratio,
        split_seed=cfg.split_seed,
    )

    train_loader = DataLoader(
        train_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
    )
    test_loader = DataLoader(
        test_dataset, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers,
    )

    trainer = Trainer(cfg, train_loader, test_loader)
    trainer.fit()


def main(argv: Optional[List[str]] = None) -> None:
    cfg = load_config(argv)
    run(cfg)


if __name__ == "__main__":
    main()
