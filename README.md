# CREWS

**CREWS: Collaborative Robust Edge WiFi Sensing with Asynchronous and Incomplete Observations**

The full paper with the supplementary appendix is now publicly available on arXiv:

📄 **Paper:** [https://arxiv.org/abs/2605.30356](https://arxiv.org/abs/2605.30356)

## Installation

conda:

```bash
conda env create -f environment.yaml
conda activate CREWS
```

## Data

Prepare your dataset then set
`data_root` in [configs/experiment_2_jitter.yaml](configs/experiment_2_jitter.yaml)
(or pass `--data_root`).

## Quick start (one-click)

```bash
# Runs seeds 0..4 with the default config
bash scripts/run_experiment_2_stra.sh

# Override via environment variables
SEEDS="0 1 2" S3_PROB=0.9 DEVICE=cuda:0 bash scripts/run_experiment_2_stra.sh
```

The legacy `bash Experiment_2_stra.sh` still works and forwards to the runner.

## Single run

```bash
# Using the src/ layout without installing (run from the project root):
PYTHONPATH=src python -m crews.experiments.experiment_2_jitter \
    --config configs/experiment_2_jitter.yaml \
    --seed 0 --device cuda:0

# If installed with `pip install -e .`, PYTHONPATH is not needed.
```

## Experiments

The dropout setting is chosen in `src/crews/training/trainer.py`:

```python
self.drop_probs_train, self.drop_probs_test = make_S3(cfg.s3_prob)
```

For example, Straggler Reversal is `make_S3(0.9)` with `jitter: false`.

---

---

## 🚧 Work in Progress

**This is not the final release.** The repository is actively being updated.

Remaining experiment settings, scripts, documentation and usage examples are still
being added, and interfaces may change without notice.

Please star or watch this repository to follow the updates. Thanks for your interest!
