set -euo pipefail

# Move to project root (this script lives in scripts/)
cd "$(dirname "$0")/.."

SEEDS="${SEEDS:-0 1 2 3 4}"
S3_PROB="${S3_PROB:-0.9}"
DEVICE="${DEVICE:-cuda:0}"
CONFIG="${CONFIG:-configs/experiment_2_stra.yaml}"

# Ensure the src/ layout is importable without installation.
export PYTHONPATH="${PYTHONPATH:-}:$(pwd)/src"

for seed in $SEEDS
do
    echo "=========================================="
    echo "Starting Experiment_2_stra: seed=$seed, s3_prob=$S3_PROB, device=$DEVICE"
    echo "=========================================="

    python -m crews.experiments.experiment_2_stra \
        --config "$CONFIG" \
        --s3_prob "$S3_PROB" \
        --seed "$seed" \
        --device "$DEVICE"

    echo "Finished seed=$seed"
    echo ""
done

echo "All experiments completed."
