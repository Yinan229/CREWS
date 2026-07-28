#!/bin/bash
# Backward-compatible wrapper. The experiment now lives in the crews package.
# This simply forwards to the new one-click runner.
cd "$(dirname "$0")"
exec bash scripts/run_experiment_2_stra.sh "$@"
