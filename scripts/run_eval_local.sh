#!/usr/bin/env bash
set -euo pipefail
export SMDD_ORACLE_PYTHON="${SMDD_ORACLE_PYTHON:-/opt/oracle/bin/python}"
export SMDD_BOLTZ_EXECUTABLE="${SMDD_BOLTZ_EXECUTABLE:-/opt/oracle/bin/boltz}"
exec /opt/conda/envs/smdd-eval/bin/python -m evaluator.evaluate_local "$@"
