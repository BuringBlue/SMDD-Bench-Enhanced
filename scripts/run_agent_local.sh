#!/usr/bin/env bash
set -euo pipefail
export SMDD_ORACLE_PYTHON="${SMDD_ORACLE_PYTHON:-/opt/oracle/bin/python}"
exec /opt/conda/envs/smdd-agent/bin/python -m agent_harness.run_local "$@"
