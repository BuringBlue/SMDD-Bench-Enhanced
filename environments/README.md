# SMDD runtime environments

- `smdd-agent` runs the LLM agent supervisor and chemistry sandbox.
- `smdd-eval` runs trusted evaluation metrics.
- `/opt/oracle` remains the only environment containing Boltz, CUDA PyTorch,
  and ADMET-AI. Both Conda environments call fixed Oracle entrypoints.

Create the environments with:

```bash
scripts/create_smdd_conda_envs.sh
```

The creation script clones the container's pre-provisioned CPU chemistry stack
from Conda base, then removes the LLM client packages from `smdd-eval`. The YAML
files in this directory record the intended dependency sets. Neither environment
copies packages from `/opt/oracle`.

Launch with:

```bash
scripts/run_agent_local.sh <agent arguments>
scripts/run_eval_local.sh <evaluator arguments>
```

These wrappers select the correct Conda Python and pin Oracle subprocesses to
`/opt/oracle/bin/python` (and evaluator Boltz to `/opt/oracle/bin/boltz`).
