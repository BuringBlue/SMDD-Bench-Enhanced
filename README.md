<h1 align="center">SMDD-Bench: Can LLMs Solve Real-World Small Molecule Drug Design Tasks?</h1>

<p align="center">Kevin Han, Renfei Zhang, Kathy Wei, Hamed Mahdavi, Niloofar Mireshghallah, Amir Farimani</p>

SMDD-Bench is a benchmark for evaluating LLM agents on small molecule drug design (SMDD) tasks. It is small-molecule drug design's analog to SWE-Bench: agents are given a task and a fixed toolkit (Python interpreter, structure prediction models, RDKit, etc.) and must produce a verifiable artifact. Tasks are designed to be computationally simple to verify (via ML models or RDKit) but scientifically challenging to solve.

## Task Types

The benchmark covers 5 task types:

1. **2D Pharmacophore Identification** — given actives/inactives for a target, derive a 2D pharmacophore that discriminates them.
2. **Interaction Point Discovery** — given a binding pocket, predict the 3 most conserved 3D pharmacophore points across diverse binders.
3. **Scaffold Hopping** — given an active, design a molecule with a substantially different scaffold that preserves the pharmacophore and pocket interactions.
4. **Lead Optimization** — improve a molecule along sampled objectives while holding other properties and hard constraints fixed. SMDD-Bench tasks go up to 5 simultaneous optimization objectives and 4 hold-constant constraints along with several hard constraints.
5. **Fragment Assembly** — assemble 1–2 fragments (with 3D poses) into a single drug-like binder.

## Repository Layout

- `tasks/` — the 500 benchmark task instances (download separately, see below).
- `evaluator/` — evaluation harness. Each task's `task.yaml` specifies the metrics, expected outputs, and pass conditions used to score a run.
- `metrics/` — verification metrics callable from `task.yaml`.
- `agent_harness/` — the minimalist LLM agent loop, tool definitions, and runner.
- `agent_env/` — Dockerfile for the sandboxed agent environment.

## Setup

1. Create the conda environment named `smdd-bench` and install the system tools (these are not pip-installable):
   - [Open Babel](https://openbabel.org/)
   - [PLIP](https://github.com/pharmai/plip)
2. Install Python dependencies:
   ```
   pip install -r requirements.txt
   ```
3. Build the agent and evaluator Docker images from `agent_env/Dockerfile` and `evaluator/Dockerfile`.

## Downloading the Tasks

The `tasks/` directory is gitignored. Download `tasks_4_4_final.tar.gz` from Hugging Face and extract it into the repo root:

- Dataset: <https://huggingface.co/datasets/extremelyconfused/smdd-bench>
- File: `tasks_4_4_final.tar.gz`


After extraction there should be a `tasks/` directory containing 500 subfolders (one per task instance), each with a `task.yaml` and any supporting files needed for verification.

## Important Note on Binding Affinity

Throughout the benchmark, `binding_affinity` values are the raw Boltz2-predicted log10(IC50) derived from an IC50 measured in μM. Therefore, task descriptions may refer to lowering the Boltz2-generated `binding_affinity` field, leading to stronger binding.

## Running

The agent and evaluation harnesses run on a single host with one or more NVIDIA GPUs. Build the Docker images once:

```bash
docker build -t smdd-agent agent_env/
docker build -t smdd-evals evaluator/
```

### `agent_harness/run.py` — running the agent

Launches one Docker container per task (one task per GPU at a time). Per-task results go to `results/{run-name}/{task_id}/`, and an `agent_summary.json` is written at the end. Before running, edit `agent_harness/config.py` to set the model, turn/token budgets, and tool limits; LiteLLM API keys go in a `.env` file at the repo root. `use_responses_api` refers to whether or not to use LiteLLM's OpenAI responses API. We recommend only setting this to true for the OpenAI models.

```bash
# Run all tasks on all visible GPUs
python -m agent_harness.run --run-name my_run

# Run on a specific set of GPUs
python -m agent_harness.run --run-name my_run --gpus 0,1,2,3

# Run a single task or a few specific tasks
python -m agent_harness.run --run-name my_run --tasks smdd_001_0001 smdd_002_0003

# Run from a file (one task ID per line)
python -m agent_harness.run --run-name my_run --task-file my_tasks.txt

# Override the model from AgentConfig
python -m agent_harness.run --run-name my_run --model gpt-5.4
```

Flags:

| Flag | Description |
| --- | --- |
| `--run-name` | **Required.** Results go to `{results-dir}/{run-name}/`. |
| `--tasks-dir` | Path to tasks directory (default: `tasks`). |
| `--results-dir` | Output directory for results (default: `results`). |
| `--tasks` | Run specific task IDs (space-separated). |
| `--task-file` | File with task IDs to run, one per line. |
| `--model` | Override the model in `AgentConfig`. |
| `--gpus` | Comma-separated GPU indices, e.g. `0,1,2,3`. Default: all GPUs visible to `nvidia-smi`. |
| `--workers` | Parallel worker processes. Default: number of GPUs. |

Notes:
- Containers run on an internal Docker network with the MSA server reachable only through an nginx TCP proxy sidecar (see `agent_harness/network.py`).
- At least 1 GPU is required. If none are detected, the run aborts.

### `evaluator/evaluate.py` — scoring agent outputs

Launches one `smdd-evals` Docker container per task to score the agent's submitted file against the metrics in `task.yaml`. Per-task `result.json` files go to `results/{run-name}/{task_id}/`, and `summary.json` is written at the end.

```bash
# Evaluate every task in tasks/
python -m evaluator.evaluate --run-name my_eval \
    --agent-outputs results/my_run --all

# Evaluate only the tasks the agent actually produced output for
python -m evaluator.evaluate --run-name my_eval \
    --agent-outputs results/my_run --all-in-folder

# Evaluate a specific subset
python -m evaluator.evaluate --run-name my_eval \
    --agent-outputs results/my_run --tasks smdd_001_0001 smdd_003_0007

# Evaluate all tasks of a single type (e.g. type 4 = Lead Optimization)
python -m evaluator.evaluate --run-name my_eval \
    --agent-outputs results/my_run --type 4
```

Flags:

| Flag | Description |
| --- | --- |
| `--run-name` | **Required.** Name of the evaluation run. |
| `--agent-outputs` | **Required.** Directory containing the per-task agent outputs (e.g. `results/my_run`). |
| `--tasks-dir` | Directory with task definitions (default: `tasks`). |
| `--docker-image` | Evaluator image name (default: `smdd-evals`). |
| `--gpus` | Comma-separated GPU indices. Default: all visible GPUs; falls back to sequential CPU-only if none are detected (most type-3/4/5 metrics need a GPU). |
| `--workers` | Parallel worker processes. Default: number of GPUs. |

Exactly one task-selection flag is required (mutually exclusive):

| Flag | Description |
| --- | --- |
| `--all` | Every task in `--tasks-dir`. |
| `--all-in-folder` | Every task present in `--agent-outputs`. |
| `--tasks` | Specific task IDs (space-separated). |
| `--task-file` | File with task IDs, one per line. |
| `--type N` | All tasks of type number `N` (matches the `smdd_{N:03d}_*` prefix). |

See `type_docs/agent_harness.md` and `type_docs/evaluation_harness.md` for the per-task output format, `task.yaml` schema, and metric reference.

## Citing Us

If you found the work interesting, you can cite us here:

```bibtex
@misc{han2026smddbench,
      title={SMDD-Bench: Can LLMs Solve Real-World Small Molecule Drug Design Tasks?}, 
      author={Kevin Han and Renfei Zhang and Kathy Wei and Hamed Mahdavi and Niloofar Mireshghallah and Amir Farimani},
      year={2026},
      eprint={2605.21740},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2605.21740}, 
}
```

## Reporting an issue

If you find a bug or have a feature request, please raise a Github issue or email kevinhan@cmu.edu.

## Questions or Collaborations

If you have any questions or want to collaborate, feel free to reach out at kevinhan@cmu.edu!