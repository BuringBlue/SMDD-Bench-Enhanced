# Conda environment

> 当前运行容器的精确 Conda/pip/系统包快照及镜像重建方法见
> [`CURRENT_CONTAINER_ENVIRONMENT.md`](CURRENT_CONTAINER_ENVIRONMENT.md)。本文件描述的是较精简的宿主开发环境。

The host-side SMDD-Bench environment is isolated in the `smdd-bench` Conda
environment. Creating it does not download the benchmark tasks or model
weights.

```bash
conda env create -f environment.yml
PIP_CONFIG_FILE=/dev/null conda run -n smdd-bench python -m pip install \
  --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  -r requirements-conda.txt
PIP_CONFIG_FILE=/dev/null conda run -n smdd-bench python -m pip install \
  --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  litellm==1.83.3
PIP_CONFIG_FILE=/dev/null conda run -n smdd-bench python -m pip install \
  --index-url https://mirrors.tuna.tsinghua.edu.cn/pypi/web/simple \
  --no-deps admet_ai==1.4.0
```

`PIP_CONFIG_FILE=/dev/null` avoids the machine-wide unreachable NVIDIA extra
index; it does not modify the user's global Pip configuration.

`admet-ai==1.4.0` incorrectly declares `pandas<2.2`, while Boltz requires
`pandas>=2.2.2` and this repository pins `pandas==2.3.3`. Installing ADMET-AI
without dependency resolution is the compatibility workaround already used by
the repository's Dockerfiles; its actual runtime dependencies are explicitly
listed in `requirements-conda.txt`.

ADMET-AI also declares `torch==2.5.0`; the host environment intentionally uses
the CUDA-12-compatible `torch==2.7.1` required by the current Boltz and
Cuequivariance stack. ADMET-AI imports successfully with this combination, and
the official evaluator keeps it isolated in its own container.

LiteLLM is also installed separately because it pins `click==8.1.8` and
`python-dotenv==1.0.1`, while the combined upstream requirements request
different patch releases. These packages run in different processes in the
benchmark; the host environment keeps LiteLLM's patch releases, while the
official evaluator remains isolated in its own container.

Activate it for host-side harness commands:

```bash
conda activate smdd-bench
```

Quick verification:

```bash
conda run -n smdd-bench python -c \
  "from openbabel import openbabel; from plip.structure.preparation import PDBComplex; from rdkit import Chem; import boltz, litellm; print('SMDD-Bench environment OK')"
```

The benchmark itself launches agent and evaluator containers. Those images
remain separate from this host environment and require Docker plus the task
archive before a full evaluation can run.
