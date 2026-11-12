# 当前容器环境快照与恢复说明

本快照于 **2026-11-02 UTC** 从仍在运行的容器中读取，用于在原镜像不可用后重建当前环境。快照不包含 API Key、环境变量、任务结果或模型服务地址。

## 当前运行环境

| 项目 | 当前值 |
| --- | --- |
| 操作系统 | Debian GNU/Linux 12 (bookworm), linux/amd64 |
| 原始基础镜像 | `continuumio/miniconda3:24.11.1-0`（依据仓库 Dockerfile） |
| 宿主内核 | Linux 5.15.0-41-generic |
| Python | 3.10.22 |
| Conda | 24.11.3 |
| pip | 26.2.2 |
| GPU | NVIDIA H800 80 GB |
| NVIDIA 驱动 | 550.144.03 |
| `nvidia-smi` CUDA | 12.4 |
| PyTorch | 2.7.1+cu126，`torch.version.cuda == 12.6` |

环境记录分为三层：

- [`environment_snapshot/conda-explicit-linux-64.txt`](environment_snapshot/conda-explicit-linux-64.txt)：Conda 原生包的精确下载 URL、版本和 build，包含 Python、OpenBabel、RDKit、PLIP 等。
- [`environment_snapshot/pip-explicit.txt`](environment_snapshot/pip-explicit.txt)：当前 `/opt/conda` 中由 pip 安装或覆盖的全部发行包及精确版本。
- [`environment_snapshot/apt-manual.txt`](environment_snapshot/apt-manual.txt)：当前 Debian 中标记为手工安装的系统包。
- [`environment_snapshot/SHA256SUMS`](environment_snapshot/SHA256SUMS)：上述快照和重建 Dockerfile 的完整性校验值。

当前 Python 前缀不是一个由单一求解器生成的“干净”环境：先由 Conda 安装原生科学计算包，随后 pip 覆盖了部分包。例如当前实际导入的是 `numpy==1.26.4`、`scipy==1.13.1`、`scikit-learn==1.6.1`、`PyYAML==6.0.2` 和 `biopython==1.84`。因此恢复时必须先安装 Conda 锁，再以 `--no-deps` 安装 pip 锁，不能交换顺序，也不要运行自动依赖升级。

pip 快照中同时存在若干 CUDA 12 与 CUDA 13 名称的底层发行包；这是当前容器的真实状态，所以精确快照予以保留。实际验证使用的 PyTorch 是 CUDA 12.6 构建，并已在驱动 550.144.03 上确认 `torch.cuda.is_available() == True`。

## 推荐恢复方式：重新构建容器

在仓库根目录执行：

```bash
docker build \
  -f environment_snapshot/Dockerfile \
  -t smdd-evals:snapshot-20261102 \
  .
```

[`environment_snapshot/Dockerfile`](environment_snapshot/Dockerfile) 会从公开的 Miniconda 基础镜像开始，安装系统依赖，按精确 URL 创建 Conda 前缀，再恢复 pip 覆盖层。运行示例：

```bash
docker run --rm -it --gpus all \
  -v "$PWD:/workspace" \
  -v "$PWD/smdd-boltz-cache:/root/.boltz:ro" \
  smdd-evals:snapshot-20261102
```

该镜像只恢复软件环境，不复制仓库、任务数据、运行结果、API Key 或 Boltz 权重。

## 不使用 Docker 时恢复

仅适用于 Linux x86-64；该 Conda 精确锁不能直接用于 ARM 或 macOS。

```bash
SMDD_ENV_PREFIX=/absolute/path/to/smdd-snapshot

conda create -y \
  -p "$SMDD_ENV_PREFIX" \
  --file environment_snapshot/conda-explicit-linux-64.txt

PIP_CONFIG_FILE=/dev/null "$SMDD_ENV_PREFIX/bin/python" -m pip install \
  --no-deps \
  -r environment_snapshot/pip-explicit.txt

export PATH="$SMDD_ENV_PREFIX/bin:$PATH"
export LD_LIBRARY_PATH="$SMDD_ENV_PREFIX/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
```

如果是新的 Debian 12 主机，还需先安装系统工具：

```bash
sudo apt-get update
sed '/^[[:space:]]*#/d; /^[[:space:]]*$/d' \
  environment_snapshot/apt-manual.txt \
  | xargs -r sudo apt-get install -y
```

## 恢复后验证

```bash
python -c "from openbabel import openbabel; from plip.structure.preparation import PDBComplex; from rdkit import Chem; import boltz, torch; from admet_ai import ADMETModel; print(torch.__version__, torch.cuda.is_available())"
nvidia-smi
```

不要把 `pip check` 作为唯一成功标准。`admet-ai==1.2.0` 声明了与当前工作环境不一致的旧版 pandas/torch 约束；仓库的 evaluator Dockerfile 本身也使用 `--no-deps` 绕过了这些错误声明。

## 必须单独保留的文件

`smdd-boltz-cache/` 当前约 5.8 GB，包含 `boltz2_conf.ckpt`、`boltz2_aff.ckpt` 和 `mols/`。这些模型文件不属于 Conda/pip 环境，离线运行前必须单独备份。`tasks/`、`tasks_4_4_final.tar.gz`、`model_evaluations/` 和密钥文件也不包含在环境快照中。

精确锁仍依赖 Anaconda 和 PyPI 将来继续保存相应制品。若要求完全离线、长期可恢复，应尽快在宿主机上把重建后的镜像导出：

```bash
docker save smdd-evals:snapshot-20261102 | gzip > smdd-evals-snapshot-20261102.tar.gz
```

镜像归档可能很大，不建议直接提交到 Git。另有一份更简洁、面向日常开发的环境说明见 [`ENVIRONMENT.md`](ENVIRONMENT.md)；本文件和 `environment_snapshot/` 才是当前容器的精确记录。
