FROM continuumio/miniconda3:24.11.1-0 AS conda-base

# DiffDock's upstream environment initially compiles OpenFold against CUDA 11.7.
# A later layer replaces the inference stack with the H800-compatible CUDA 12.1
# wheels used by this benchmark.
FROM nvidia/cuda:11.7.1-devel-ubuntu22.04 AS diffdock-builder

ARG DIFFDOCK_VERSION=v1.1.3
ARG DIFFDOCK_MODELS_SHA256=5a95b6a1555be47ab1d6f0a8ffd25152f7fe32f5956005bb821e13e7a37d4a3d

COPY --from=conda-base /opt/conda /opt/conda
ENV PATH=/opt/conda/bin:${PATH}

RUN apt-get update && apt-get install -y gcc g++ git curl unzip && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch "${DIFFDOCK_VERSION}" \
        https://github.com/gcorso/DiffDock.git /tmp/DiffDock && \
    conda env create -p /opt/diffdock -f /tmp/DiffDock/environment.yml && \
    mv /tmp/DiffDock /opt/diffdock/app && \
    conda clean -afy

# Torch 1.13/CUDA 11.7 from the upstream v1.1.3 environment cannot execute on
# NVIDIA H800 GPUs. OpenFold is not used when DiffDock receives a receptor PDB,
# so remove it and install the tested Torch/PyG CUDA 12.1 inference stack.
RUN /opt/diffdock/bin/pip uninstall -y \
        openfold torch torch-cluster torch-scatter torch-sparse torch-spline-conv && \
    /opt/diffdock/bin/pip install --no-cache-dir \
        --index-url https://download.pytorch.org/whl/cu121 \
        "torch==2.1.2" && \
    /opt/diffdock/bin/pip install --no-cache-dir \
        --find-links https://data.pyg.org/whl/torch-2.1.0+cu121.html \
        "torch-cluster==1.6.3+pt21cu121" \
        "torch-scatter==2.1.2+pt21cu121" \
        "torch-sparse==0.6.18+pt21cu121" \
        "torch-spline-conv==1.2.2+pt21cu121"

RUN mkdir -p /opt/diffdock/app/workdir/v1.1 && \
    curl -L --fail --retry 3 \
        https://github.com/gcorso/DiffDock/releases/download/v1.1/diffdock_models.zip \
        -o /tmp/diffdock_models.zip && \
    echo "${DIFFDOCK_MODELS_SHA256}  /tmp/diffdock_models.zip" | sha256sum -c - && \
    unzip -q /tmp/diffdock_models.zip -d /opt/diffdock/app/workdir/v1.1 && \
    rm /tmp/diffdock_models.zip

# Runtime networking is intentionally restricted, so cache the ESM-2 weights
# and DiffDock's SO(2)/SO(3) lookup tables while building the image.
RUN mkdir -p /opt/diffdock/cache/torch /opt/diffdock/cache/xdg && \
    TORCH_HOME=/opt/diffdock/cache/torch \
        /opt/diffdock/bin/python -c \
        "from esm import pretrained; pretrained.load_model_and_alphabet('esm2_t33_650M_UR50D')" && \
    cd /opt/diffdock/app && \
    TORCH_HOME=/opt/diffdock/cache/torch XDG_CACHE_HOME=/opt/diffdock/cache/xdg \
        /opt/diffdock/bin/python utils/precompute_series.py


FROM conda-base

ARG VINA_VERSION=1.2.7
ARG MEEKO_VERSION=0.8.0
ARG VINA_GPU_COMMIT=180272b8a5265d6ed9664178345933cebe2cd349
ARG VINA_GPU_SHA256=c5d8ca922f6ca4130dcc809aa620e1120be24306ca475046faef5ab7ccbf6aa5

# Triton (via PyTorch/Boltz) needs a C compiler for JIT kernel compilation.
# Vina-GPU uses the NVIDIA OpenCL implementation supplied by the container
# runtime; only the vendor-neutral loader and headers are installed here.
RUN apt-get update && apt-get install -y \
        curl gcc g++ libc6-dev patch \
        libboost-filesystem-dev libboost-program-options-dev \
        libboost-system-dev libboost-thread-dev \
        ocl-icd-opencl-dev opencl-headers && \
    rm -rf /var/lib/apt/lists/*

# Install OpenBabel + RDKit + PLIP + science stack via conda
RUN conda install -y -c conda-forge \
    python=3.11 \
    rdkit=2026.03.4 \
    openbabel=3.2.1 \
    plip=3.0.0 \
    numpy=2.2.6 \
    pandas=2.3.3 \
    scikit-learn=1.7.2 \
    scipy=1.15.2 \
    pyyaml=6.0.3 \
    biopython=1.86 \
    lxml=6.0.2 \
    && conda clean -afy

RUN pip install --no-cache-dir --upgrade pip

# Agent-accessible packages (no oracle tools here)
RUN pip install --no-cache-dir biotite==1.2.0

# Oracle tools — isolated venv, NOT accessible to the agent's default Python
RUN python -m venv /opt/oracle
RUN /opt/oracle/bin/pip install --no-cache-dir boltz==2.2.1 biotite==1.2.0 cuequivariance-torch==0.8.1 cuequivariance-ops-torch-cu12==0.9.1
# admet-ai has an incorrect pandas<2.2.0 constraint but works fine with pandas 2.3.3
# (pulled by boltz). Install with --no-deps, then install its real deps separately.
RUN /opt/oracle/bin/pip install --no-cache-dir --no-deps admet-ai==1.2.0
RUN /opt/oracle/bin/pip install --no-cache-dir chemfunc==1.0.12 chemprop==1.6.1 descriptastorus==2.8.0 Flask==3.1.3 hyperopt==0.2.7 typed-argument-parser==1.11.0
# setuptools <81 needed for pkg_resources (required by hyperopt -> chemprop -> admet-ai)
# Must come AFTER admet-ai since its dependency chain can upgrade setuptools to 82+
RUN /opt/oracle/bin/pip install --no-cache-dir "setuptools<81"

# AutoDock Vina oracle environment. Keep it separate from both the agent's
# default Python and Boltz/ADMET so Meeko/RDKit upgrades cannot affect them.
RUN conda create -y -p /opt/vina -c conda-forge \
    python=3.10 pip rdkit scipy gemmi tqdm && \
    /opt/vina/bin/pip install --no-cache-dir \
        "vina==${VINA_VERSION}" "meeko==${MEEKO_VERSION}" && \
    conda clean -afy

# Vina-GPU 2.1 has no Conda package. Build the pinned OpenCL backend and place
# it inside the /opt/vina prefix; the Conda environment remains responsible for
# RDKit/Meeko ligand and receptor preparation.
COPY vina-gpu-nvidia.patch /tmp/vina-gpu-nvidia.patch
RUN mkdir -p /tmp/vina-gpu /opt/vina/share/vina-gpu && \
    curl -L --fail --retry 3 \
        "https://github.com/DeltaGroupNJUPT/Vina-GPU-2.1/archive/${VINA_GPU_COMMIT}.tar.gz" \
        -o /tmp/vina-gpu.tar.gz && \
    echo "${VINA_GPU_SHA256}  /tmp/vina-gpu.tar.gz" | sha256sum -c - && \
    tar -xzf /tmp/vina-gpu.tar.gz -C /tmp/vina-gpu --strip-components=1 && \
    cd /tmp/vina-gpu/AutoDock-Vina-GPU-2.1 && \
    patch -p1 < /tmp/vina-gpu-nvidia.patch && \
    g++ -O3 -o /opt/vina/bin/AutoDock-Vina-GPU-2-1 \
        -I./lib -I./OpenCL/inc ./main/main.cpp ./lib/*.cpp ./OpenCL/src/wrapcl.cpp \
        -lboost_program_options -lboost_system -lboost_filesystem -lboost_thread \
        -lOpenCL -lstdc++fs -lm -lpthread \
        -DOPENCL_2_0 -DNVIDIA_PLATFORM \
        -DSMALL_BOX -DBOOST_TIMER_ENABLE_DEPRECATED -DNDEBUG \
        -DBUILD_KERNEL_FROM_SOURCE && \
    cp -a OpenCL /opt/vina/share/vina-gpu/OpenCL && \
    cp /tmp/vina-gpu/LICENSE /opt/vina/share/vina-gpu/LICENSE && \
    rm -rf /tmp/vina-gpu /tmp/vina-gpu.tar.gz /tmp/vina-gpu-nvidia.patch

COPY nvidia-opencl.icd /opt/vina/share/vina-gpu/nvidia.icd

# Copy the already-built Conda prefix to the same absolute path used at build
# time, together with the pinned DiffDock application and cached assets.
COPY --from=diffdock-builder /opt/diffdock /opt/diffdock

# Fail the image build early if either isolated environment is incomplete.
RUN /opt/vina/bin/python -c \
        "from importlib.metadata import version; assert version('vina') == '${VINA_VERSION}'; assert version('meeko') == '${MEEKO_VERSION}'" && \
    /opt/vina/bin/AutoDock-Vina-GPU-2-1 --version && \
    cd /opt/diffdock/app && \
    /opt/diffdock/bin/python -c \
        "import torch, torch_cluster, torch_scatter, torch_sparse, torch_spline_conv; assert torch.__version__.startswith('2.1.2+cu121')"

# Fixed trusted entrypoints called by the host-side ToolExecutor.
COPY oracle_tools/ /opt/smdd-tools/

# Pre-download Boltz2 weights so they're cached in the image
# (the agent container has no general internet access at runtime)
RUN mkdir -p /root/.boltz && \
    /opt/oracle/bin/python -c "\
from boltz.main import download_boltz2; \
from pathlib import Path; \
download_boltz2(Path.home() / '.boltz')"

WORKDIR /workspace
CMD ["/bin/bash"]
