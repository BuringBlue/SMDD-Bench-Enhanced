FROM continuumio/miniconda3:24.11.1-0

# Triton (via PyTorch/Boltz) needs a C compiler for JIT kernel compilation
RUN apt-get update && apt-get install -y gcc g++ libc6-dev && rm -rf /var/lib/apt/lists/*

# Install OpenBabel + RDKit + PLIP + science stack via conda
RUN conda install -y -c conda-forge \
    python=3.10 \
    rdkit=2025.09.5 \
    openbabel=3.1.1 \
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

# Pre-download Boltz2 weights so they're cached in the image
# (the agent container has no general internet access at runtime)
RUN mkdir -p /root/.boltz && \
    /opt/oracle/bin/python -c "\
from boltz.main import download_boltz2; \
from pathlib import Path; \
download_boltz2(Path.home() / '.boltz')"

WORKDIR /workspace
CMD ["/bin/bash"]
