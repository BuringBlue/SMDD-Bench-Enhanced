FROM continuumio/miniconda3:24.11.1-0

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

# Conda-built native extensions (e.g. matplotlib) link against conda's
# libstdc++ (6.0.34). Without this, the older system libstdc++ gets
# loaded instead, causing missing CXXABI symbol errors.
ENV LD_LIBRARY_PATH=/opt/conda/lib

RUN pip install --no-cache-dir --upgrade pip

# Boltz (structure prediction)
RUN pip install --no-cache-dir boltz==2.2.1 biotite==1.2.0 cuequivariance-torch==0.8.1 cuequivariance-ops-torch-cu12==0.9.1

# ADMET-AI (has incorrect pandas<2.2.0 constraint but works fine with pandas 2.3.3)
RUN pip install --no-cache-dir --no-deps admet-ai==1.2.0
RUN pip install --no-cache-dir chemfunc==1.0.12 chemprop==1.6.1 descriptastorus==2.8.0 Flask==3.1.3 hyperopt==0.2.7 typed-argument-parser==1.11.0

# setuptools <81 needed for pkg_resources (required by hyperopt -> chemprop -> admet-ai)
# Must come AFTER admet-ai since its dependency chain can upgrade setuptools to 82+
RUN pip install --no-cache-dir "setuptools<81"

# Verify critical imports
RUN python -c "from openbabel import openbabel; print('OpenBabel OK')"
RUN python -c "from plip.structure.preparation import PDBComplex; print('PLIP OK')"
RUN python -c "from rdkit import Chem; print('RDKit OK')"
RUN python -c "import boltz; print('Boltz OK')"
RUN python -c "from admet_ai import ADMETModel; print('ADMET-AI OK')"

# Copy evaluation code
COPY evaluator/ /eval/code/evaluator/
COPY metrics/ /eval/code/metrics/
COPY evaluator/eval_entrypoint.py /eval/eval_entrypoint.py

WORKDIR /eval
CMD ["python", "/eval/eval_entrypoint.py"]
