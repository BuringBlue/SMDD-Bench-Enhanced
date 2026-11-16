ARG AGENT_IMAGE=smdd-agent:latest
FROM ${AGENT_IMAGE}

# The evaluator inherits the isolated Vina and DiffDock Conda prefixes from the
# agent image. Only trusted scoring dependencies are added to the default
# Python environment here.

# Boltz (structure prediction)
RUN pip install --no-cache-dir boltz==2.2.1 biotite==1.2.0 cuequivariance-torch==0.8.1 cuequivariance-ops-torch-cu12==0.9.1

# ADMET-AI (has incorrect pandas<2.2.0 constraint but works fine with pandas 2.3.3)
RUN pip install --no-cache-dir --no-deps admet-ai==1.2.0
RUN pip install --no-cache-dir chemfunc==1.0.12 chemprop==1.6.1 descriptastorus==2.8.0 Flask==3.1.3 hyperopt==0.2.7 typed-argument-parser==1.11.0

# setuptools <81 needed for pkg_resources (required by hyperopt -> chemprop -> admet-ai)
# Must come AFTER admet-ai since its dependency chain can upgrade setuptools to 82+
RUN pip install --no-cache-dir "setuptools<81"

RUN /opt/vina/bin/python -c "import vina; from meeko import MoleculePreparation" && \
    cd /opt/diffdock/app && \
    /opt/diffdock/bin/python -c "import torch, torch_geometric; assert torch.__version__.startswith('2.1.2+cu121')"

# Verify critical imports
RUN python -c "from openbabel import openbabel; print('OpenBabel OK')"
RUN python -c "from plip.structure.preparation import PDBComplex; print('PLIP OK')"
RUN python -c "from rdkit import Chem; print('RDKit OK')"
RUN python -c "import boltz; print('Boltz OK')"
RUN python -c "from admet_ai import ADMETModel; print('ADMET-AI OK')"

# Copy evaluation code
COPY evaluator/ /eval/code/evaluator/
COPY metrics/ /eval/code/metrics/
COPY smdd_runtime/ /eval/code/smdd_runtime/
COPY evaluator/eval_entrypoint.py /eval/eval_entrypoint.py

WORKDIR /eval
CMD ["python", "/eval/eval_entrypoint.py"]
