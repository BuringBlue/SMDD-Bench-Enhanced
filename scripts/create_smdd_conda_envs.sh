#!/usr/bin/env bash
set -euo pipefail

conda_root="${CONDA_ROOT:-/opt/conda}"
conda_bin="${conda_root}/bin/conda"
agent_path="${conda_root}/envs/smdd-agent"
eval_path="${conda_root}/envs/smdd-eval"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repo_root="$(cd "${script_dir}/.." && pwd)"

create_if_missing() {
    local target_path="$1"
    local environment_file="$2"

    if [[ -x "${target_path}/bin/python" ]]; then
        echo "Keeping existing environment: ${target_path}"
        return
    fi

    CONDA_NO_PLUGINS=true "${conda_bin}" env create -y \
        --prefix "${target_path}" \
        --file "${environment_file}"
}

# Resolve each environment independently. This intentionally avoids cloned
# conda-meta history and allows Conda to select a coherent ICU/libpq/libxml2 ABI
# stack for RDKit 2026.03.4 and OpenBabel 3.2.1.
create_if_missing "${agent_path}" "${repo_root}/environments/smdd-agent.yml"
create_if_missing "${eval_path}" "${repo_root}/environments/smdd-eval.yml"

"${agent_path}/bin/python" -c \
    "import biotite, boto3, litellm, yaml; from rdkit import rdBase; from openbabel import openbabel; assert rdBase.rdkitVersion == '2026.03.4'; assert openbabel.OBReleaseVersion() == '3.2.1'"
"${eval_path}/bin/python" -c \
    "import importlib.util, biotite, yaml; from rdkit import rdBase; from openbabel import openbabel; assert rdBase.rdkitVersion == '2026.03.4'; assert openbabel.OBReleaseVersion() == '3.2.1'; assert importlib.util.find_spec('litellm') is None"

echo "SMDD Conda environments are ready."
