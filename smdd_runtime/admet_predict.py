"""Fixed ADMET-AI subprocess entrypoint for trusted Oracle execution."""

from __future__ import annotations

import argparse
import contextlib
import functools
import io
import json

import torch


COLUMN_MAP = {
    "hERG": "herg",
    "BBB_Martins": "bbb",
    "Solubility_AqSolDB": "solubility",
    "CYP3A4_Veith": "cyp3a4",
    "Clearance_Hepatocyte_AZ": "clearance",
    "Caco2_Wang": "caco2",
    "PPBR_AZ": "ppb",
    "AMES": "ames",
}


def predict(smiles: str) -> dict[str, float]:
    original_load = torch.load
    torch.load = functools.partial(original_load, weights_only=False)
    try:
        from admet_ai import ADMETModel

        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            predictions = ADMETModel().predict(smiles=smiles)
    finally:
        torch.load = original_load

    result = {}
    for dataset_name, property_name in COLUMN_MAP.items():
        value = predictions[dataset_name]
        result[property_name] = float(value.iloc[0] if hasattr(value, "iloc") else value)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", required=True)
    args = parser.parse_args()
    print(json.dumps(predict(args.smiles), sort_keys=True))


if __name__ == "__main__":
    main()
