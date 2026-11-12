"""Deterministic medicinal-chemistry validation used by the agent tool."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Crippen, Descriptors, Lipinski, RDConfig, rdFingerprintGenerator
from rdkit.Chem import rdMolDescriptors
from rdkit.Chem.FilterCatalog import FilterCatalog, FilterCatalogParams


RDLogger.DisableLog("rdApp.*")


def _catalog_matches(mol, names: tuple[str, ...]) -> list[str]:
    params = FilterCatalogParams()
    catalogs = FilterCatalogParams.FilterCatalogs
    for name in names:
        params.AddCatalog(getattr(catalogs, name))
    catalog = FilterCatalog(params)
    return sorted({entry.GetDescription() for entry in catalog.GetMatches(mol)})


def _reference_molecule(path: str | None):
    if not path or not Path(path).is_file():
        return None
    supplier = Chem.SDMolSupplier(path, removeHs=True, sanitize=True)
    return next((mol for mol in supplier if mol is not None), None)


def validate_molecule(smiles: str, reference_path: str | None = "reference.sdf") -> dict:
    result = {
        "input_smiles": smiles,
        "valid": False,
        "sanitizable": False,
    }
    try:
        mol = Chem.MolFromSmiles(smiles, sanitize=True)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result
    if mol is None:
        result["error"] = "RDKit could not parse and sanitize the SMILES"
        return result

    try:
        Chem.SanitizeMol(mol)
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result.update({
        "valid": True,
        "sanitizable": True,
        "canonical_smiles": Chem.MolToSmiles(mol),
    })

    fragments = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    radical_electrons = sum(atom.GetNumRadicalElectrons() for atom in mol.GetAtoms())
    descriptors = {
        "molecular_weight": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "tpsa": rdMolDescriptors.CalcTPSA(mol),
        "hbd": rdMolDescriptors.CalcNumHBD(mol),
        "hba": rdMolDescriptors.CalcNumHBA(mol),
        "rotatable_bonds": Lipinski.NumRotatableBonds(mol),
        "formal_charge": Chem.GetFormalCharge(mol),
    }

    import sys

    sa_dir = str(Path(RDConfig.RDContribDir) / "SA_Score")
    if sa_dir not in sys.path:
        sys.path.append(sa_dir)
    import sascorer

    sa_score = float(sascorer.calculateScore(mol))
    pains = _catalog_matches(mol, ("PAINS_A", "PAINS_B", "PAINS_C"))
    brenk = _catalog_matches(mol, ("BRENK",))
    nih = _catalog_matches(mol, ("NIH",))

    reference = _reference_molecule(reference_path)
    similarity = None
    if reference is not None:
        generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        similarity = float(DataStructs.TanimotoSimilarity(
            generator.GetFingerprint(reference),
            generator.GetFingerprint(mol),
        ))

    checks = {
        "single_fragment": len(fragments) == 1,
        "no_radicals": radical_electrons == 0,
        "molecular_weight_lt_600": descriptors["molecular_weight"] < 600,
        "logp_between_minus_1_and_5": -1 < descriptors["logp"] < 5,
        "tpsa_lt_140": descriptors["tpsa"] < 140,
        "hbd_lte_5": descriptors["hbd"] <= 5,
        "hba_lte_10": descriptors["hba"] <= 10,
        "rotatable_bonds_lte_10": descriptors["rotatable_bonds"] <= 10,
        "formal_charge_between_minus_2_and_2": -2 <= descriptors["formal_charge"] <= 2,
        "sa_score_lt_4_5": sa_score < 4.5,
        "pains_pass": not pains,
        "brenk_pass": not brenk,
        "nih_pass": not nih,
    }
    if similarity is not None:
        checks["tanimoto_morgan_radius_2_2048_gte_0_7"] = similarity >= 0.7

    result.update({
        "fragment_count": len(fragments),
        "radical_electrons": radical_electrons,
        "descriptors": descriptors,
        "synthetic_accessibility_score": sa_score,
        "alerts": {
            "pains": pains,
            "brenk": brenk,
            "nih": nih,
        },
        "reference_similarity": {
            "reference_file_available": reference is not None,
            "tanimoto_morgan_radius_2_2048": similarity,
        },
        "default_hard_constraint_checks": checks,
        "default_hard_constraints_pass": all(checks.values()),
    })
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smiles", required=True)
    parser.add_argument("--reference", default="reference.sdf")
    args = parser.parse_args()
    print(json.dumps(validate_molecule(args.smiles, args.reference), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
