"""Obfuscate task files to prevent data leakage of protein/ligand identifiers.

Copies task files to a temp directory with identifiers stripped or replaced.
"""

import os
import re
import shutil
import tempfile
import yaml


# PDB HEADER/TITLE line patterns
PDB_HEADER_RE = re.compile(r"^(HEADER|TITLE|COMPND|SOURCE|KEYWDS|EXPDTA|AUTHOR|JRNL|REMARK\s+\d)", re.IGNORECASE)


def _extract_identifier(task_id):
    """Extract the identifier from a standardised task ID.

    Task IDs follow the format ``smdd_{NNN}_{identifier}_{index}``.
    The identifier is the DUD-E target name (task 1), UniProt ID (tasks 2-3, 6),
    target name (task 4), or ligand code (task 5).  The index is always the last ``_``-delimited token
    and is a bare integer.

    Returns (identifier, index) or (None, None) on parse failure.
    """
    parts = task_id.split("_")
    # Minimum: ["smdd", "NNN", identifier, index]
    if len(parts) < 4 or parts[0] != "smdd":
        return None, None
    # index is always the last token
    index = parts[-1]
    # identifier is everything between the type prefix and the index
    identifier = "_".join(parts[2:-1])
    return identifier, index


def _obfuscate_pdb_file(src_path, dst_path):
    """Copy a PDB file with HEADER/TITLE/metadata records stripped."""
    with open(src_path, "r") as f_in, open(dst_path, "w") as f_out:
        for line in f_in:
            if PDB_HEADER_RE.match(line):
                # Replace with generic line preserving record type
                record_type = line[:6].strip()
                f_out.write(f"{record_type:<6s}    OBFUSCATED\n")
            else:
                f_out.write(line)


def _obfuscate_sdf_file(src_path, dst_path):
    """Copy an SDF file with molecule name replaced."""
    with open(src_path, "r") as f_in:
        lines = f_in.readlines()

    if lines:
        # First line is the molecule name
        lines[0] = "reference_ligand\n"

    with open(dst_path, "w") as f_out:
        f_out.writelines(lines)


def obfuscate_task(task_dir, task_alias):
    """Create an obfuscated copy of a task directory.

    Args:
        task_dir: Path to original task directory
        task_alias: Name of the task

    Returns:
        obfuscated_dir: Temp directory path
    """
    obfuscated_dir = tempfile.mkdtemp(prefix=f"smdd_obfuscated_{task_alias}_")

    # Load task YAML
    yaml_path = os.path.join(task_dir, "task.yaml")
    with open(yaml_path, "r") as f:
        task_yaml = yaml.safe_load(f)

    # Copy and obfuscate other files
    input_files = task_yaml.get("input_files", [])
    for filename in input_files:
        src = os.path.join(task_dir, filename)
        dst = os.path.join(obfuscated_dir, filename)
        if not os.path.isfile(src):
            continue

        if filename.endswith(".pdb"):
            _obfuscate_pdb_file(src, dst)
        elif filename.endswith(".sdf"):
            _obfuscate_sdf_file(src, dst)
        else:
            shutil.copy2(src, dst)

    return obfuscated_dir


def cleanup_obfuscated(obfuscated_dir):
    """Remove a temporary obfuscated directory."""
    shutil.rmtree(obfuscated_dir, ignore_errors=True)
