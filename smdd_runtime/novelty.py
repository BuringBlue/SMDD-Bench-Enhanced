"""Trusted novelty queries for ChEMBL, DrugBank, and SureChEMBL."""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import signal
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlencode
from urllib.request import Request, urlopen

from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

from smdd_runtime.msa import _recv_message, _send_message


CHEMBL_API = "https://www.ebi.ac.uk/chembl/api/data"
SURECHEMBL_API = "https://www.surechembl.org/api"
PUBCHEM_API = "https://pubchem.ncbi.nlm.nih.gov/rest/pug"
DEFAULT_SEARCH_FLOOR = 0.40
MAX_HTTP_BYTES = 32 * 1024 * 1024
MAX_REQUEST_BYTES = 16 * 1024
MAX_RESPONSE_BYTES = 2 * 1024 * 1024
PUBCHEM_CANDIDATE_LIMIT = 200
_DRUGBANK_ID = re.compile(r"DB[0-9]{5}$", re.IGNORECASE)
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
_DRUGBANK_INDEXES: dict[str, tuple[int, "_LocalDrugBankIndex"]] = {}
_DRUGBANK_LOCK = threading.Lock()


def _request_json(
    url: str,
    *,
    payload: dict | None = None,
    timeout: int = 60,
    attempts: int = 3,
) -> dict:
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    headers = {
        "Accept": "application/json",
        "User-Agent": "SMDD-Bench-Enhanced/novelty-check",
    }
    if data is not None:
        headers["Content-Type"] = "application/json"

    for attempt in range(attempts):
        try:
            request = Request(url, data=data, headers=headers)
            with urlopen(request, timeout=timeout) as response:
                body = response.read(MAX_HTTP_BYTES + 1)
            if len(body) > MAX_HTTP_BYTES:
                raise RuntimeError(f"response from {url} exceeds {MAX_HTTP_BYTES} bytes")
            result = json.loads(body)
            if not isinstance(result, dict):
                raise RuntimeError(f"non-object JSON response from {url}")
            return result
        except HTTPError as exc:
            retryable = exc.code == 429 or 500 <= exc.code < 600
            if not retryable or attempt + 1 == attempts:
                raise RuntimeError(f"HTTP {exc.code} from {url}") from exc
        except (URLError, TimeoutError, json.JSONDecodeError) as exc:
            if attempt + 1 == attempts:
                raise RuntimeError(f"request failed for {url}: {exc}") from exc
        time.sleep(0.5 * (2**attempt))
    raise AssertionError("unreachable")


def _scaffold(molecule: Chem.Mol) -> tuple[Chem.Mol | None, str | None]:
    scaffold = MurckoScaffold.GetScaffoldForMol(molecule)
    if scaffold.GetNumAtoms() == 0:
        return None, None
    return scaffold, Chem.MolToSmiles(scaffold, canonical=True)


def _normalized_similarity(value) -> float:
    similarity = float(value)
    return similarity / 100.0 if similarity > 1.0 else similarity


def _chembl_query(smiles: str, scaffold_smiles: str | None, floor: float) -> dict:
    threshold = max(1, round(floor * 100))
    encoded = quote(smiles, safe="")
    query = urlencode({
        "limit": 1,
        # Supplying fields avoids an intermittent ChEMBL pagination error. The
        # service may still return additional fields, which are ignored.
        "fields": "molecule_chembl_id,molecule_structures,similarity",
    })
    response = _request_json(
        f"{CHEMBL_API}/similarity/{encoded}/{threshold}.json?{query}"
    )
    molecules = response.get("molecules") or []
    best = molecules[0] if molecules else None

    scaffold_seen = None
    scaffold_match_id = None
    if scaffold_smiles:
        scaffold_response = _request_json(
            f"{CHEMBL_API}/substructure/{quote(scaffold_smiles, safe='')}.json?{query}"
        )
        scaffold_matches = scaffold_response.get("molecules") or []
        scaffold_seen = bool(scaffold_matches)
        if scaffold_matches:
            scaffold_match_id = scaffold_matches[0].get("molecule_chembl_id")

    return {
        "source": "ChEMBL",
        "method": "ChEMBL similarity and substructure APIs",
        "max_similarity": _normalized_similarity(best["similarity"]) if best else None,
        "similarity_below": None if best else floor,
        "match_id": best.get("molecule_chembl_id") if best else None,
        "scaffold_seen": scaffold_seen,
        "scaffold_match_id": scaffold_match_id,
        "exhaustive_above_search_floor": True,
    }


def _surechembl_search(
    structure: str,
    search_type: str,
    threshold: float | None = None,
) -> dict:
    payload = {
        "struct": structure,
        "struct_search_type": search_type,
        "page": 1,
        "items_per_page": 1,
    }
    if threshold is not None:
        payload["threshold"] = threshold
    return _request_json(
        f"{SURECHEMBL_API}/search/structure",
        payload=payload,
    )


def _surechembl_query(smiles: str, scaffold_smiles: str | None, floor: float) -> dict:
    response = _surechembl_search(smiles, "similarity", floor)
    matches = response.get("chemical_ids") or []
    best = matches[0] if matches else None
    if isinstance(best, list):
        match_id = best[0]
        similarity = _normalized_similarity(best[1])
    else:
        match_id = best
        similarity = None

    scaffold_seen = None
    scaffold_match_id = None
    if scaffold_smiles:
        scaffold_response = _surechembl_search(scaffold_smiles, "substructure")
        scaffold_matches = scaffold_response.get("chemical_ids") or []
        scaffold_seen = bool(scaffold_matches)
        if scaffold_matches:
            scaffold_match = scaffold_matches[0]
            scaffold_match_id = (
                scaffold_match[0] if isinstance(scaffold_match, list) else scaffold_match
            )

    return {
        "source": "SureChEMBL",
        "method": "SureChEMBL similarity and substructure APIs",
        "max_similarity": similarity,
        "similarity_below": None if best else floor,
        "match_id": str(match_id) if match_id is not None else None,
        "scaffold_seen": scaffold_seen,
        "scaffold_match_id": (
            str(scaffold_match_id) if scaffold_match_id is not None else None
        ),
        "exhaustive_above_search_floor": True,
    }


class _LocalDrugBankIndex:
    def __init__(self, path: str):
        self.path = path
        self.records = self._read_records(path)
        if not self.records:
            raise ValueError(f"DrugBank structure file contains no valid molecules: {path}")
        self.fingerprints = [_MORGAN.GetFingerprint(record[1]) for record in self.records]

    @staticmethod
    def _record_id(molecule: Chem.Mol, fallback: str) -> str:
        for name in ("DRUGBANK_ID", "DATABASE_ID", "DrugBank ID", "_Name"):
            if molecule.HasProp(name) and molecule.GetProp(name).strip():
                return molecule.GetProp(name).strip()
        return fallback

    @classmethod
    def _read_records(cls, path: str) -> list[tuple[str, Chem.Mol, str]]:
        lower = path.lower()
        records = []
        if lower.endswith((".sdf", ".sdf.gz")):
            if lower.endswith(".gz"):
                handle = gzip.open(path, "rb")
                supplier = Chem.ForwardSDMolSupplier(handle)
            else:
                handle = None
                supplier = Chem.SDMolSupplier(path)
            try:
                for index, molecule in enumerate(supplier):
                    if molecule is None:
                        continue
                    smiles = Chem.MolToSmiles(molecule, canonical=True)
                    records.append((cls._record_id(molecule, str(index)), molecule, smiles))
            finally:
                if handle is not None:
                    handle.close()
            return records

        with open(path, "r", encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                stripped = line.strip()
                if not stripped or stripped.startswith("#"):
                    continue
                fields = stripped.split()
                molecule = Chem.MolFromSmiles(fields[0])
                if molecule is None:
                    continue
                identifier = fields[1] if len(fields) > 1 else str(index)
                records.append(
                    (identifier, molecule, Chem.MolToSmiles(molecule, canonical=True))
                )
        return records

    def query(self, molecule: Chem.Mol, scaffold: Chem.Mol | None) -> dict:
        similarities = DataStructs.BulkTanimotoSimilarity(
            _MORGAN.GetFingerprint(molecule),
            self.fingerprints,
        )
        best_index = max(range(len(similarities)), key=similarities.__getitem__)
        scaffold_match = next(
            (
                identifier
                for identifier, known_molecule, _ in self.records
                if scaffold is not None and known_molecule.HasSubstructMatch(scaffold)
            ),
            None,
        )
        return {
            "source": "DrugBank",
            "method": "local licensed DrugBank structure export",
            "max_similarity": float(similarities[best_index]),
            "similarity_below": None,
            "match_id": self.records[best_index][0],
            "scaffold_seen": None if scaffold is None else scaffold_match is not None,
            "scaffold_match_id": scaffold_match,
            "exhaustive_above_search_floor": True,
        }


def _local_drugbank_index(path: str) -> _LocalDrugBankIndex:
    resolved = os.path.realpath(path)
    modified = os.stat(resolved).st_mtime_ns
    with _DRUGBANK_LOCK:
        cached = _DRUGBANK_INDEXES.get(resolved)
        if cached is None or cached[0] != modified:
            cached = (modified, _LocalDrugBankIndex(resolved))
            _DRUGBANK_INDEXES[resolved] = cached
        return cached[1]


def _chunks(values: list[int], size: int = 50):
    for offset in range(0, len(values), size):
        yield values[offset : offset + size]


def _pubchem_drugbank_candidates(smiles: str, floor: float) -> list[dict]:
    threshold = max(1, round(floor * 100))
    response = _request_json(
        f"{PUBCHEM_API}/compound/fastsimilarity_2d/smiles/"
        f"{quote(smiles, safe='')}/cids/JSON?"
        f"{urlencode({'Threshold': threshold, 'MaxRecords': PUBCHEM_CANDIDATE_LIMIT})}"
    )
    cids = [int(value) for value in response.get("IdentifierList", {}).get("CID", [])]
    drugbank_ids: dict[int, list[str]] = {}
    for batch in _chunks(cids):
        joined = ",".join(str(value) for value in batch)
        xrefs = _request_json(
            f"{PUBCHEM_API}/compound/cid/{joined}/xrefs/RegistryID/JSON"
        )
        for entry in xrefs.get("InformationList", {}).get("Information", []):
            identifiers = [
                value
                for value in entry.get("RegistryID", [])
                if _DRUGBANK_ID.fullmatch(value)
            ]
            if identifiers:
                drugbank_ids[int(entry["CID"])] = identifiers

    records = []
    for batch in _chunks(list(drugbank_ids)):
        joined = ",".join(str(value) for value in batch)
        properties = _request_json(
            f"{PUBCHEM_API}/compound/cid/{joined}/property/ConnectivitySMILES/JSON"
        )
        for entry in properties.get("PropertyTable", {}).get("Properties", []):
            candidate_smiles = entry.get("ConnectivitySMILES") or entry.get("CanonicalSMILES")
            molecule = Chem.MolFromSmiles(candidate_smiles or "")
            if molecule is None:
                continue
            cid = int(entry["CID"])
            records.append({
                "cid": cid,
                "drugbank_ids": drugbank_ids[cid],
                "molecule": molecule,
            })
    return records


def _drugbank_query(
    smiles: str,
    molecule: Chem.Mol,
    scaffold: Chem.Mol | None,
    floor: float,
    path: str | None,
) -> dict:
    if path:
        return _local_drugbank_index(path).query(molecule, scaffold)

    records = _pubchem_drugbank_candidates(smiles, floor)
    if not records:
        return {
            "source": "DrugBank",
            "method": "PubChem similarity candidates filtered by DrugBank registry IDs",
            "max_similarity": None,
            "similarity_below": None,
            "match_id": None,
            "scaffold_seen": None if scaffold is None else False,
            "scaffold_match_id": None,
            "exhaustive_above_search_floor": False,
            "candidate_limit": PUBCHEM_CANDIDATE_LIMIT,
        }

    query_fingerprint = _MORGAN.GetFingerprint(molecule)
    similarities = [
        DataStructs.TanimotoSimilarity(
            query_fingerprint,
            _MORGAN.GetFingerprint(record["molecule"]),
        )
        for record in records
    ]
    best_index = max(range(len(records)), key=similarities.__getitem__)
    scaffold_match = next(
        (
            record
            for record in records
            if scaffold is not None and record["molecule"].HasSubstructMatch(scaffold)
        ),
        None,
    )
    best = records[best_index]
    return {
        "source": "DrugBank",
        "method": "PubChem similarity candidates filtered by DrugBank registry IDs",
        "max_similarity": float(similarities[best_index]),
        "similarity_below": None,
        "match_id": best["drugbank_ids"][0],
        "pubchem_cid": best["cid"],
        "scaffold_seen": None if scaffold is None else scaffold_match is not None,
        "scaffold_match_id": (
            scaffold_match["drugbank_ids"][0] if scaffold_match else None
        ),
        "exhaustive_above_search_floor": False,
        "candidate_limit": PUBCHEM_CANDIDATE_LIMIT,
    }


def compute_novelty(
    smiles: str,
    *,
    search_floor: float = DEFAULT_SEARCH_FLOOR,
    drugbank_path: str | None = None,
) -> dict:
    if not 0.0 < search_floor <= 1.0:
        raise ValueError("search_floor must be in (0, 1]")
    molecule = Chem.MolFromSmiles(smiles)
    if molecule is None:
        raise ValueError("SMILES could not be parsed")
    canonical_smiles = Chem.MolToSmiles(molecule, canonical=True, isomericSmiles=True)
    scaffold, scaffold_smiles = _scaffold(molecule)
    drugbank_path = drugbank_path or os.environ.get("SMDD_DRUGBANK_STRUCTURES")

    with ThreadPoolExecutor(max_workers=3) as pool:
        futures = {
            "chembl": pool.submit(
                _chembl_query, canonical_smiles, scaffold_smiles, search_floor
            ),
            "drugbank": pool.submit(
                _drugbank_query,
                canonical_smiles,
                molecule,
                scaffold,
                search_floor,
                drugbank_path,
            ),
            "patent": pool.submit(
                _surechembl_query, canonical_smiles, scaffold_smiles, search_floor
            ),
        }
        sources = {name: future.result() for name, future in futures.items()}

    seen_in = [
        name for name, result in sources.items() if result["scaffold_seen"] is True
    ]
    scaffold_complete = all(
        result["exhaustive_above_search_floor"] for result in sources.values()
    )
    return {
        "canonical_smiles": canonical_smiles,
        "search_floor": search_floor,
        "database_similarity": sources,
        "scaffold_novelty": {
            "scaffold_smiles": scaffold_smiles,
            "applicable": scaffold_smiles is not None,
            "seen_in": seen_in,
            "is_novel": None if scaffold_smiles is None else not seen_in,
            "complete": scaffold_complete,
            "affects_pass": False,
        },
    }


def request_novelty(
    smiles: str,
    *,
    search_floor: float = DEFAULT_SEARCH_FLOOR,
    timeout: int = 300,
) -> dict:
    socket_path = os.environ.get("SMDD_NOVELTY_SOCKET")
    if not socket_path:
        return compute_novelty(smiles, search_floor=search_floor)
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as connection:
        connection.settimeout(timeout)
        connection.connect(socket_path)
        _send_message(connection, {"smiles": smiles, "search_floor": search_floor})
        response = _recv_message(connection, MAX_RESPONSE_BYTES)
    if "error" in response:
        raise RuntimeError(response["error"])
    return response["result"]


def _handle_connection(connection: socket.socket) -> None:
    with connection:
        request = _recv_message(connection, MAX_REQUEST_BYTES)
        try:
            result = compute_novelty(
                request["smiles"],
                search_floor=float(request.get("search_floor", DEFAULT_SEARCH_FLOOR)),
            )
            response = {"result": result}
        except Exception as exc:
            response = {"error": f"{type(exc).__name__}: {exc}"}
        _send_message(connection, response)


def serve(socket_path: str) -> None:
    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(socket_path)
        os.chmod(socket_path, 0o600)
        server.listen(16)
        while True:
            connection, _ = server.accept()
            threading.Thread(
                target=_handle_connection,
                args=(connection,),
                daemon=True,
            ).start()


class NoveltyService:
    """Lifecycle wrapper for novelty queries outside the networkless sandbox."""

    def __init__(self, socket_path: str, python: str = sys.executable):
        self.socket_path = os.path.abspath(socket_path)
        self.python = python
        self.process = None

    def start(self) -> str:
        repo_root = str(Path(__file__).resolve().parent.parent)
        environment = {
            "HOME": os.environ.get("HOME", "/root"),
            "LANG": "C.UTF-8",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONPATH": repo_root,
        }
        for key in (
            "HTTP_PROXY",
            "HTTPS_PROXY",
            "NO_PROXY",
            "SMDD_DRUGBANK_STRUCTURES",
        ):
            if key in os.environ:
                environment[key] = os.environ[key]
        self.process = subprocess.Popen(
            [
                self.python,
                "-m",
                "smdd_runtime.novelty",
                "serve",
                "--socket",
                self.socket_path,
            ],
            stdout=subprocess.DEVNULL,
            stderr=None,
            text=True,
            env=environment,
            start_new_session=True,
        )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if os.path.exists(self.socket_path):
                return self.socket_path
            if self.process.poll() is not None:
                raise RuntimeError(
                    f"novelty broker failed to start with code {self.process.returncode}"
                )
            time.sleep(0.05)
        self.close()
        raise RuntimeError("novelty broker did not create its socket within 15 seconds")

    def close(self) -> None:
        if self.process and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=5)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try:
                    os.killpg(self.process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                self.process.wait()
        self.process = None
        try:
            os.unlink(self.socket_path)
        except FileNotFoundError:
            pass

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("serve",))
    parser.add_argument("--socket", required=True)
    args = parser.parse_args()
    serve(args.socket)


if __name__ == "__main__":
    main()
