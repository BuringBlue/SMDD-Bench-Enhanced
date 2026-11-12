"""Fixed-purpose Unix-socket broker for ColabFold MSA generation."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path

import yaml


MSA_SERVER_URL = "https://api.colabfold.com"
MAX_REQUEST_BYTES = 1_000_000
MAX_RESPONSE_BYTES = 256_000_000
AMINO_ACIDS = frozenset("ABCDEFGHIJKLMNOPQRSTUVWXYZ-")
MSA_CACHE_SCHEMA = "v1-colabfold-greedy"
DEFAULT_MSA_CACHE_DIR = "/root/.cache/smdd/msa"


def _recv_message(conn: socket.socket, limit: int) -> dict:
    header = _recv_exact(conn, 8)
    length = struct.unpack("!Q", header)[0]
    if length > limit:
        raise ValueError(f"message exceeds {limit} bytes")
    return json.loads(_recv_exact(conn, length).decode("utf-8"))


def _recv_exact(conn: socket.socket, length: int) -> bytes:
    chunks = []
    remaining = length
    while remaining:
        chunk = conn.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("unexpected end of MSA broker message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _send_message(conn: socket.socket, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    conn.sendall(struct.pack("!Q", len(data)) + data)


def _normalize_sequences(values: list[str]) -> list[str]:
    if not values or len(values) > 32:
        raise ValueError("MSA request must contain 1-32 protein sequences")
    normalized = []
    for value in values:
        sequence = "".join(value.split()).upper()
        if not sequence or len(sequence) > 20_000 or not set(sequence) <= AMINO_ACIDS:
            raise ValueError("invalid protein sequence in MSA request")
        if sequence not in normalized:
            normalized.append(sequence)
    return normalized


def msa_cache_dir() -> str:
    """Return the trusted cache path, kept outside task-visible Boltz assets."""
    return os.path.abspath(os.environ.get("SMDD_MSA_CACHE", DEFAULT_MSA_CACHE_DIR))


def _compute_uncached(sequences: list[str]) -> dict[str, str]:
    from boltz.main import compute_msa

    names = {f"seq_{index:02d}": sequence for index, sequence in enumerate(sequences)}
    temp_dir = tempfile.mkdtemp(prefix="smdd_msa_")
    try:
        compute_msa(
            data=names,
            target_id="smdd",
            msa_dir=Path(temp_dir),
            msa_server_url=MSA_SERVER_URL,
            msa_pairing_strategy="greedy",
        )
        return {
            sequence: Path(temp_dir, f"{name}.csv").read_text(encoding="utf-8")
            for name, sequence in names.items()
        }
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


class PersistentMsaCache:
    """Content-addressed MSA cache safe across threads and runner processes.

    A multi-chain ColabFold response contains paired rows whose contents and
    row keys depend on the complete ordered sequence set. Therefore the cache
    is keyed by the full normalized request rather than mixing independently
    cached chains, which would silently change Boltz inputs.
    """

    def __init__(self, root: str) -> None:
        self.root = os.path.join(os.path.abspath(root), MSA_CACHE_SCHEMA)
        self.lock_root = os.path.join(self.root, ".locks")
        os.makedirs(self.lock_root, mode=0o700, exist_ok=True)
        os.chmod(self.root, 0o700)
        os.chmod(self.lock_root, 0o700)

    @staticmethod
    def _digest(sequences: list[str]) -> str:
        identity = json.dumps(
            {
                "schema": MSA_CACHE_SCHEMA,
                "server": MSA_SERVER_URL,
                "pairing": "greedy",
                "sequences": sequences,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
        return hashlib.sha256(identity.encode("ascii")).hexdigest()

    def _entry_path(self, digest: str) -> str:
        return os.path.join(self.root, f"{digest}.json")

    def _read(self, digest: str, sequences: list[str]) -> dict[str, str] | None:
        try:
            with open(self._entry_path(digest), "r", encoding="utf-8") as handle:
                payload = json.load(handle)
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None
        if not isinstance(payload, dict):
            return None
        if payload.get("sequences") != sequences:
            return None
        msas = payload.get("msas")
        if not isinstance(msas, dict):
            return None
        if any(not isinstance(msas.get(sequence), str) or not msas[sequence]
               for sequence in sequences):
            return None
        return {sequence: msas[sequence] for sequence in sequences}

    def _write(
        self,
        digest: str,
        sequences: list[str],
        msas: dict[str, str],
    ) -> None:
        fd, temp_path = tempfile.mkstemp(prefix=f".{digest}.", dir=self.root)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump({"sequences": sequences, "msas": msas}, handle)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temp_path, self._entry_path(digest))
        finally:
            try:
                os.unlink(temp_path)
            except FileNotFoundError:
                pass

    def get_or_compute(self, sequences: list[str]) -> dict[str, str]:
        normalized = _normalize_sequences(sequences)
        digest = self._digest(normalized)

        # flock prevents duplicate ColabFold requests from another broker
        # process (for example Agent and evaluator running concurrently).
        with open(
            os.path.join(self.lock_root, f"{digest}.lock"),
            "a+",
            encoding="utf-8",
        ) as lock:
            os.chmod(lock.name, 0o600)
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            cached = self._read(digest, normalized)
            if cached is not None:
                return cached

            computed = _compute_uncached(normalized)
            if any(not computed.get(sequence) for sequence in normalized):
                raise RuntimeError(
                    "ColabFold returned no MSA for a requested sequence"
                )
            result = {sequence: computed[sequence] for sequence in normalized}
            self._write(digest, normalized, result)
            return result


def _compute(sequences: list[str], cache: PersistentMsaCache) -> dict[str, str]:
    return cache.get_or_compute(sequences)


def request_msa(socket_path: str, sequences: list[str], timeout: int = 1800) -> dict[str, str]:
    inherited_fd = os.environ.get("SMDD_MSA_FD")
    with socket.socket(fileno=os.dup(int(inherited_fd))) if inherited_fd else socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(timeout)
        if not inherited_fd:
            conn.connect(socket_path)
        _send_message(conn, {"sequences": sequences})
        response = _recv_message(conn, MAX_RESPONSE_BYTES)
    if "error" in response:
        raise RuntimeError(response["error"])
    return response["msas"]


def attach_msas(yaml_path: str, output_dir: str, socket_path: str) -> str:
    """Return a copy of a Boltz YAML whose protein entries use prefetched MSAs."""
    with open(yaml_path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    proteins = [item["protein"] for item in data.get("sequences", []) if "protein" in item]
    sequences = [protein["sequence"] for protein in proteins]
    msas = request_msa(socket_path, sequences)

    os.makedirs(output_dir, exist_ok=True)
    for protein in proteins:
        sequence = "".join(protein["sequence"].split()).upper()
        digest = hashlib.sha256(sequence.encode()).hexdigest()[:16]
        msa_path = os.path.join(output_dir, f"msa_{digest}.csv")
        with open(msa_path, "w", encoding="utf-8") as handle:
            handle.write(msas[sequence])
        protein["msa"] = msa_path

    # offline_path = os.path.join(output_dir, "input_with_msa.yaml")
    offline_path = os.path.join(output_dir, os.path.basename(yaml_path))
    with open(offline_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(data, handle, sort_keys=False)
    return offline_path


def _handle_connection(conn: socket.socket, cache: PersistentMsaCache) -> None:
    with conn:
        while True:
            try:
                request = _recv_message(conn, MAX_REQUEST_BYTES)
            except ConnectionError:
                return
            try:
                response = {"msas": _compute(request["sequences"], cache)}
            except Exception as exc:
                response = {"error": f"{type(exc).__name__}: {exc}"}
            _send_message(conn, response)


def serve(socket_path: str, cache_dir: str | None = None) -> None:
    cache = PersistentMsaCache(cache_dir or msa_cache_dir())
    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(socket_path)
        os.chmod(socket_path, 0o666)
        server.listen(16)
        while True:
            conn, _ = server.accept()
            threading.Thread(
                target=_handle_connection,
                args=(conn, cache),
                daemon=True,
            ).start()


class MsaService:
    """Lifecycle wrapper for the trusted, fixed-destination MSA broker."""

    def __init__(
        self,
        socket_path: str,
        python: str = sys.executable,
        cache_dir: str | None = None,
    ):
        self.socket_path = os.path.abspath(socket_path)
        self.python = python
        self.cache_dir = os.path.abspath(cache_dir or msa_cache_dir())
        self.process = None

    def start(self) -> str:
        repo_root = str(Path(__file__).resolve().parent.parent)
        env = {
            "HOME": os.environ.get("HOME", "/root"),
            "LANG": "C.UTF-8",
            "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
            "PYTHONPATH": repo_root,
        }
        for key in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY"):
            if key in os.environ:
                env[key] = os.environ[key]
        self.process = subprocess.Popen(
            [
                self.python,
                "-m",
                "smdd_runtime.msa",
                "serve",
                "--socket",
                self.socket_path,
                "--cache-dir",
                self.cache_dir,
            ],
            stdout=subprocess.DEVNULL,
            stderr=None,
            text=True,
            env=env,
            start_new_session=True,
        )
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if os.path.exists(self.socket_path):
                return self.socket_path
            if self.process.poll() is not None:
                raise RuntimeError(f"MSA broker failed to start with code {self.process.returncode}")
            time.sleep(0.05)
        self.close()
        raise RuntimeError("MSA broker did not create its socket within 15 seconds")

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
    parser.add_argument("serve", nargs="?")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--cache-dir", default=msa_cache_dir())
    args = parser.parse_args()
    serve(args.socket, args.cache_dir)


if __name__ == "__main__":
    main()
