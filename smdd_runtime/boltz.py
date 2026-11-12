"""Boltz cache checks shared by daemonless runners."""

import fcntl
import hashlib
import os
import shutil
import tempfile


REQUIRED_CACHE_ENTRIES = ("mols", "boltz2_conf.ckpt", "boltz2_aff.ckpt")
DEFAULT_CACHE_DIR = "/root/.boltz"


def cache_dir():
    return os.path.abspath(os.environ.get("SMDD_BOLTZ_CACHE", DEFAULT_CACHE_DIR))


def require_cache():
    path = cache_dir()
    missing = [name for name in REQUIRED_CACHE_ENTRIES if not os.path.exists(os.path.join(path, name))]
    if missing:
        raise RuntimeError(
            f"Boltz cache is incomplete at {path}; missing: {', '.join(missing)}. "
            "Populate it before the network-restricted run or set SMDD_BOLTZ_CACHE."
        )
    return path


def stage_cache(source: str) -> str:
    """Copy required Boltz assets once to a versioned node-local cache.

    Staging is disabled by default because the container cache is expected at
    ``/root/.boltz``. Set ``SMDD_BOLTZ_LOCAL_CACHE`` to a directory to enable
    it explicitly. A filesystem lock and atomic rename make concurrent runs
    safe.
    """
    source = os.path.realpath(source)
    configured = os.environ.get("SMDD_BOLTZ_LOCAL_CACHE")
    if not configured or configured.lower() in {"0", "false", "no", "off"}:
        return source
    local_root = os.path.abspath(configured)
    if os.path.commonpath((source, local_root)) == source:
        return source

    signature_parts = [source]
    for name in REQUIRED_CACHE_ENTRIES:
        stat = os.stat(os.path.join(source, name))
        signature_parts.append(f"{name}:{stat.st_size}:{stat.st_mtime_ns}")
    signature = hashlib.sha256("\n".join(signature_parts).encode()).hexdigest()[:16]
    target = os.path.join(local_root, signature)

    def complete(path):
        return all(os.path.exists(os.path.join(path, name)) for name in REQUIRED_CACHE_ENTRIES)

    os.makedirs(local_root, exist_ok=True)
    lock_path = os.path.join(local_root, ".stage.lock")
    with open(lock_path, "a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if complete(target):
            return target

        staging = tempfile.mkdtemp(prefix=f".{signature}.staging-", dir=local_root)
        try:
            for name in REQUIRED_CACHE_ENTRIES:
                src = os.path.join(source, name)
                dst = os.path.join(staging, name)
                if name == "mols":
                    # This directory contains many small immutable CCD files.
                    # Only checkpoint deserialization is on the critical path;
                    # copying mols would add substantial metadata I/O.
                    os.symlink(src, dst, target_is_directory=True)
                else:
                    shutil.copy2(src, dst)
            if os.path.exists(target):
                shutil.rmtree(target)
            os.replace(staging, target)
        finally:
            if os.path.exists(staging):
                shutil.rmtree(staging, ignore_errors=True)
    return target
