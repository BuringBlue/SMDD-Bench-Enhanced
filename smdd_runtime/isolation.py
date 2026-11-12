"""Fail-closed Landlock and namespace sandbox for local task execution."""

from __future__ import annotations

import ctypes
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence


SYSTEM_PATHS = ("/bin", "/sbin", "/usr", "/lib", "/lib64", "/opt", "/etc", "/sys")
PROC_PATHS = ("/proc/self", "/proc/cpuinfo", "/proc/meminfo", "/proc/driver/nvidia")
BASE_DEVICES = ("/dev/null", "/dev/zero", "/dev/full", "/dev/random", "/dev/urandom", "/dev/pts")
SAFE_ENV_KEYS = (
    "CUDA_VISIBLE_DEVICES",
    "LD_LIBRARY_PATH",
    "NVIDIA_DRIVER_CAPABILITIES",
    "NVIDIA_VISIBLE_DEVICES",
    "OMP_NUM_THREADS",
)

# Landlock ABI 1 filesystem access rights.
LL_EXECUTE = 1 << 0
LL_WRITE_FILE = 1 << 1
LL_READ_FILE = 1 << 2
LL_READ_DIR = 1 << 3
LL_REMOVE_DIR = 1 << 4
LL_REMOVE_FILE = 1 << 5
LL_MAKE_CHAR = 1 << 6
LL_MAKE_DIR = 1 << 7
LL_MAKE_REG = 1 << 8
LL_MAKE_SOCK = 1 << 9
LL_MAKE_FIFO = 1 << 10
LL_MAKE_BLOCK = 1 << 11
LL_MAKE_SYM = 1 << 12
LL_READ = LL_EXECUTE | LL_READ_FILE | LL_READ_DIR
LL_WRITE = (
    LL_WRITE_FILE
    | LL_REMOVE_DIR
    | LL_REMOVE_FILE
    | LL_MAKE_DIR
    | LL_MAKE_REG
    | LL_MAKE_SOCK
    | LL_MAKE_FIFO
    | LL_MAKE_SYM
)
LL_HANDLED = LL_READ | LL_WRITE | LL_MAKE_CHAR | LL_MAKE_BLOCK


@dataclass(frozen=True)
class Bind:
    """A host path visible to the sandbox."""

    source: str
    writable: bool = False


@dataclass
class RunResult:
    stdout: str
    stderr: str
    returncode: int


class SandboxUnavailable(RuntimeError):
    pass


class SandboxTimeout(TimeoutError):
    pass


class _RulesetAttr(ctypes.Structure):
    _fields_ = [("handled_access_fs", ctypes.c_uint64)]


class _PathBeneathAttr(ctypes.Structure):
    _fields_ = [
        ("allowed_access", ctypes.c_uint64),
        ("parent_fd", ctypes.c_int32),
        ("reserved", ctypes.c_uint32),
    ]


class _SockFilter(ctypes.Structure):
    _fields_ = [("code", ctypes.c_ushort), ("jt", ctypes.c_ubyte), ("jf", ctypes.c_ubyte), ("k", ctypes.c_uint32)]


class _SockFprog(ctypes.Structure):
    _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.POINTER(_SockFilter))]


def _kill_group(proc: subprocess.Popen) -> None:
    if proc.poll() is None:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _set_no_new_privs(libc: ctypes.CDLL) -> None:
    if libc.prctl(38, 1, 0, 0, 0) != 0:  # PR_SET_NO_NEW_PRIVS
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def _apply_landlock(readonly: Iterable[str], writable: Iterable[str], devices: Iterable[str]) -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    create_ruleset, add_rule, restrict_self = 444, 445, 446  # x86_64
    abi = libc.syscall(create_ruleset, 0, 0, 1)  # LANDLOCK_CREATE_RULESET_VERSION
    if abi < 1:
        raise SandboxUnavailable("Landlock is unavailable in this container kernel")

    ruleset_attr = _RulesetAttr(LL_HANDLED)
    ruleset_fd = libc.syscall(create_ruleset, ctypes.byref(ruleset_attr), ctypes.sizeof(ruleset_attr), 0)
    if ruleset_fd < 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))

    try:
        seen: set[tuple[str, int]] = set()
        rules = [*((p, LL_READ) for p in readonly), *((p, LL_READ | LL_WRITE_FILE) for p in devices)]
        rules.extend((p, LL_READ | LL_WRITE) for p in writable)
        for path, access in rules:
            real_path = os.path.realpath(path)
            if not os.path.isdir(real_path):
                access &= LL_EXECUTE | LL_READ_FILE | LL_WRITE_FILE
            key = (real_path, access)
            if not os.path.exists(real_path) or key in seen:
                continue
            seen.add(key)
            path_fd = os.open(real_path, os.O_PATH | os.O_CLOEXEC)
            try:
                attr = _PathBeneathAttr(access, path_fd, 0)
                if libc.syscall(add_rule, ruleset_fd, 1, ctypes.byref(attr), 0) != 0:
                    err = ctypes.get_errno()
                    raise OSError(err, os.strerror(err), real_path)
            finally:
                os.close(path_fd)

        _set_no_new_privs(libc)
        if libc.syscall(restrict_self, ruleset_fd, 0) != 0:
            err = ctypes.get_errno()
            raise OSError(err, os.strerror(err))
    finally:
        os.close(ruleset_fd)


def _install_seccomp(allow_unix_sockets: bool = False) -> None:
    """Block sockets and unsafe syscalls, optionally allowing local IPC."""
    if platform.machine() not in ("x86_64", "amd64"):
        raise SandboxUnavailable("the local sandbox currently supports linux/amd64 only")

    # Enforce x86_64 and deny selected syscalls. Trusted evaluator processes
    # may opt into AF_UNIX, but all other socket domains remain blocked.
    bpf_ld_w_abs = 0x20
    bpf_jmp_jeq_k = 0x15
    bpf_ret_k = 0x06
    seccomp_ret_errno = 0x00050000
    seccomp_ret_allow = 0x7FFF0000
    seccomp_ret_kill = 0x80000000
    instructions = [
        _SockFilter(bpf_ld_w_abs, 0, 0, 4),  # seccomp_data.arch
        _SockFilter(bpf_jmp_jeq_k, 1, 0, 0xC000003E),  # AUDIT_ARCH_X86_64
        _SockFilter(bpf_ret_k, 0, 0, seccomp_ret_kill),
        _SockFilter(bpf_ld_w_abs, 0, 0, 0),  # seccomp_data.nr
    ]
    blocked = [
        29, 30, 31, 64, 65, 66, 68, 69, 70, 71,  # System V IPC
        76, 109, 112, 425,  # truncate, setpgid, setsid, io_uring_setup
    ]
    if not allow_unix_sockets:
        blocked.extend((41, 53))  # socket, socketpair
    for syscall_number in blocked:
        instructions.extend((
            _SockFilter(bpf_jmp_jeq_k, 0, 1, syscall_number),
            _SockFilter(bpf_ret_k, 0, 0, seccomp_ret_errno | 1),
        ))
    if allow_unix_sockets:
        instructions.extend((
            _SockFilter(bpf_jmp_jeq_k, 0, 3, 41),  # socket(2)
            _SockFilter(bpf_ld_w_abs, 0, 0, 16),  # seccomp_data.args[0]: domain
            _SockFilter(bpf_jmp_jeq_k, 1, 0, 1),  # AF_UNIX
            _SockFilter(bpf_ret_k, 0, 0, seccomp_ret_errno | 1),
        ))
    instructions.append(_SockFilter(bpf_ret_k, 0, 0, seccomp_ret_allow))
    filters = (_SockFilter * len(instructions))(*instructions)
    program = _SockFprog(len(filters), filters)
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(22, 2, ctypes.byref(program)) != 0:  # PR_SET_SECCOMP, FILTER
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def _drop_capabilities() -> None:
    libc = ctypes.CDLL(None, use_errno=True)
    for cap in range(64):
        libc.prctl(24, cap, 0, 0, 0)  # PR_CAPBSET_DROP

    class Header(ctypes.Structure):
        _fields_ = [("version", ctypes.c_uint32), ("pid", ctypes.c_int)]

    class Data(ctypes.Structure):
        _fields_ = [("effective", ctypes.c_uint32), ("permitted", ctypes.c_uint32), ("inheritable", ctypes.c_uint32)]

    header = Header(0x20080522, 0)
    data = (Data * 2)()
    if libc.capset(ctypes.byref(header), ctypes.byref(data)) != 0:
        err = ctypes.get_errno()
        raise OSError(err, os.strerror(err))


def restrict_trusted_gpu_process() -> None:
    """Deny network sockets and capabilities while preserving CUDA IPC."""
    _set_no_new_privs(ctypes.CDLL(None, use_errno=True))
    _install_seccomp(allow_unix_sockets=True)
    _drop_capabilities()


def _sandbox_init(spec_path: str, command: Sequence[str]) -> None:
    with open(spec_path, "r", encoding="utf-8") as handle:
        spec = json.load(handle)

    readonly = [path for path in (*SYSTEM_PATHS, *PROC_PATHS, *spec["runtime_roots"]) if os.path.exists(path)]
    writable = [item["source"] for item in spec["binds"] if item["writable"]]
    readonly.extend(item["source"] for item in spec["binds"] if not item["writable"])
    devices = list(BASE_DEVICES)
    visible_gpus = spec["env"].get("CUDA_VISIBLE_DEVICES", "")
    for common in ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset", "/dev/nvidia-caps"):
        if os.path.exists(common):
            devices.append(common)
    for gpu in visible_gpus.split(","):
        gpu = gpu.strip()
        if gpu.isdigit() and os.path.exists(f"/dev/nvidia{gpu}"):
            devices.append(f"/dev/nvidia{gpu}")

    os.chdir(spec["cwd"])
    if spec["landlock"]:
        _apply_landlock(readonly, writable, devices)
    else:
        _set_no_new_privs(ctypes.CDLL(None, use_errno=True))
    _install_seccomp(spec.get("allow_unix_sockets", False))
    if spec["drop_caps"]:
        _drop_capabilities()
    os.umask(0o077)
    os.environ.clear()
    os.environ.update(spec["env"])
    os.execvpe(command[0], list(command), os.environ)


class Sandbox:
    """Execute a command with no public network and an allowlisted filesystem."""

    def __init__(
        self,
        root_dir: str,
        binds: Iterable[Bind],
        cwd: str,
        env: Mapping[str, str] | None = None,
        pass_fds: Sequence[int] = (),
        namespaces: bool = True,
        landlock: bool = True,
        new_session: bool = True,
        drop_caps: bool = True,
        allow_unix_sockets: bool = False,
    ) -> None:
        self.root_dir = os.path.abspath(root_dir)
        self.binds = list(binds)
        self.cwd = os.path.abspath(cwd)
        self.env = self._safe_env(env or {})
        self.pass_fds = tuple(pass_fds)
        self.namespaces = namespaces
        self.landlock = landlock
        self.new_session = new_session
        self.drop_caps = drop_caps
        self.allow_unix_sockets = allow_unix_sockets
        if self.pass_fds and not self.allow_unix_sockets:
            raise ValueError("pass_fds requires allow_unix_sockets=True")
        Path(self.root_dir).mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _safe_env(extra: Mapping[str, str]) -> dict[str, str]:
        runtime_bin = os.path.dirname(os.path.realpath(sys.executable))
        env = {
            "LANG": "C.UTF-8",
            "LC_ALL": "C.UTF-8",
            "PATH": os.pathsep.join((
                runtime_bin,
                "/opt/oracle/bin",
                "/usr/local/sbin",
                "/usr/local/bin",
                "/usr/sbin",
                "/usr/bin",
                "/sbin",
                "/bin",
            )),
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONNOUSERSITE": "1",
        }
        for key in SAFE_ENV_KEYS:
            if key in os.environ:
                env[key] = os.environ[key]
        env.update({key: str(value) for key, value in extra.items()})
        return env

    def run(self, command: Sequence[str], timeout: int) -> RunResult:
        unshare = shutil.which("unshare")
        if self.namespaces and not unshare:
            raise SandboxUnavailable("local backend requires the image's unshare command")

        spec_path = os.path.join(self.root_dir, ".sandbox.json")
        spec = {
            "cwd": self.cwd,
            "env": self.env,
            "binds": [{"source": os.path.abspath(bind.source), "writable": bind.writable} for bind in self.binds],
            "runtime_roots": [sys.prefix],
            "landlock": self.landlock,
            "drop_caps": self.drop_caps,
            "allow_unix_sockets": self.allow_unix_sockets,
        }
        with open(spec_path, "w", encoding="utf-8") as handle:
            json.dump(spec, handle)

        repo_root = str(Path(__file__).resolve().parent.parent)
        init = [
            sys.executable,
            "-m",
            "smdd_runtime.isolation",
            "--init",
            spec_path,
            "--",
            *command,
        ]
        wrapper = init
        if self.namespaces:
            wrapper = [
                unshare,
                "--user",
                "--map-root-user",
                "--net",
                "--pid",
                "--fork",
                "--kill-child=SIGKILL",
                "--ipc",
                "--uts",
                *init,
            ]
        proc = subprocess.Popen(
            wrapper,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            env={"LANG": "C.UTF-8", "PATH": os.environ.get("PATH", "/usr/bin:/bin"), "PYTHONPATH": repo_root},
            pass_fds=self.pass_fds,
            start_new_session=self.new_session,
        )
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired as exc:
            if self.new_session:
                _kill_group(proc)
            else:
                proc.kill()
            proc.communicate()
            raise SandboxTimeout(f"sandbox command timed out after {timeout} seconds") from exc
        # A normal parent exit must not leave a redirected background child.
        if self.new_session:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        return RunResult(stdout, stderr, proc.returncode)


def verify_isolation(
    runtime_dir: str,
    denied_path: str,
    python: str | None = None,
    allow_unix_sockets: bool = False,
) -> None:
    """Verify filesystem and network denial before scheduling any real task."""
    workspace = os.path.join(runtime_dir, "preflight_workspace")
    home = os.path.join(runtime_dir, "preflight_home")
    meta = os.path.join(runtime_dir, "preflight_meta")
    for path in (workspace, home):
        os.makedirs(path, exist_ok=True)
    code = (
        f"p={os.path.abspath(denied_path)!r}; denied=False\n"
        "try:\n open(p).read(1)\n"
        "except PermissionError:\n denied=True\n"
        "assert denied, 'filesystem allowlist is not enforced'\n"
        "import os,socket\n"
        f"allow_unix={allow_unix_sockets!r}\n"
        "if allow_unix:\n"
        " server=socket.socket(socket.AF_UNIX); sock_path=os.path.join(os.environ['HOME'],'preflight.sock')\n"
        " server.bind(sock_path); server.listen(1)\n"
        " client=socket.socket(socket.AF_UNIX); client.connect(sock_path); client.close(); server.close()\n"
        " left,right=socket.socketpair(); left.close(); right.close()\n"
        "else:\n"
        " unix_denied=pair_denied=False\n"
        " try:\n  socket.socket(socket.AF_UNIX)\n"
        " except OSError:\n  unix_denied=True\n"
        " try:\n  socket.socketpair()\n"
        " except OSError:\n  pair_denied=True\n"
        " assert unix_denied and pair_denied, 'Unix sockets are not denied'\n"
        "for domain in (socket.AF_INET,socket.AF_INET6):\n"
        " try:\n  socket.socket(domain)\n"
        " except OSError:\n  continue\n"
        " raise AssertionError(f'network socket domain {domain} is not denied')\n"
    )
    python = python or sys.executable
    sandbox = Sandbox(meta, [Bind(workspace, writable=True), Bind(home, writable=True)], workspace,
                      {"HOME": home, "TMPDIR": home}, allow_unix_sockets=allow_unix_sockets)
    result = sandbox.run([python, "-c", code], 10)
    if result.returncode:
        raise SandboxUnavailable(result.stderr.strip() or "sandbox preflight failed")


def main() -> None:
    if len(sys.argv) < 5 or sys.argv[1] != "--init" or "--" not in sys.argv:
        raise SystemExit("internal sandbox entrypoint")
    separator = sys.argv.index("--")
    _sandbox_init(sys.argv[2], sys.argv[separator + 1 :])


if __name__ == "__main__":
    main()
