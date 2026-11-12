"""Persistent, one-GPU Boltz inference worker for trusted local backends.

The worker loads the Boltz structure and affinity checkpoints once. Prediction
requests are serialized and successfully used models stay on the worker's GPU.
If retained weights or inference exhaust GPU memory, the worker automatically
switches to a one-model-at-a-time CPU fallback and retries the request once.
"""

from __future__ import annotations

import argparse
import ctypes
import gc
import json
import os
import signal
import socket
import struct
import subprocess
import time
from dataclasses import asdict
from pathlib import Path


MAX_REQUEST_BYTES = 1_000_000
MAX_RESPONSE_BYTES = 64_000_000
BOLTZ_SOCKET_BACKLOG = 128
BOLTZ_EXECUTABLE = "/opt/oracle/bin/boltz"


def _recv_exact(conn: socket.socket, length: int) -> bytes:
    chunks = []
    remaining = length
    while remaining:
        chunk = conn.recv(min(remaining, 1 << 20))
        if not chunk:
            raise ConnectionError("unexpected end of Boltz worker message")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _recv_message(conn: socket.socket, limit: int) -> dict:
    header = _recv_exact(conn, 8)
    length = struct.unpack("!Q", header)[0]
    if length > limit:
        raise ValueError(f"message exceeds {limit} bytes")
    return json.loads(_recv_exact(conn, length).decode("utf-8"))


def _send_message(conn: socket.socket, payload: dict) -> None:
    data = json.dumps(payload).encode("utf-8")
    conn.sendall(struct.pack("!Q", len(data)) + data)


def boltz_thread_env(threads: int = 16) -> dict[str, str]:
    """Return consistent native-library limits for one Boltz worker."""
    value = str(max(1, int(threads)))
    return {
        "OMP_NUM_THREADS": value,
        "MKL_NUM_THREADS": value,
        "OPENBLAS_NUM_THREADS": value,
        "NUMEXPR_NUM_THREADS": value,
        "MALLOC_ARENA_MAX": value,
        "TOKENIZERS_PARALLELISM": "false",
    }


def cuda_smoke_test(torch_module=None) -> dict[str, object]:
    """Prove that the worker's selected CUDA device can execute an operation."""
    torch = torch_module
    if torch is None:
        import torch as torch_module_import

        torch = torch_module_import
    if not torch.cuda.is_available() or torch.cuda.device_count() < 1:
        raise RuntimeError(
            "No supported CUDA GPU is visible inside the Boltz worker; "
            "check --gpus/CUDA_VISIBLE_DEVICES and the container GPU runtime"
        )
    try:
        value = torch.ones(1, device="cuda:0")
        value.add_(1)
        torch.cuda.synchronize(0)
        if value.item() != 2:
            raise RuntimeError("CUDA smoke-test calculation returned an invalid value")
        properties = torch.cuda.get_device_properties(0)
        info = {
            "name": properties.name,
            "total_memory": int(properties.total_memory),
            "visible_devices": int(torch.cuda.device_count()),
        }
        del value
        torch.cuda.empty_cache()
        return info
    except Exception as exc:
        raise RuntimeError(f"Boltz CUDA smoke test failed: {type(exc).__name__}: {exc}") from exc


class _ResidentModels:
    """Own immutable Boltz weights and serve them through the upstream CLI."""

    def __init__(self, cache: str, threads: int) -> None:
        # These imports are intentionally delayed.  The supervisor can import
        # this module without importing torch or allocating a CUDA context.
        import torch
        from boltz import main as boltz_main
        from boltz.model.models.boltz2 import Boltz2

        self.cache = os.path.realpath(cache)
        self.torch = torch
        self.boltz_main = boltz_main
        self.model_class = Boltz2
        self._original_loader = Boltz2.load_from_checkpoint
        self.models: dict[str, object] = {}
        self.active_model = None
        self.low_memory_mode = False

        torch.set_num_threads(max(1, threads))
        try:
            torch.set_num_interop_threads(max(1, min(4, threads)))
        except RuntimeError:
            # PyTorch permits setting this only before inter-op work starts.
            pass

        self._load_models()
        self._install_cached_loader()

    def _load_models(self) -> None:
        main = self.boltz_main
        cache = Path(self.cache)
        diffusion_args = asdict(main.Boltz2DiffusionParams())
        pairformer_args = asdict(main.PairformerArgsV2())
        msa_args = asdict(main.MSAModuleArgs(
            # Match the installed Click command's effective default.  The
            # callback signature says True, but the is_flag option defaults
            # to False for actual CLI invocations made by the harness.
            subsample_msa=False,
            num_subsampled_msa=1024,
            use_paired_feature=True,
        ))

        structure_predict_args = {
            "recycling_steps": 3,
            "sampling_steps": 200,
            "diffusion_samples": 1,
            "max_parallel_samples": 2,
            "write_confidence_summary": True,
            "write_full_pae": False,
            "write_full_pde": False,
        }
        structure_steering = main.BoltzSteeringParams()
        structure_steering.fk_steering = True
        structure_steering.physical_guidance_update = True

        affinity_predict_args = {
            "recycling_steps": 5,
            "sampling_steps": 200,
            "diffusion_samples": 1,
            "max_parallel_samples": 1,
            "write_confidence_summary": False,
            "write_full_pae": False,
            "write_full_pde": False,
        }
        affinity_steering = main.BoltzSteeringParams()
        affinity_steering.fk_steering = False
        affinity_steering.physical_guidance_update = False
        affinity_steering.contact_guidance_update = False

        structure_path = cache / "boltz2_conf.ckpt"
        affinity_path = cache / "boltz2_aff.ckpt"
        print(f"[boltz-worker] loading structure checkpoint: {structure_path}", flush=True)
        structure = self._original_loader(
            structure_path,
            strict=True,
            predict_args=structure_predict_args,
            map_location="cpu",
            diffusion_process_args=diffusion_args,
            ema=False,
            use_kernels=True,
            pairformer_args=pairformer_args,
            msa_args=msa_args,
            steering_args=asdict(structure_steering),
        )
        structure.eval().cpu()
        self.models[os.path.realpath(structure_path)] = structure
        print("[boltz-worker] structure checkpoint ready on CPU", flush=True)

        print(f"[boltz-worker] loading affinity checkpoint: {affinity_path}", flush=True)
        affinity = self._original_loader(
            affinity_path,
            strict=True,
            predict_args=affinity_predict_args,
            map_location="cpu",
            diffusion_process_args=diffusion_args,
            ema=False,
            pairformer_args=pairformer_args,
            msa_args=msa_args,
            steering_args=asdict(affinity_steering),
            affinity_mw_correction=False,
        )
        affinity.eval().cpu()
        self.models[os.path.realpath(affinity_path)] = affinity
        print("[boltz-worker] affinity checkpoint ready on CPU", flush=True)

    def _install_cached_loader(self) -> None:
        resident = self

        @classmethod
        def cached_load_from_checkpoint(cls, checkpoint_path, *args, **kwargs):
            del cls, args
            path = os.path.realpath(os.fspath(checkpoint_path))
            try:
                model = resident.models[path]
            except KeyError as exc:
                raise ValueError(f"persistent worker has no cached checkpoint: {path}") from exc

            if (
                resident.low_memory_mode
                and resident.active_model is not None
                and resident.active_model is not model
            ):
                resident.active_model.cpu()
                resident.torch.cuda.empty_cache()

            # These are the request-scoped settings used directly by
            # prediction_step.  Model architecture and weights remain fixed.
            if "predict_args" in kwargs:
                model.predict_args = dict(kwargs["predict_args"])
            if "steering_args" in kwargs:
                model.steering_args = dict(kwargs["steering_args"])
            if "affinity_mw_correction" in kwargs:
                model.affinity_mw_correction = kwargs["affinity_mw_correction"]
            if "use_kernels" in kwargs:
                model.use_kernels = kwargs["use_kernels"]
            model.eval()
            resident.active_model = model
            return model

        self.model_class.load_from_checkpoint = cached_load_from_checkpoint

    def _clear_trainer_references(self) -> None:
        for model in self.models.values():
            # Lightning replaces this reference on the next CLI invocation.
            # Keeping it would retain callbacks and request-scoped data.
            if hasattr(model, "_trainer"):
                model._trainer = None

    def _is_cuda_oom(self, exc: BaseException | None, output: str = "") -> bool:
        if exc is not None and isinstance(exc, self.torch.cuda.OutOfMemoryError):
            return True
        detail = f"{type(exc).__name__}: {exc}\n{output}".lower()
        return "cuda out of memory" in detail or "cuda error: out of memory" in detail

    def _retain_on_gpu(self) -> None:
        """Retain weights when possible, permanently degrading after an OOM."""
        targets = (
            [self.active_model]
            if self.low_memory_mode and self.active_model is not None
            else list(self.models.values())
        )
        if self.low_memory_mode:
            for model in self.models.values():
                if model is not self.active_model:
                    model.cpu()
        try:
            for model in targets:
                model.eval().cuda()
            self.torch.cuda.synchronize()
        except Exception as exc:
            if not self._is_cuda_oom(exc):
                raise
            print(
                "[boltz-worker] insufficient memory for retained weights; "
                "falling back to CPU-backed one-model-at-a-time mode",
                flush=True,
            )
            self.low_memory_mode = True
            self.release_to_cpu()

    def release_to_cpu(self) -> None:
        """Leave no resident model or stale Trainer state on the GPU."""
        for model in self.models.values():
            model.cpu()
            model.eval()
        self._clear_trainer_references()
        self.active_model = None
        gc.collect()
        self.torch.cuda.empty_cache()

    @staticmethod
    def _validate_command(command: list[str]) -> tuple[str, str]:
        # Each trusted supervisor pins the one CLI belonging to its Python
        # environment.  Agent workers use /opt/oracle; evaluator workers use
        # /opt/conda.  The value is never accepted from a task request.
        expected_boltz = os.path.realpath(
            os.environ.get("SMDD_BOLTZ_EXECUTABLE", BOLTZ_EXECUTABLE)
        )
        if (
            len(command) < 5
            or os.path.realpath(command[0]) != expected_boltz
            or command[1] != "predict"
        ):
            raise ValueError("Boltz worker accepts only the fixed predict command")
        value_options = {
            "--out_dir",
            "--diffusion_samples",
            "--seed",
            "--diffusion_samples_affinity",
            "--max_parallel_samples",
            "--num_workers",
        }
        flag_options = {"--use_potentials"}
        parsed = {}
        index = 3
        while index < len(command):
            option = command[index]
            if option in flag_options:
                parsed[option] = True
                index += 1
            elif option in value_options and index + 1 < len(command):
                parsed[option] = command[index + 1]
                index += 2
            else:
                raise ValueError(f"unsupported persistent Boltz option: {option}")
        if parsed.get("--use_potentials") is not True:
            raise ValueError("persistent Boltz requires --use_potentials")
        if (
            parsed.get("--max_parallel_samples") not in {"1", "2"}
            or parsed.get("--num_workers") != "0"
        ):
            raise ValueError("persistent Boltz configuration does not match the harness")
        data_path = os.path.realpath(command[2])
        try:
            out_path = os.path.realpath(parsed["--out_dir"])
        except KeyError as exc:
            raise ValueError("Boltz predict command is missing --out_dir") from exc
        workspace = os.path.dirname(data_path)
        # Evaluator-side MSA attachment writes the final YAML under a private
        # .msa directory next to the requested output directory.
        if os.path.basename(workspace) == ".msa":
            workspace = os.path.dirname(workspace)
        if not os.path.isfile(data_path):
            raise ValueError(f"Boltz input does not exist: {data_path}")
        if os.path.commonpath((workspace, out_path)) != workspace:
            raise ValueError("Boltz output directory escapes the task workspace")
        return data_path, out_path

    def predict(self, command: list[str]) -> tuple[str, int]:
        from click.testing import CliRunner

        self._validate_command(command)
        cli_args = [*command[1:], "--cache", self.cache]
        result = None
        try:
            result = CliRunner().invoke(self.boltz_main.cli, cli_args)
            output = result.output
            if result.exception is not None and result.exit_code != 0:
                output += (
                    f"\n{type(result.exception).__name__}: "
                    f"{result.exception}\n"
                )

            if result.exit_code != 0 and self._is_cuda_oom(result.exception, output):
                print(
                    "[boltz-worker] CUDA OOM during prediction; moving cached "
                    "models to CPU and retrying once",
                    flush=True,
                )
                self.low_memory_mode = True
                failed_output = output
                # Click retains the exception traceback on Result. Drop it
                # before emptying CUDA so failed request tensors can be freed.
                result.exception = None
                result.exc_info = None
                result = None
                self.release_to_cpu()
                retry = CliRunner().invoke(self.boltz_main.cli, cli_args)
                retry_output = retry.output
                if retry.exception is not None and retry.exit_code != 0:
                    retry_output += (
                        f"\n{type(retry.exception).__name__}: "
                        f"{retry.exception}\n"
                    )
                if retry.exit_code == 0:
                    output = (
                        "[boltz-worker] recovered from CUDA OOM using "
                        "CPU-backed model residency\n" + retry_output
                    )
                else:
                    output = (
                        failed_output
                        + "\n[boltz-worker] retry after CUDA OOM\n"
                        + retry_output
                    )
                result = retry

            self._clear_trainer_references()
            if result.exit_code == 0:
                self._retain_on_gpu()
            else:
                self.release_to_cpu()
            return output, result.exit_code
        except Exception:
            self.release_to_cpu()
            raise


def request_boltz(socket_path: str, command: list[str], timeout: int) -> tuple[str, int]:
    """Run one serialized prediction through a resident worker."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(timeout)
        conn.connect(socket_path)
        _send_message(conn, {"op": "predict", "command": command})
        response = _recv_message(conn, MAX_RESPONSE_BYTES)
    return str(response.get("output", "")), int(response.get("returncode", 1))


def _serve(socket_path: str, cache: str, threads: int, lazy: bool = False) -> None:
    from smdd_runtime.isolation import restrict_trusted_gpu_process

    # Do not leave a multi-gigabyte GPU worker behind if run_local is killed.
    parent_pid = os.getppid()
    libc = ctypes.CDLL(None, use_errno=True)
    if libc.prctl(1, signal.SIGTERM) != 0:  # PR_SET_PDEATHSIG
        error = ctypes.get_errno()
        raise OSError(error, os.strerror(error))
    if os.getppid() != parent_pid:
        raise SystemExit("Boltz worker parent exited during startup")

    # This process runs only fixed Boltz code.  It needs AF_UNIX for its local
    # request socket and NVIDIA initialization, but it never needs a network.
    restrict_trusted_gpu_process()
    cuda_info = cuda_smoke_test()
    print(
        "[boltz-worker] CUDA preflight passed: "
        f"{cuda_info['name']} ({cuda_info['total_memory'] // (1024 ** 2)} MiB)",
        flush=True,
    )
    resident = None if lazy else _ResidentModels(cache, threads)
    path = Path(socket_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        path.unlink()
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as server:
        server.bind(socket_path)
        os.chmod(socket_path, 0o600)
        # Multiple agent processes assigned to this GPU may connect at once.
        # Keep their connections queued while this process serializes GPU work.
        server.listen(BOLTZ_SOCKET_BACKLOG)
        print(f"[boltz-worker] ready: {socket_path}", flush=True)
        while True:
            conn, _ = server.accept()
            with conn:
                try:
                    request = _recv_message(conn, MAX_REQUEST_BYTES)
                    if request.get("op") == "ping":
                        response = {"status": "ready"}
                    elif request.get("op") == "predict":
                        if resident is None:
                            print("[boltz-worker] first eligible request; loading models", flush=True)
                            resident = _ResidentModels(cache, threads)
                        output, returncode = resident.predict(request["command"])
                        response = {"output": output, "returncode": returncode}
                    else:
                        raise ValueError("unsupported Boltz worker operation")
                except Exception as exc:
                    response = {
                        "output": f"{type(exc).__name__}: {exc}",
                        "returncode": 1,
                    }
                try:
                    _send_message(conn, response)
                except (BrokenPipeError, ConnectionError):
                    pass


class BoltzService:
    """Lifecycle wrapper for one persistent Boltz process on one GPU."""

    def __init__(
        self,
        socket_path: str,
        cache: str,
        gpu_id: int,
        log_path: str,
        python: str = "/opt/oracle/bin/python",
        threads: int = 16,
        startup_timeout: int = 3600,
        jit_cache_dir: str | None = None,
        lazy: bool = False,
        boltz_executable: str | None = None,
    ) -> None:
        self.socket_path = os.path.abspath(socket_path)
        self.cache = os.path.abspath(cache)
        self.gpu_id = int(gpu_id)
        self.log_path = os.path.abspath(log_path)
        self.python = python
        self.threads = int(threads)
        self.startup_timeout = int(startup_timeout)
        self.jit_cache_dir = os.path.abspath(jit_cache_dir) if jit_cache_dir else None
        self.lazy = bool(lazy)
        self.boltz_executable = os.path.abspath(
            boltz_executable
            or os.path.join(os.path.dirname(os.path.abspath(self.python)), "boltz")
        )
        self.process: subprocess.Popen | None = None
        self._log_handle = None

    def _worker_env(self) -> dict[str, str]:
        """Build the worker environment with evaluator-private JIT caches."""
        repo_root = str(Path(__file__).resolve().parent.parent)
        # The worker needs CUDA/runtime settings but never model API keys or
        # task-facing credentials.  Start from a small allowlist instead of
        # inheriting the supervisor's complete environment.
        allowed = (
            "LANG",
            "LC_ALL",
            "PATH",
            "LD_LIBRARY_PATH",
            "NVIDIA_VISIBLE_DEVICES",
            "NVIDIA_DRIVER_CAPABILITIES",
            "CUDA_MODULE_LOADING",
            "HOME",
            "TMPDIR",
            "PYTHONPATH",
        )
        env = {key: os.environ[key] for key in allowed if key in os.environ}
        env.update(boltz_thread_env(self.threads))
        env.update({
            "CUDA_VISIBLE_DEVICES": str(self.gpu_id),
            "PYTHONPATH": os.pathsep.join(
                value for value in (repo_root, env.get("PYTHONPATH", "")) if value
            ),
            "PYTHONUNBUFFERED": "1",
            "SMDD_BOLTZ_EXECUTABLE": self.boltz_executable,
        })
        if self.jit_cache_dir:
            os.makedirs(self.jit_cache_dir, mode=0o700, exist_ok=True)
            os.chmod(self.jit_cache_dir, 0o700)
            cache_paths = {
                "TRITON_CACHE_DIR": os.path.join(self.jit_cache_dir, "triton"),
                "CUDA_CACHE_PATH": os.path.join(self.jit_cache_dir, "cuda"),
                "TORCHINDUCTOR_CACHE_DIR": os.path.join(self.jit_cache_dir, "torchinductor"),
                "CUEQ_TRITON_CACHE_DIR": os.path.join(self.jit_cache_dir, "cueq"),
                "XDG_CACHE_HOME": os.path.join(self.jit_cache_dir, "xdg"),
                "TMPDIR": os.path.join(self.jit_cache_dir, "tmp"),
            }
            for path in cache_paths.values():
                os.makedirs(path, mode=0o700, exist_ok=True)
                os.chmod(path, 0o700)
            env.update(cache_paths)
            # Catch any library cache that does not honor the explicit cache
            # variables above without exposing the evaluator supervisor's home.
            env["HOME"] = self.jit_cache_dir
        return env

    def start(self) -> str:
        env = self._worker_env()
        os.makedirs(os.path.dirname(self.log_path), exist_ok=True)
        self._log_handle = open(self.log_path, "a", encoding="utf-8")
        command = [
            self.python,
            "-m",
            "smdd_runtime.boltz_worker",
            "serve",
            "--socket", self.socket_path,
            "--cache", self.cache,
            "--threads", str(self.threads),
        ]
        if self.lazy:
            command.append("--lazy")
        self.process = subprocess.Popen(
            command,
            stdout=self._log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True,
        )
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                code = self.process.returncode
                self.close()
                raise RuntimeError(
                    f"Boltz worker for GPU {self.gpu_id} failed to start with code {code}; "
                    f"see {self.log_path}"
                )
            if os.path.exists(self.socket_path):
                try:
                    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                        conn.settimeout(2)
                        conn.connect(self.socket_path)
                        _send_message(conn, {"op": "ping"})
                        response = _recv_message(conn, MAX_RESPONSE_BYTES)
                    if response.get("status") == "ready":
                        return self.socket_path
                except (ConnectionError, OSError, ValueError):
                    pass
            time.sleep(0.2)
        self.close()
        raise RuntimeError(
            f"Boltz worker for GPU {self.gpu_id} was not ready after "
            f"{self.startup_timeout} seconds; see {self.log_path}"
        )

    def close(self) -> None:
        if self.process and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
                self.process.wait(timeout=10)
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
        if self._log_handle is not None:
            self._log_handle.close()
            self._log_handle = None

    def __enter__(self) -> str:
        return self.start()

    def __exit__(self, *_):
        self.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("serve", nargs="?")
    parser.add_argument("--socket", required=True)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--threads", type=int, default=16)
    parser.add_argument("--lazy", action="store_true")
    args = parser.parse_args()
    _serve(args.socket, args.cache, args.threads, lazy=args.lazy)


if __name__ == "__main__":
    main()
