"""Tool executor for a daemonless, sandboxed agent container."""

from __future__ import annotations

import hashlib
import os
import shutil
import subprocess
from pathlib import Path

from agent_harness.tools import ToolExecutor
from smdd_runtime.boltz import cache_dir
from smdd_runtime.boltz_worker import boltz_thread_env, request_boltz
from smdd_runtime.isolation import Bind, Sandbox, SandboxTimeout
from smdd_runtime.msa import request_msa


class LocalToolExecutor(ToolExecutor):
    def __init__(self, workspace_dir, runtime_dir, msa_socket, config, boltz_socket=None):
        super().__init__(None, config, workspace_dir=os.path.abspath(workspace_dir))
        self.runtime_dir = os.path.abspath(runtime_dir)
        self.msa_socket = msa_socket
        self.boltz_socket = os.path.abspath(boltz_socket) if boltz_socket else None
        self.home_dir = os.path.join(self.runtime_dir, "home")
        self.temp_dir = os.path.join(self.runtime_dir, "tmp")
        self.msa_dir = os.path.join(self.runtime_dir, "msa")
        self.sandbox_dir = os.path.join(self.runtime_dir, "sandbox")
        for path in (self.home_dir, self.temp_dir, self.msa_dir, self.sandbox_dir):
            os.makedirs(path, exist_ok=True)

        cache = cache_dir()
        cache_link = os.path.join(self.home_dir, ".boltz")
        if os.path.isdir(cache) and not os.path.lexists(cache_link):
            os.symlink(cache, cache_link)

        binds = [
            Bind(self.workspace_dir, writable=True),
            Bind(self.home_dir, writable=True),
            Bind(self.temp_dir, writable=True),
            Bind(self.msa_dir, writable=False),
        ]
        if os.path.isdir(cache):
            binds.append(Bind(cache, writable=False))
        common_env = {
            "HOME": self.home_dir,
            "TMPDIR": self.temp_dir,
            **boltz_thread_env(self.config.boltz_cpu_threads),
        }
        self.sandbox = Sandbox(
            self.sandbox_dir,
            binds,
            self.workspace_dir,
            {**common_env, "CUDA_VISIBLE_DEVICES": ""},
        )
        self.gpu_sandbox = Sandbox(
            os.path.join(self.runtime_dir, "gpu_sandbox"),
            binds,
            self.workspace_dir,
            common_env,
            namespaces=False,
            landlock=False,
            # This path runs only fixed trusted GPU tool commands. NVIDIA's
            # driver requires AF_UNIX during initialization on H800 systems.
            allow_unix_sockets=True,
        )

    def _inside_workspace(self, path):
        return os.path.commonpath((self.workspace_dir, os.path.realpath(path))) == self.workspace_dir

    def _copy_to_container(self, local_path, container_path):
        destination = os.path.abspath(container_path)
        if not self._inside_workspace(os.path.dirname(destination)):
            raise ValueError("tool path escapes the workspace")
        if os.path.lexists(destination) and not self._inside_workspace(destination):
            raise ValueError("tool destination resolves outside the workspace")
        shutil.copyfile(local_path, destination)

    def _copy_from_container(self, container_path, local_path):
        source = os.path.abspath(container_path)
        if not self._inside_workspace(source):
            raise ValueError("tool source resolves outside the workspace")
        shutil.copyfile(source, local_path)

    def _docker_exec(self, cmd, timeout=120, workdir=None):
        try:
            result = self.sandbox.run(cmd, timeout)
        except SandboxTimeout as exc:
            raise subprocess.TimeoutExpired(cmd, timeout) from exc
        return result.stdout + result.stderr, result.returncode

    def _docker_exec_shell(self, shell_cmd, timeout=120, workdir=None):
        return self._docker_exec(["bash", "-c", shell_cmd], timeout, workdir=workdir)

    def _prepare_boltz_input(self, boltz_input, call_number):
        proteins = [item["protein"] for item in boltz_input["sequences"] if "protein" in item]
        msas = request_msa(
            self.msa_socket,
            [protein["sequence"] for protein in proteins],
            timeout=self.config.boltz_timeout,
        )
        call_dir = os.path.join(self.msa_dir, f"call_{call_number}")
        os.makedirs(call_dir, exist_ok=True)
        for protein in proteins:
            sequence = "".join(protein["sequence"].split()).upper()
            name = hashlib.sha256(sequence.encode()).hexdigest()[:16] + ".csv"
            msa_path = os.path.join(call_dir, name)
            Path(msa_path).write_text(msas[sequence], encoding="utf-8")
            protein["msa"] = msa_path
        return boltz_input

    def _use_msa_server(self):
        return False

    def _boltz_num_workers(self):
        return 0

    def _exec_boltz(self, command, timeout):
        if self.boltz_socket:
            try:
                return request_boltz(self.boltz_socket, list(command), timeout)
            except TimeoutError as exc:
                raise subprocess.TimeoutExpired(command, timeout) from exc
        try:
            result = self.gpu_sandbox.run(command, timeout)
        except SandboxTimeout as exc:
            raise subprocess.TimeoutExpired(command, timeout) from exc
        return result.stdout + result.stderr, result.returncode

    def _exec_diffdock(self, command, timeout):
        """Run only the fixed DiffDock wrapper in the trusted GPU sandbox."""
        expected = ["/opt/diffdock/bin/python", "/opt/smdd-tools/diffdock_dock.py"]
        if list(command[:2]) != expected:
            raise ValueError("local DiffDock execution requires the fixed trusted entrypoint")
        try:
            result = self.gpu_sandbox.run(command, timeout)
        except SandboxTimeout as exc:
            raise subprocess.TimeoutExpired(command, timeout) from exc
        return result.stdout + result.stderr, result.returncode

    def _exec_vina(self, command, timeout):
        """Run only the fixed Vina wrapper in the trusted GPU sandbox."""
        expected = ["/opt/vina/bin/python", "/opt/smdd-tools/vina_dock.py"]
        if list(command[:2]) != expected:
            raise ValueError("local Vina execution requires the fixed trusted entrypoint")
        try:
            result = self.gpu_sandbox.run(command, timeout)
        except SandboxTimeout as exc:
            raise subprocess.TimeoutExpired(command, timeout) from exc
        return result.stdout + result.stderr, result.returncode

    def run_submit(self, arguments):
        file_path = str(arguments.get("file_path", "")).strip()
        if file_path.startswith("/workspace/"):
            file_path = file_path[len("/workspace/"):]
        if not file_path or os.path.isabs(file_path):
            return "ERROR: Submission path must be relative to the workspace.", {}

        normalized = os.path.normpath(file_path)
        candidate = os.path.join(self.workspace_dir, normalized)
        if normalized == ".." or normalized.startswith("../") or not self._inside_workspace(candidate):
            return "ERROR: Submission path must stay inside the workspace.", {}
        if not os.path.isfile(candidate):
            return f"ERROR: File not found in workspace: {normalized}", {}
        return f"Submission accepted: {normalized}", {"submitted_file": normalized}
