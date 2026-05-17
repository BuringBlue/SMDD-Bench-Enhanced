"""Nginx TCP proxy sidecar for MSA server access from agent containers.

Architecture:
    [Agent container] --smdd-internal--> [Nginx proxy] --bridge--> api.colabfold.com:443

The agent container sits on an --internal Docker network (no internet).
The nginx proxy sits on both the internal network and the default bridge,
forwarding TCP on port 443 to api.colabfold.com:443 (Layer 4 passthrough,
TLS is preserved end-to-end). Agent containers use --add-host to resolve
api.colabfold.com to the proxy's internal IP.
"""

import json
import subprocess
import tempfile
import os

COLABFOLD_HOST = "api.colabfold.com"
NETWORK_NAME = "smdd-internal"
PROXY_CONTAINER_NAME = "smdd-msa-proxy"
PROXY_IMAGE = "nginx:alpine"

NGINX_CONFIG = """\
daemon off;
events { worker_connections 512; }
stream {
    resolver 127.0.0.11 valid=300s;
    server {
        listen 443;
        proxy_pass api.colabfold.com:443;
    }
}
"""


def ensure_network():
    """Create the internal Docker network if it doesn't already exist."""
    result = subprocess.run(
        ["docker", "network", "inspect", NETWORK_NAME],
        capture_output=True,
    )
    if result.returncode == 0:
        return  # already exists

    proc = subprocess.run(
        ["docker", "network", "create", "--internal", NETWORK_NAME],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Failed to create network {NETWORK_NAME}: {proc.stderr}")


def get_proxy_ip():
    """Get the proxy container's IP on the internal network."""
    proc = subprocess.run(
        ["docker", "inspect", PROXY_CONTAINER_NAME],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"Proxy container not found: {proc.stderr}")

    info = json.loads(proc.stdout)
    networks = info[0]["NetworkSettings"]["Networks"]
    if NETWORK_NAME not in networks:
        raise RuntimeError(f"Proxy not connected to {NETWORK_NAME}")
    return networks[NETWORK_NAME]["IPAddress"]


def start_proxy():
    """Start the nginx TCP proxy container. Returns the proxy's IP on the internal network."""
    # Check if proxy is already running
    result = subprocess.run(
        ["docker", "inspect", "-f", "{{.State.Running}}", PROXY_CONTAINER_NAME],
        capture_output=True, text=True,
    )
    if result.returncode == 0 and result.stdout.strip() == "true":
        return get_proxy_ip()

    # Clean up any stopped proxy container
    subprocess.run(
        ["docker", "rm", "-f", PROXY_CONTAINER_NAME],
        capture_output=True,
    )

    # Write nginx config to a temp file
    tmpdir = tempfile.mkdtemp(prefix="smdd-nginx-")
    config_path = os.path.join(tmpdir, "nginx.conf")
    with open(config_path, "w") as f:
        f.write(NGINX_CONFIG)

    try:
        # Create the proxy container on the default bridge
        proc = subprocess.run(
            [
                "docker", "create",
                "--name", PROXY_CONTAINER_NAME,
                PROXY_IMAGE,
                "nginx", "-c", "/etc/nginx/nginx.conf",
            ],
            capture_output=True, text=True,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"Failed to create proxy container: {proc.stderr}")

        # Copy nginx config into the container
        subprocess.run(
            ["docker", "cp", config_path, f"{PROXY_CONTAINER_NAME}:/etc/nginx/nginx.conf"],
            check=True, capture_output=True,
        )

        # Connect proxy to the internal network
        subprocess.run(
            ["docker", "network", "connect", NETWORK_NAME, PROXY_CONTAINER_NAME],
            check=True, capture_output=True, text=True,
        )

        # Start the proxy
        subprocess.run(
            ["docker", "start", PROXY_CONTAINER_NAME],
            check=True, capture_output=True,
        )

        return get_proxy_ip()

    finally:
        os.unlink(config_path)
        os.rmdir(tmpdir)


def stop_proxy():
    """Stop and remove the proxy container."""
    subprocess.run(["docker", "stop", PROXY_CONTAINER_NAME], capture_output=True)
    subprocess.run(["docker", "rm", PROXY_CONTAINER_NAME], capture_output=True)
