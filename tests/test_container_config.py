"""Validate Compose contracts without starting containers or requiring a daemon."""

import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def service(tmp_path):
    if shutil.which("docker") is None:
        pytest.skip("Docker CLI/Compose is not installed")
    version = subprocess.run(
        ["docker", "compose", "version"], capture_output=True, check=False, timeout=10
    )
    if version.returncode:
        pytest.skip("Docker Compose plugin is not installed")
    environment = {
        **os.environ,
        "APP_PORT": "18000",
        "MODELS_DIR": str(tmp_path / "models"),
        "CONFIG_DIR": str(tmp_path / "config"),
    }
    result = subprocess.run(
        [
            "docker",
            "compose",
            "-f",
            str(ROOT / "compose.yaml"),
            "config",
            "--format",
            "json",
        ],
        cwd=ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=10,
    )
    return json.loads(result.stdout)["services"]["prompting-pot"]


def test_all_nvidia_gpus_are_reserved(service):
    devices = service["deploy"]["resources"]["reservations"]["devices"]
    assert len(devices) == 1
    assert devices[0]["driver"] == "nvidia"
    # Newer Compose serializes "all" as the Engine API's -1 sentinel.
    assert devices[0]["count"] in {"all", -1}
    assert devices[0]["capabilities"] == ["gpu"]
    assert service["environment"]["NVIDIA_VISIBLE_DEVICES"] == "all"
    assert service["environment"]["NVIDIA_DRIVER_CAPABILITIES"] == "compute,utility"


def test_readonly_mounts_keep_existing_container_paths(service):
    mounts = {volume["target"]: volume for volume in service["volumes"]}
    assert set(mounts) == {"/app/config", "/mnt/llm/models"}
    for mount in mounts.values():
        assert mount["type"] == "bind"
        assert mount["read_only"] is True
        assert mount["bind"].get("create_host_path", False) is False


def test_public_binding_restart_and_shutdown(service):
    assert service["ports"][0]["host_ip"] == "0.0.0.0"
    assert service["ports"][0]["published"] == "18000"
    assert service["ports"][0]["target"] == 8000
    assert service["restart"] == "unless-stopped"
    assert service["init"] is True
    assert service["stop_grace_period"] == "45s"
