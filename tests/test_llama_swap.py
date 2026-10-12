import asyncio
import importlib
import json
import os
import signal
import sys
from io import BytesIO
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from prompting_pot import llama_swap as lifecycle

backend = importlib.import_module("prompting_pot.app")


@pytest.fixture
def process(monkeypatch):
    child = Mock(pid=12345, returncode=None)
    child.wait = AsyncMock(return_value=0)
    monkeypatch.setattr(lifecycle.os, "killpg", Mock())
    return child


@pytest.fixture
def startup(tmp_path, monkeypatch, process):
    config = tmp_path / "model.yaml"
    config.write_text("models: {}")
    monkeypatch.setattr(
        lifecycle, "is_llama_swap_ready", AsyncMock(side_effect=[False, True])
    )
    monkeypatch.setattr(lifecycle, "port_is_occupied", AsyncMock(return_value=False))
    monkeypatch.setattr(
        lifecycle.asyncio, "create_subprocess_exec", AsyncMock(return_value=process)
    )
    return lifecycle.LlamaSwap(), config


def test_launch_wait_and_graceful_stop(startup, process):
    manager, config = startup

    async def run():
        assert await manager.start(config) == lifecycle.BASE_URL
        assert manager.process is process
        await manager.stop()
        await manager.stop()  # idempotent

    asyncio.run(run())
    args = lifecycle.asyncio.create_subprocess_exec.call_args
    assert args.args == (
        "llama-swap",
        "--listen",
        f"{lifecycle.IP}:8080",
        "--config",
        str(config),
    )
    assert args.kwargs["stdout"] == asyncio.subprocess.DEVNULL
    assert args.kwargs["stderr"] == asyncio.subprocess.DEVNULL
    lifecycle.os.killpg.assert_called_once_with(process.pid, signal.SIGTERM)
    process.wait.assert_awaited_once()
    assert manager.process is None


@pytest.mark.skipif(os.name != "posix", reason="Linux process-group shutdown")
def test_real_owned_subprocess_graceful_shutdown():
    async def run():
        manager = lifecycle.LlamaSwap()
        manager.process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-c",
            "import signal,sys,time; signal.signal(signal.SIGTERM, lambda *_: sys.exit(0)); print('ready', flush=True); time.sleep(60)",
            stdout=asyncio.subprocess.PIPE,
            start_new_session=True,
        )
        child = manager.process
        try:
            assert (
                await asyncio.wait_for(child.stdout.readline(), timeout=5) == b"ready\n"
            )
        finally:
            await manager.stop()
        assert child.returncode == 0
        assert manager.process is None

    asyncio.run(run())


def test_port_probe(monkeypatch):
    writer = Mock()
    writer.wait_closed = AsyncMock()
    connection = AsyncMock(return_value=(Mock(), writer))
    monkeypatch.setattr(lifecycle.asyncio, "open_connection", connection)
    assert asyncio.run(lifecycle.port_is_occupied()) is True
    writer.close.assert_called_once()
    connection.side_effect = ConnectionRefusedError()
    assert asyncio.run(lifecycle.port_is_occupied()) is False
    connection.side_effect = TimeoutError()
    with pytest.raises(RuntimeError, match="Cannot determine"):
        asyncio.run(lifecycle.port_is_occupied())


def test_external_server_not_owned(startup, monkeypatch):
    manager, config = startup
    monkeypatch.setattr(lifecycle, "is_llama_swap_ready", AsyncMock(return_value=True))

    async def run():
        assert await manager.start(config) == lifecycle.BASE_URL
        await manager.stop()

    asyncio.run(run())
    lifecycle.asyncio.create_subprocess_exec.assert_not_awaited()
    lifecycle.os.killpg.assert_not_called()


def test_occupied_unready_port_does_not_launch(startup, monkeypatch):
    manager, config = startup
    monkeypatch.setattr(lifecycle, "port_is_occupied", AsyncMock(return_value=True))
    with pytest.raises(RuntimeError, match="occupied"):
        asyncio.run(manager.start(config))
    lifecycle.asyncio.create_subprocess_exec.assert_not_awaited()


def test_missing_config_does_not_launch(startup):
    manager, config = startup
    config.unlink()
    with pytest.raises(RuntimeError, match="does not exist"):
        asyncio.run(manager.start(config))
    lifecycle.asyncio.create_subprocess_exec.assert_not_awaited()


def test_startup_timeout_cleans_up(startup, monkeypatch, process):
    manager, config = startup
    monkeypatch.setattr(lifecycle, "STARTUP_TIMEOUT", 0)
    with pytest.raises(RuntimeError, match="did not become ready"):
        asyncio.run(manager.start(config))
    process.wait.assert_awaited_once()
    assert manager.process is None


def test_early_exit_cleans_up(startup, process):
    manager, config = startup
    process.returncode = 1
    with pytest.raises(RuntimeError, match="exited during startup"):
        asyncio.run(manager.start(config))
    assert manager.process is None


def test_cancelled_startup_cleans_up(startup, monkeypatch, process):
    manager, config = startup
    monkeypatch.setattr(
        lifecycle,
        "is_llama_swap_ready",
        AsyncMock(side_effect=[False, asyncio.CancelledError()]),
    )
    with pytest.raises(asyncio.CancelledError):
        asyncio.run(manager.start(config))
    process.wait.assert_awaited_once()
    assert manager.process is None


def test_shutdown_force_kill_fallback(process):
    manager = lifecycle.LlamaSwap()
    manager.process = process
    process.wait.side_effect = [asyncio.TimeoutError(), 0]
    asyncio.run(manager.stop())
    assert lifecycle.os.killpg.call_args_list[0].args == (process.pid, signal.SIGTERM)
    assert lifecycle.os.killpg.call_args_list[1].args == (process.pid, signal.SIGKILL)
    assert manager.process is None


def test_process_already_gone(process):
    manager = lifecycle.LlamaSwap()
    manager.process = process
    lifecycle.os.killpg.side_effect = ProcessLookupError()
    asyncio.run(manager.stop())
    process.wait.assert_awaited_once()
    assert manager.process is None


@pytest.mark.parametrize(
    "payload,ready", [({"data": []}, True), ({}, False), ([], False)]
)
def test_readiness_payload(monkeypatch, payload, ready):
    client = AsyncMock()
    client.get.return_value = httpx.Response(
        200, json=payload, request=httpx.Request("GET", lifecycle.BASE_URL)
    )
    context = Mock()
    context.__aenter__ = AsyncMock(return_value=client)
    context.__aexit__ = AsyncMock(return_value=False)
    monkeypatch.setattr(lifecycle.httpx, "AsyncClient", Mock(return_value=context))
    assert asyncio.run(lifecycle.is_llama_swap_ready()) is ready


@pytest.fixture
def api(tmp_path, monkeypatch):
    (tmp_path / "targets.json").write_text(
        json.dumps(
            [
                {
                    "id": "vision",
                    "name": "Vision",
                    "config": "model.yaml",
                    "vision": True,
                },
            ]
        )
    )
    monkeypatch.setattr(backend, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(backend, "CAPTION_MODEL_ID", "vision")
    manager = lifecycle.LlamaSwap()
    manager.start = AsyncMock(return_value=lifecycle.BASE_URL)
    manager.stop = AsyncMock()
    monkeypatch.setattr(backend, "llama_swap", manager)
    return manager


@pytest.mark.parametrize("fail", [False, True])
def test_generation_cleanup_and_shutdown(api, monkeypatch, fail):
    async def infer(**kwargs):
        assert api.lock.locked()
        api.stop.assert_not_awaited()
        if fail:
            raise NotImplementedError("not connected")
        return "a flower"

    monkeypatch.setattr(backend, "generate_prompt", infer)
    with TestClient(backend.app) as client:
        response = client.post(
            "/api/generate",
            data={"target_id": "vision", "mode": "text", "description": "flower", "image_generator": "krea2"},
        )
        assert response.status_code == (501 if fail else 200)
        api.stop.assert_awaited_once()
        assert not api.lock.locked()
    assert api.stop.await_count == 2  # request cleanup + shutdown fallback


def test_caption_cleanup_only_after_whole_batch(api, monkeypatch):
    buffer = BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, "PNG")
    calls = []

    async def infer(**kwargs):
        assert api.lock.locked()
        api.stop.assert_not_awaited()
        calls.append("caption")
        return "TOK stands beside a flower."

    monkeypatch.setattr(backend, "generate_prompt", infer)
    with TestClient(backend.app) as client:
        response = client.post(
            "/api/caption",
            data={"target_id": "vision", "trigger_word": "TOK"},
            files=[
                ("images", (f"{i}.png", buffer.getvalue(), "image/png"))
                for i in range(2)
            ],
        )
        assert response.status_code == 200
        assert calls == ["caption", "caption"]
        api.start.assert_awaited_once()
        api.stop.assert_awaited_once()


def test_startup_errors_become_503(api):
    api.start.side_effect = RuntimeError("Port 8080 is occupied")
    with TestClient(backend.app) as client:
        response = client.post(
            "/api/generate",
            data={"target_id": "vision", "mode": "text", "description": "flower", "image_generator": "krea2"},
        )
        assert response.status_code == 503
        assert "occupied" in response.json()["detail"]
        api.stop.assert_awaited_once()


def test_inference_sessions_are_serialized(api):
    target = backend.Target("vision", "Vision", backend.CONFIG_DIR / "model.yaml", True)
    events = []

    async def request(name):
        async with backend.inference_session(target):
            events.append(f"{name}:start")
            await asyncio.sleep(0.01)
            events.append(f"{name}:end")

    async def run():
        await asyncio.gather(request("a"), request("b"))

    asyncio.run(run())
    assert events == ["a:start", "a:end", "b:start", "b:end"]
    assert api.stop.await_count == 2
