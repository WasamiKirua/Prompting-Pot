import asyncio
import base64
import importlib
import json
from io import BytesIO
from unittest.mock import AsyncMock
from zipfile import ZipFile

import pytest
from fastapi import Request, UploadFile
from fastapi.testclient import TestClient
from PIL import Image

from prompting_pot.progress import caption_progress_response

backend = importlib.import_module("prompting_pot.app")


@pytest.fixture
def api(tmp_path, monkeypatch):
    (tmp_path / "targets.json").write_text(
        json.dumps(
            [
                {
                    "id": "Captioning",
                    "name": "Qwen",
                    "config": "caption.yaml",
                    "vision": True,
                }
            ]
        )
    )
    monkeypatch.setattr(backend, "CONFIG_DIR", tmp_path)
    start = AsyncMock(return_value="http://localhost:8080/v1")
    stop = AsyncMock()
    monkeypatch.setattr(backend, "ensure_llama_swap", start)
    monkeypatch.setattr(backend.llama_swap, "stop", stop)
    monkeypatch.setattr(
        backend,
        "generate_prompt",
        AsyncMock(return_value="TOK wears a jacket beside a window."),
    )
    return TestClient(backend.app), start, stop


def image(name, content=None):
    if content is None:
        buffer = BytesIO()
        Image.new("RGB", (2, 2)).save(buffer, "PNG")
        content = buffer.getvalue()
    return "images", (name, content, "image/png")


def test_stream_reports_captioned_skipped_failed_and_final_zip(api):
    http, start, stop = api
    response = http.post(
        "/api/caption",
        headers={"Accept": "application/x-ndjson"},
        data={
            "target_id": "Captioning",
            "trigger_word": "TOK",
            "existing_captions": '["kept.txt"]',
        },
        files=[image("kept.png"), image("new.png"), image("bad.png", b"invalid")],
    )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/x-ndjson")
    events = [json.loads(line) for line in response.text.splitlines()]
    updates = [event for event in events if event["type"] == "progress"]
    assert [event["processed"] for event in updates] == [1, 2, 3]
    assert all(event["total"] == 3 for event in updates)
    assert updates[0]["counts"] == {"captioned": 0, "skipped": 1, "failed": 0}
    assert updates[-1]["counts"] == {"captioned": 1, "skipped": 1, "failed": 1}
    final = events[-1]
    assert final["type"] == "complete"
    assert final["counts"] == updates[-1]["counts"]
    with ZipFile(BytesIO(base64.b64decode(final["archive"]))) as archive:
        assert set(archive.namelist()) == {"new.txt", "caption-manifest.json"}
    start.assert_awaited_once()
    stop.assert_awaited_once()


def test_all_skipped_stream_never_starts_model(api):
    http, start, _ = api
    response = http.post(
        "/api/caption",
        headers={"Accept": "application/x-ndjson"},
        data={
            "target_id": "Captioning",
            "trigger_word": "TOK",
            "existing_captions": '["kept.txt"]',
        },
        files=[image("kept.png")],
    )
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1]["type"] == "complete"
    assert events[-1]["counts"]["skipped"] == 1
    start.assert_not_awaited()


def test_stream_startup_failure_is_terminal_error(api):
    http, start, stop = api
    start.side_effect = RuntimeError("Model cannot start")
    response = http.post(
        "/api/caption",
        headers={"Accept": "application/x-ndjson"},
        data={"target_id": "Captioning", "trigger_word": "TOK"},
        files=[image("test.png")],
    )
    events = [json.loads(line) for line in response.text.splitlines()]
    assert events[-1] == {"type": "error", "message": "Model cannot start"}
    assert not any(event["type"] == "complete" for event in events)
    stop.assert_awaited_once()


def test_invalid_stream_input_still_returns_422(api):
    http, start, _ = api
    response = http.post(
        "/api/caption",
        headers={"Accept": "application/x-ndjson"},
        data={"target_id": "Captioning", "trigger_word": ""},
        files=[image("test.png")],
    )
    assert response.status_code == 422
    start.assert_not_awaited()


def test_disconnecting_active_batch_releases_model_lock_and_uploads(api, monkeypatch):
    _, _, stop = api

    async def check():
        started = asyncio.Event()

        async def infer(**kwargs):
            started.set()
            await asyncio.sleep(60)

        monkeypatch.setattr(backend, "generate_prompt", infer)
        upload = UploadFile(file=BytesIO(image("test.png")[1][1]), filename="test.png")
        response = await backend.caption_folder(
            request=Request(
                {"type": "http", "headers": [(b"accept", b"application/x-ndjson")]}
            ),
            target_id="Captioning",
            images=[upload],
            existing_captions="[]",
            caption_style="character",
            trigger_word="TOK",
        )
        iterator = response.body_iterator
        await anext(iterator)
        await anext(iterator)
        await started.wait()
        assert backend.llama_swap.lock.locked()
        await iterator.aclose()
        assert not backend.llama_swap.lock.locked()
        assert upload.file.closed
        stop.assert_awaited_once()

    asyncio.run(check())


def test_disconnecting_stream_cancels_worker_and_runs_finally():
    async def check():
        cleaned = asyncio.Event()

        async def run(emit):
            try:
                await emit({"type": "progress", "processed": 1, "total": 2})
                await asyncio.sleep(60)
                return b"", {}
            finally:
                cleaned.set()

        response = caption_progress_response(run)
        iterator = response.body_iterator
        assert json.loads(await anext(iterator))["type"] == "stage"
        assert json.loads(await anext(iterator))["type"] == "progress"
        await iterator.aclose()
        assert cleaned.is_set()

    asyncio.run(check())


@pytest.mark.parametrize(
    "error",
    [
        RuntimeError("Inference failed"),
        ValueError("Invalid batch"),
        OSError("Read failed"),
        Exception("Unexpected failure"),
    ],
)
def test_runner_errors_end_the_stream(error):
    async def check():
        async def run(emit):
            raise error

        response = caption_progress_response(run)
        events = [json.loads(line) async for line in response.body_iterator]
        assert events[-1]["type"] == "error"
        assert not any(event["type"] == "complete" for event in events)

    asyncio.run(check())
