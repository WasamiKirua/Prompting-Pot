import importlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError, APIStatusError, APITimeoutError

from prompting_pot import text_generation
from prompting_pot.prompts import IMAGE_GENERATORS, build_text_prompt

backend = importlib.import_module("prompting_pot.app")


@pytest.fixture
def api(tmp_path, monkeypatch):
    (tmp_path / "targets.json").write_text(
        json.dumps(
            [
                {"id": model, "name": model, "vision": True, "config": model + ".yaml"}
                for model in ["GemmaUnseen", "Captioning", "Fibo"]
            ]
        )
    )
    monkeypatch.setattr(backend, "CONFIG_DIR", tmp_path)
    start = AsyncMock(return_value="http://localhost:8080/v1")
    stop = AsyncMock()
    monkeypatch.setattr(backend, "ensure_llama_swap", start)
    monkeypatch.setattr(backend.llama_swap, "stop", stop)
    client = Mock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="A red boat on calm water."),
                    finish_reason="stop",
                )
            ]
        )
    )
    factory = Mock()
    context = Mock()
    context.__aenter__ = AsyncMock(return_value=client)

    async def close(*args):
        await factory.call_args.kwargs["http_client"].aclose()

    context.__aexit__ = AsyncMock(side_effect=close)
    factory.return_value = context
    monkeypatch.setattr(text_generation, "AsyncOpenAI", factory)
    return TestClient(backend.app), client, start, stop


@pytest.mark.parametrize("model", ["GemmaUnseen", "Captioning"])
@pytest.mark.parametrize("generator", [g for g in IMAGE_GENERATORS if g != "ideogram4"])
def test_all_prose_text_targets(api, model, generator):
    http, sdk, start, stop = api
    response = http.post(
        "/api/generate",
        data={
            "target_id": model,
            "mode": "text",
            "image_generator": generator,
            "description": " A red boat. ",
        },
    )
    assert response.status_code == 200
    assert response.json()["output"] == "A red boat on calm water."
    request = sdk.chat.completions.create.call_args.kwargs
    assert request["model"] == model
    assert request["messages"] == [
        {"role": "system", "content": build_text_prompt(generator)},
        {"role": "user", "content": "A red boat."},
    ]
    assert "response_format" not in request
    assert request["extra_body"] == {"reasoning": "off"}
    start.assert_awaited_once()
    stop.assert_awaited_once()


@pytest.mark.parametrize("model", ["Captioning", "Fibo"])
def test_text_json_paths(api, model):
    http, sdk, _, _ = api
    if model == "Fibo":
        scene = {"general": {"description": "A red boat."}, "objects": []}
    else:
        scene = {
            "high_level_description": "A red boat.",
            "style_description": {
                "aesthetics": "warm",
                "lighting": "sunlight",
                "photo": "wide framing",
                "medium": "photograph",
            },
            "compositional_deconstruction": {
                "background": "Calm water.",
                "elements": [{"type": "obj", "desc": "A red boat."}],
            },
        }
    sdk.chat.completions.create.return_value.choices[0].message.content = json.dumps(
        scene
    )
    response = http.post(
        "/api/generate",
        data={
            "target_id": model,
            "mode": "text",
            "image_generator": "ideogram4",
            "description": "A red boat.",
        },
    )
    assert response.status_code == 200
    output = response.json()["output"]
    assert (output if isinstance(output, dict) else json.loads(output)) == scene
    request = sdk.chat.completions.create.call_args.kwargs
    if model == "Fibo":
        assert request["messages"] == [
            {
                "role": "user",
                "content": "<expand><full>\n<aspect_ratio>1.00</aspect_ratio>\nA red boat.",
            }
        ]
        assert "response_format" not in request
        assert request["extra_body"]["special"] is True
    else:
        assert request["response_format"]["type"] == "json_schema"


@pytest.mark.parametrize(
    "model,generator",
    [
        ("Fibo", "krea2"),
        ("GemmaUnseen", "ideogram4"),
        ("Captioning", ""),
        ("GemmaUnseen", "bad"),
    ],
)
def test_incompatible_text_requests_rejected_before_startup(api, model, generator):
    http, _, start, _ = api
    response = http.post(
        "/api/generate",
        data={
            "target_id": model,
            "mode": "text",
            "image_generator": generator,
            "description": "A boat.",
        },
    )
    assert response.status_code == 422
    start.assert_not_awaited()


@pytest.mark.parametrize(
    "model,failure",
    [
        (model, failure)
        for model in ["GemmaUnseen", "Captioning", "Fibo"]
        for failure in ["empty", "truncated", "invalid-json"]
        if not (model == "GemmaUnseen" and failure == "invalid-json")
    ],
)
def test_text_failures_become_502_and_clean_up(api, model, failure):
    http, sdk, _, stop = api
    choice = sdk.chat.completions.create.return_value.choices[0]
    choice.message.content = "" if failure == "empty" else "not JSON"
    choice.finish_reason = "length" if failure == "truncated" else "stop"
    response = http.post(
        "/api/generate",
        data={
            "target_id": model,
            "mode": "text",
            "image_generator": "krea2" if model == "GemmaUnseen" else "ideogram4",
            "description": "A boat.",
        },
    )
    assert response.status_code == 502
    stop.assert_awaited_once()


@pytest.mark.parametrize("error", ["connection", "timeout", "status", "malformed"])
def test_transport_errors_and_malformed_response(api, error):
    http, sdk, _, stop = api
    request = httpx.Request("POST", "http://localhost/v1/chat/completions")
    if error == "malformed":
        sdk.chat.completions.create.return_value = SimpleNamespace(choices=[])
    else:
        errors = {
            "connection": APIConnectionError(request=request),
            "timeout": APITimeoutError(request=request),
            "status": APIStatusError(
                "bad response", response=httpx.Response(500, request=request), body=None
            ),
        }
        sdk.chat.completions.create.side_effect = errors[error]
    response = http.post(
        "/api/generate",
        data={
            "target_id": "GemmaUnseen",
            "mode": "text",
            "image_generator": "krea2",
            "description": "A boat.",
        },
    )
    assert response.status_code == 502
    stop.assert_awaited_once()


def test_text_does_not_accept_uploaded_image(api):
    http, _, start, _ = api
    response = http.post(
        "/api/generate",
        data={
            "target_id": "GemmaUnseen",
            "mode": "text",
            "image_generator": "krea2",
            "description": "A boat.",
        },
        files={"image": ("test.png", b"image", "image/png")},
    )
    assert response.status_code == 422
    start.assert_not_awaited()
