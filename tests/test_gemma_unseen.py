import asyncio
import base64
import importlib
import json
from io import BytesIO
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError, APIStatusError, APITimeoutError
from PIL import Image

from prompting_pot import gemma_unseen
from prompting_pot.prompts import (
    IMAGE_GENERATORS,
    IMAGE_SYSTEM_PROMPTS,
    QWEN_IMAGE_2_1_SYSTEM_PROMPT,
)

backend = importlib.import_module("prompting_pot.app")


@pytest.fixture
def sdk(monkeypatch):
    client = Mock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content="a polished prompt"),
                    finish_reason="stop",
                ),
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
    monkeypatch.setattr(gemma_unseen, "AsyncOpenAI", factory)
    return client, factory


def call(generator="krea2", description="make it pink"):
    return asyncio.run(
        gemma_unseen.generate_from_image(
            base_url="http://127.0.0.1:8080/v1",
            model="GemmaUnseen",
            system_prompt=IMAGE_SYSTEM_PROMPTS[generator],
            description=description,
            image=b"reference bytes",
            content_type="image/png",
            image_generator=generator,
        )
    )


def test_real_sdk_against_local_fake_api():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(
                (
                    self.path,
                    json.loads(self.rfile.read(int(self.headers["Content-Length"]))),
                )
            )
            response = json.dumps(
                {
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "GemmaUnseen",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": "a real SDK response",
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            result = asyncio.run(
                gemma_unseen.generate_from_image(
                    base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    model="GemmaUnseen",
                    system_prompt=IMAGE_SYSTEM_PROMPTS["krea2"],
                    description="make it pink",
                    image=b"image",
                    content_type="image/png",
                    image_generator="krea2",
                )
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
    assert result == "a real SDK response"
    route, body = received[0]
    assert route == "/v1/chat/completions"
    assert body["model"] == "GemmaUnseen"
    assert body["reasoning"] == "off"
    assert body["messages"][1]["content"][0]["text"] == "make it pink"


def test_qwen_generation_template_is_standalone_and_keeps_output_sections():
    prompt = QWEN_IMAGE_2_1_SYSTEM_PROMPT
    assert IMAGE_SYSTEM_PROMPTS["qwen-image-2.1"] == prompt
    assert "must be usable without any input image" in prompt
    assert "- Mode: Text-to-image" in prompt
    assert "no fixed word count" in prompt
    assert (
        "Incorporate the user's requested changes into the description of the finished scene"
        in prompt
    )
    for obsolete in (
        "ratio_follow",
        "<image1>",
        "<image2>",
        "Image reference / edit rules:",
        "250–500",
    ):
        assert obsolete not in prompt
    for section in (
        "**Optimization Breakdown:**",
        "**Ready-to-Use Qwen-Image-2.1 Prompt:**",
        "**Tweak Suggestions:**",
    ):
        assert prompt.count(section) == 1
    assert "wh_ratio" not in prompt.split("```text\n", 1)[1].split("```", 1)[0]


def test_qwen_generation_rules_cover_grounding_and_specific_visual_details():
    prompt = QWEN_IMAGE_2_1_SYSTEM_PROMPT
    for requirement in (
        "negative space",
        "contact shadows",
        "dominant and accent colors",
        "original script inside double quotes",
        "Do not invent or guess unreadable lettering",
        "Do not force photographic realism onto artwork",
        "positive visual descriptions",
    ):
        assert requirement in prompt


@pytest.mark.parametrize("generator", IMAGE_SYSTEM_PROMPTS)
def test_payload_and_reference_settings(sdk, generator):
    client, factory = sdk
    assert call(generator) == "a polished prompt"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "GemmaUnseen"
    assert kwargs["messages"][0] == {
        "role": "system",
        "content": IMAGE_SYSTEM_PROMPTS[generator],
    }
    user = kwargs["messages"][1]["content"]
    assert user[0] == {"type": "text", "text": "make it pink"}
    assert user[1]["image_url"]["url"] == "data:image/png;base64," + base64.b64encode(
        b"reference bytes"
    ).decode("ascii")
    assert kwargs["temperature"] == 0.7
    assert kwargs["max_tokens"] == 2000
    assert kwargs["extra_body"] == {"reasoning": "off"}
    assert factory.call_args.kwargs["timeout"] == 300.0
    assert factory.call_args.kwargs["max_retries"] == 0
    assert factory.call_args.kwargs["http_client"].is_closed


@pytest.mark.parametrize("description", ["", "  ", "as-is", "Recreate As Is"])
def test_optional_changes_default_to_as_is(sdk, description):
    client, _ = sdk
    call("anima", description)
    text = client.chat.completions.create.call_args.kwargs["messages"][1]["content"][0][
        "text"
    ]
    assert text == "Recreate this reference image as-is as a anima prompt."


@pytest.mark.parametrize(
    "text",
    [
        "An illustration in blue.",
        "**Prompt Breakdown:**\n- Light\n```text\na flower\n```",
        '{"prompt": "a flower"}',
    ],
)
def test_preserve_output_text(sdk, text):
    client, _ = sdk
    client.chat.completions.create.return_value.choices[0].message.content = text
    assert call() == text


@pytest.mark.parametrize(
    "choices",
    [
        [],
        [SimpleNamespace(message=SimpleNamespace(content=""), finish_reason="stop")],
        [
            SimpleNamespace(
                message=SimpleNamespace(content=None, reasoning_content="thinking"),
                finish_reason="length",
            )
        ],
    ],
)
def test_bad_responses_report_errors(sdk, choices):
    client, _ = sdk
    client.chat.completions.create.return_value.choices = choices
    with pytest.raises(RuntimeError):
        call()


@pytest.mark.parametrize(
    "kind,match",
    [
        ("connection", "Could not connect"),
        ("timeout", "timed out"),
        ("status", "HTTP 500"),
    ],
)
def test_sdk_errors(sdk, kind, match):
    client, _ = sdk
    request = httpx.Request("POST", "http://127.0.0.1:8080/v1/chat/completions")
    errors = {
        "connection": APIConnectionError(request=request),
        "timeout": APITimeoutError(request=request),
        "status": APIStatusError(
            "bad response", response=httpx.Response(500, request=request), body=None
        ),
    }
    client.chat.completions.create.side_effect = errors[kind]
    with pytest.raises(RuntimeError, match=match):
        call()


@pytest.fixture
def api(tmp_path, monkeypatch):
    (tmp_path / "targets.json").write_text(
        json.dumps(
            [
                {
                    "id": "GemmaUnseen",
                    "name": "Gemma Unseen",
                    "config": "unseen.yaml",
                    "vision": True,
                },
            ]
        )
    )
    monkeypatch.setattr(backend, "CONFIG_DIR", tmp_path)
    start = AsyncMock(return_value="http://127.0.0.1:8080/v1")
    stop = AsyncMock()
    monkeypatch.setattr(backend, "ensure_llama_swap", start)
    monkeypatch.setattr(backend.llama_swap, "stop", stop)
    return TestClient(backend.app), start, stop


def upload():
    image = BytesIO()
    Image.new("RGB", (2, 2)).save(image, "PNG")
    return {"image": ("image.png", image.getvalue(), "image/png")}


@pytest.mark.parametrize("generator", IMAGE_SYSTEM_PROMPTS)
def test_image_endpoint_calls_inference(api, sdk, generator):
    http, start, stop = api
    client, _ = sdk
    response = http.post(
        "/api/generate",
        data={
            "target_id": "GemmaUnseen",
            "mode": "image",
            "image_generator": generator,
            "description": "change the background to blue",
        },
        files=upload(),
    )
    assert response.status_code == 200
    assert response.json() == {"output": "a polished prompt"}
    kwargs = client.chat.completions.create.call_args.kwargs
    assert kwargs["model"] == "GemmaUnseen"
    assert kwargs["messages"][0]["content"] == IMAGE_SYSTEM_PROMPTS[generator]
    assert (
        kwargs["messages"][1]["content"][0]["text"] == "change the background to blue"
    )
    start.assert_awaited_once()
    stop.assert_awaited_once()


@pytest.mark.parametrize("generator", ["", "unknown", "GemmaUnseen"])
def test_invalid_generator_does_not_start_model(api, generator):
    http, start, stop = api
    response = http.post(
        "/api/generate",
        data={
            "target_id": "GemmaUnseen",
            "mode": "image",
            "image_generator": generator,
        },
        files=upload(),
    )
    assert response.status_code == 422
    start.assert_not_awaited()
    stop.assert_not_awaited()


def test_inference_failure_becomes_502_and_cleans_up(api, sdk):
    http, _, stop = api
    client, _ = sdk
    client.chat.completions.create.return_value.choices = []
    response = http.post(
        "/api/generate",
        data={
            "target_id": "GemmaUnseen",
            "mode": "image",
            "image_generator": "krea2",
        },
        files=upload(),
    )
    assert response.status_code == 502
    assert "unexpected" in response.json()["detail"]
    stop.assert_awaited_once()


def test_generator_choices_are_independent_of_local_models(api):
    http, _, _ = api
    assert http.get("/api/image-generators").json() == [
        {"id": key, "name": value} for key, value in IMAGE_GENERATORS.items()
    ]
    assert http.get("/api/targets").json()[0]["id"] == "GemmaUnseen"
