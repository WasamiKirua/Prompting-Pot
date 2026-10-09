import asyncio
import base64
import importlib
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock
from zipfile import ZipFile

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError, APIStatusError, APITimeoutError
from PIL import Image

from prompting_pot import captioning
from prompting_pot.prompts import (
    CAPTION_PROMPT,
    CAPTIONING_TAG_PROMPT,
    QWEN_IMAGE_2_1_TAG_CAPTION_PROMPT,
    TAGGED_CAPTION_STYLES,
    build_caption_request,
)

backend = importlib.import_module("prompting_pot.app")


def completion(text="35mm film photograph of a café"):
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=text))]
    )


@pytest.fixture
def sdk(monkeypatch):
    client = Mock()
    client.chat.completions.create = AsyncMock(return_value=completion())
    factory = Mock()
    context = Mock()
    context.__aenter__ = AsyncMock(return_value=client)

    async def close(*args):
        await factory.call_args.kwargs["http_client"].aclose()

    context.__aexit__ = AsyncMock(side_effect=close)
    factory.return_value = context
    monkeypatch.setattr(captioning, "AsyncOpenAI", factory)
    return client, factory


def call(style="standard", trigger=""):
    prompt, instruction = build_caption_request(style, trigger)
    return asyncio.run(
        captioning.generate_caption(
            base_url="http://local-server:8080/v1",
            model="Captioning",
            system_prompt=prompt,
            instruction=instruction,
            image=b"image bytes",
            content_type="image/bmp",
            caption_style=style,
        )
    )


@pytest.mark.parametrize(
    "style,trigger",
    [
        ("standard", ""),
        ("tagged", "alex cooper"),
        ("tagged", "café character"),
        ("qwen-image-2.1-tagged", "alex cooper"),
        ("qwen-image-2.1-tagged", "Café Character"),
    ],
)
def test_caption_payload_and_server_defaults(sdk, style, trigger):
    client, factory = sdk
    assert call(style, trigger) == "35mm film photograph of a café"
    kwargs = client.chat.completions.create.call_args.kwargs
    assert set(kwargs) == {
        "model",
        "messages",
    }  # No image-prompt sampling/reasoning overrides.
    assert kwargs["model"] == "Captioning"
    prompt, instruction = build_caption_request(style, trigger)
    assert kwargs["messages"][0] == {"role": "system", "content": prompt}
    content = kwargs["messages"][1]["content"]
    assert content[0]["image_url"][
        "url"
    ] == "data:image/bmp;base64," + base64.b64encode(b"image bytes").decode("ascii")
    assert content[1] == {"type": "text", "text": instruction}
    if style in TAGGED_CAPTION_STYLES:
        assert prompt == (
            QWEN_IMAGE_2_1_TAG_CAPTION_PROMPT
            if style == "qwen-image-2.1-tagged"
            else CAPTIONING_TAG_PROMPT
        )
        assert (
            instruction
            == f"Start the captioning with this exact trigger word: '{trigger}'"
        )
    else:
        assert prompt == CAPTION_PROMPT
        assert instruction == "Caption this image following the system instructions."
    settings = factory.call_args.kwargs
    assert settings["base_url"] == "http://local-server:8080/v1"
    assert settings["timeout"] == 300.0
    assert settings["max_retries"] == 0
    assert settings["http_client"].is_closed


@pytest.mark.parametrize(
    "kind,match",
    [
        ("connection", "Could not connect"),
        ("timeout", "timed out"),
        ("status", "HTTP 500"),
    ],
)
def test_caption_sdk_errors(sdk, kind, match):
    client, _ = sdk
    request = httpx.Request("POST", "http://local-server/v1/chat/completions")
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


@pytest.mark.parametrize(
    "result",
    [
        completion(""),
        completion("  "),
        completion(None),
        SimpleNamespace(choices=[]),
        SimpleNamespace(choices=[SimpleNamespace()]),
    ],
)
def test_empty_or_malformed_caption_does_not_succeed(sdk, result):
    client, _ = sdk
    client.chat.completions.create.return_value = result
    with pytest.raises(RuntimeError):
        call()


@pytest.fixture
def api(tmp_path, monkeypatch):
    (tmp_path / "targets.json").write_text(
        json.dumps(
            [
                {
                    "id": "Captioning",
                    "name": "Qwen3.5 Captioning",
                    "config": "caption.yaml",
                    "vision": True,
                },
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
    start = AsyncMock(return_value="http://custom-address:8080/v1")
    stop = AsyncMock()
    monkeypatch.setattr(backend, "ensure_llama_swap", start)
    monkeypatch.setattr(backend.llama_swap, "stop", stop)
    return TestClient(backend.app), start, stop


@pytest.mark.parametrize(
    "style",
    ["standard", "tagged", "qwen-image-2.1", "qwen-image-2.1-tagged", "ideogram4"],
)
def test_gemma_is_rejected_for_all_caption_styles(api, style):
    http, start, stop = api
    response = http.post(
        "/api/caption",
        data={
            "target_id": "GemmaUnseen",
            "caption_style": style,
            "trigger_word": "a character",
        },
        files=[image()],
    )
    assert response.status_code == 422
    assert "Qwen3.5 Captioning" in response.json()["detail"]
    start.assert_not_awaited()
    stop.assert_not_awaited()


def test_caption_model_metadata(api):
    http, _, _ = api
    models = {model["id"]: model for model in http.get("/api/targets").json()}
    assert models["Captioning"]["captioning"] is True
    assert models["GemmaUnseen"]["captioning"] is False


def image(name="image.png", format="PNG"):
    buffer = BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format)
    return (
        "images",
        (name, buffer.getvalue(), "image/bmp" if format == "BMP" else "image/png"),
    )


@pytest.mark.parametrize(
    "style,trigger",
    [
        ("standard", "ignored"),
        ("tagged", "alex cooper"),
        ("qwen-image-2.1-tagged", "alex cooper"),
        ("qwen-image-2.1-tagged", "Café O'Connor"),
    ],
)
def test_endpoint_caption_styles_skip_and_bmp(api, sdk, style, trigger):
    http, start, stop = api
    client, factory = sdk
    text = (
        f"{trigger}, portrait, soft light"
        if style in TAGGED_CAPTION_STYLES
        else "35mm film photograph, soft light"
    )
    client.chat.completions.create.return_value = completion(text)
    response = http.post(
        "/api/caption",
        data={
            "target_id": "Captioning",
            "caption_style": style,
            "trigger_word": trigger,
            "existing_captions": '["kept.txt"]',
        },
        files=[image("kept.png"), image("new.bmp", "BMP"), image("other.png")],
    )
    assert response.status_code == 200
    assert response.headers["X-captioned"] == "2"
    assert response.headers["X-skipped"] == "1"
    start.assert_awaited_once()
    stop.assert_awaited_once()
    assert client.chat.completions.create.await_count == 2
    assert factory.call_args.kwargs["base_url"] == "http://custom-address:8080/v1"
    message = client.chat.completions.create.call_args.kwargs["messages"]
    assert message[0]["content"] == (
        QWEN_IMAGE_2_1_TAG_CAPTION_PROMPT
        if style == "qwen-image-2.1-tagged"
        else CAPTIONING_TAG_PROMPT if style == "tagged" else CAPTION_PROMPT
    )
    assert message[1]["content"][1]["text"] == build_caption_request(style, trigger)[1]
    with ZipFile(BytesIO(response.content)) as archive:
        assert "kept.txt" not in archive.namelist()
        assert archive.read("new.txt").decode("utf-8") == text + "\n"
        assert archive.read("other.txt").decode("utf-8") == text + "\n"


@pytest.mark.parametrize(
    "style,trigger",
    [
        ("unknown", ""),
        ("tagged", ""),
        ("tagged", "  "),
        ("tagged", "a\nb"),
        ("tagged", "a" * 201),
        ("qwen-image-2.1-tagged", ""),
        ("qwen-image-2.1-tagged", "  "),
        ("qwen-image-2.1-tagged", "a\nb"),
        ("qwen-image-2.1-tagged", "a\rb"),
        ("qwen-image-2.1-tagged", "a" * 201),
    ],
)
def test_caption_validation_before_startup(api, style, trigger):
    http, start, stop = api
    response = http.post(
        "/api/caption",
        data={
            "target_id": "Captioning",
            "caption_style": style,
            "trigger_word": trigger,
        },
        files=[image()],
    )
    assert response.status_code == 422
    start.assert_not_awaited()
    stop.assert_not_awaited()


def test_one_empty_caption_does_not_stop_remaining_images(api, sdk):
    http, start, stop = api
    client, _ = sdk
    client.chat.completions.create.side_effect = [
        completion(""),
        completion("valid caption"),
    ]
    response = http.post(
        "/api/caption",
        data={"target_id": "Captioning"},
        files=[image("bad.png"), image("good.png")],
    )
    assert response.status_code == 200
    assert response.headers["X-captioned"] == "1"
    assert response.headers["X-failed"] == "1"
    start.assert_awaited_once()
    stop.assert_awaited_once()
    with ZipFile(BytesIO(response.content)) as archive:
        assert "bad.txt" not in archive.namelist()
        assert archive.read("good.txt") == b"valid caption\n"
        assert (
            json.loads(archive.read("caption-manifest.json"))["files"][0]["status"]
            == "failed"
        )


def test_bmp_is_not_accepted_by_single_image_prompt_endpoint(api):
    http, start, _ = api
    _, (_, content, content_type) = image("image.bmp", "BMP")
    response = http.post(
        "/api/generate",
        data={
            "target_id": "Captioning",
            "mode": "image",
            "image_generator": "krea2",
        },
        files={"image": ("image.bmp", content, content_type)},
    )
    assert response.status_code == 422
    start.assert_not_awaited()
