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
from prompting_pot.prompts import FOLDER_CAPTION_STYLES, build_caption_request

backend = importlib.import_module("prompting_pot.app")


def completion(text="TOK wears a blue jacket beside a window."):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")
        ]
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
    monkeypatch.setattr(
        importlib.import_module("prompting_pot.gemma_unseen"), "AsyncOpenAI", factory
    )
    return client, factory


def call(style="character", trigger="TOK"):
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


@pytest.mark.parametrize("style", ["character", "style"])
@pytest.mark.parametrize("trigger", ["TOK", "Café O'Connor"])
def test_caption_payload_and_server_defaults(sdk, style, trigger):
    client, factory = sdk
    assert call(style, trigger) == "TOK wears a blue jacket beside a window."
    kwargs = client.chat.completions.create.call_args.kwargs
    assert set(kwargs) == {"model", "messages"}
    assert kwargs["model"] == "Captioning"
    prompt, instruction = build_caption_request(style, trigger)
    assert kwargs["messages"][0] == {"role": "system", "content": prompt}
    content = kwargs["messages"][1]["content"]
    assert content[0]["image_url"][
        "url"
    ] == "data:image/bmp;base64," + base64.b64encode(b"image bytes").decode("ascii")
    assert content[1] == {"type": "text", "text": instruction}
    assert prompt == FOLDER_CAPTION_STYLES[style]
    assert instruction.startswith(
        f"Begin with this exact trigger word followed by a space: {trigger}\n"
    )
    settings = factory.call_args.kwargs
    assert settings["base_url"] == "http://local-server:8080/v1"
    assert settings["timeout"] == 300.0
    assert settings["max_retries"] == 0
    assert settings["http_client"].is_closed


@pytest.mark.parametrize("style", ["character", "style"])
def test_krea2_prompt_contract(style):
    prompt = FOLDER_CAPTION_STYLES[style]
    assert "natural, complete English sentences" in prompt
    assert "never comma-separated tags" in prompt
    assert "exact user-supplied trigger word followed by a space" in prompt
    if style == "character":
        assert "Do not describe the face" in prompt
        assert "fixed identity features" in prompt
        assert "outfit" in prompt and "pose/action" in prompt
    else:
        assert "visible content normally" in prompt
        assert "consistent aesthetic across the dataset carry the style" in prompt
        assert "Visible facial features may be described" in prompt
        assert "Do not describe the face" not in prompt


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
                {"id": "Fibo", "name": "Fibo", "config": "fibo.yaml", "vision": True},
            ]
        )
    )
    monkeypatch.setattr(backend, "CONFIG_DIR", tmp_path)
    start = AsyncMock(return_value="http://custom-address:8080/v1")
    stop = AsyncMock()
    monkeypatch.setattr(backend, "ensure_llama_swap", start)
    monkeypatch.setattr(backend.llama_swap, "stop", stop)
    return TestClient(backend.app), start, stop


def image(name="image.png", format="PNG"):
    buffer = BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format)
    return "images", (
        name,
        buffer.getvalue(),
        "image/bmp" if format == "BMP" else "image/png",
    )


@pytest.mark.parametrize("model", ["Fibo", "missing"])
@pytest.mark.parametrize("style", ["character", "style"])
def test_other_models_are_rejected_for_captioning(api, model, style):
    http, start, stop = api
    response = http.post(
        "/api/caption",
        data={"target_id": model, "caption_style": style, "trigger_word": "TOK"},
        files=[image()],
    )
    assert response.status_code == 422
    assert "Qwen3.5 Captioning" in response.json()["detail"]
    start.assert_not_awaited()
    stop.assert_not_awaited()


@pytest.mark.parametrize("model", ["Captioning", "GemmaUnseen"])
def test_caption_models_require_vision(api, model):
    http, start, _ = api
    path = backend.CONFIG_DIR / "targets.json"
    targets = json.loads(path.read_text())
    next(target for target in targets if target["id"] == model)["vision"] = False
    path.write_text(json.dumps(targets))
    metadata = {target["id"]: target for target in http.get("/api/targets").json()}
    assert metadata[model]["captioning"] is False
    response = http.post(
        "/api/caption",
        data={"target_id": model, "trigger_word": "TOK"},
        files=[image()],
    )
    assert response.status_code == 422
    start.assert_not_awaited()


def test_caption_model_metadata(api):
    http, _, _ = api
    models = {model["id"]: model for model in http.get("/api/targets").json()}
    assert models["Captioning"]["captioning"] is True
    assert models["GemmaUnseen"]["captioning"] is True
    assert models["Fibo"]["captioning"] is False


@pytest.mark.parametrize("model", ["Captioning", "GemmaUnseen"])
@pytest.mark.parametrize("style", ["character", "style"])
@pytest.mark.parametrize("trigger", ["TOK", "Café O'Connor"])
def test_endpoint_caption_types_skip_and_bmp(api, sdk, model, style, trigger):
    http, start, stop = api
    client, factory = sdk
    text = f"{trigger} wears a blue jacket beside a window. Soft light falls from the left."
    client.chat.completions.create.return_value = completion(text)
    response = http.post(
        "/api/caption",
        data={
            "target_id": model,
            "caption_style": style,
            "trigger_word": f"  {trigger}  ",
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
    assert message[0]["content"] == FOLDER_CAPTION_STYLES[style]
    instruction_index = 0 if model == "GemmaUnseen" else 1
    assert (
        message[1]["content"][instruction_index]["text"]
        == build_caption_request(style, trigger)[1]
    )
    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == model
    if model == "GemmaUnseen":
        assert request["extra_body"] == {"reasoning": "off"}
        assert request["max_tokens"] == 2000
        assert request["temperature"] == 0.7
    else:
        assert set(request) == {"model", "messages"}
    with ZipFile(BytesIO(response.content)) as archive:
        assert "kept.txt" not in archive.namelist()
        assert archive.read("new.txt").decode("utf-8") == text + "\n"
        assert archive.read("other.txt").decode("utf-8") == text + "\n"


@pytest.mark.parametrize(
    "style",
    [
        "unknown",
        "standard",
        "tagged",
        "qwen-image-2.1",
        "qwen-image-2.1-tagged",
        "ideogram4",
    ],
)
def test_removed_folder_formats_are_rejected_before_startup(api, style):
    http, start, stop = api
    response = http.post(
        "/api/caption",
        data={"target_id": "Captioning", "caption_style": style, "trigger_word": "TOK"},
        files=[image()],
    )
    assert response.status_code == 422
    assert "Character or Style" in response.json()["detail"]
    start.assert_not_awaited()
    stop.assert_not_awaited()


@pytest.mark.parametrize("style", ["character", "style"])
@pytest.mark.parametrize(
    "trigger", ["", "  ", "a\nb", "a\rb", "TOK\n", "\rTOK", "a" * 201]
)
def test_trigger_validation_before_startup(api, style, trigger):
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


@pytest.mark.parametrize("model", ["Captioning", "GemmaUnseen"])
@pytest.mark.parametrize(
    "invalid", ["", "Wrong wears a coat.", "TOKother wears a coat.", "TOK", "TOK   "]
)
def test_invalid_trigger_output_does_not_stop_remaining_images(
    api, sdk, model, invalid
):
    http, start, stop = api
    client, _ = sdk
    client.chat.completions.create.side_effect = [
        completion(invalid),
        completion("TOK stands near a window."),
    ]
    response = http.post(
        "/api/caption",
        data={"target_id": model, "trigger_word": "TOK"},
        files=[image("bad.png"), image("good.png")],
    )
    assert response.status_code == 200
    assert response.headers["X-captioned"] == "1"
    assert response.headers["X-failed"] == "1"
    start.assert_awaited_once()
    stop.assert_awaited_once()
    with ZipFile(BytesIO(response.content)) as archive:
        assert "bad.txt" not in archive.namelist()
        assert archive.read("good.txt") == b"TOK stands near a window.\n"
        assert (
            json.loads(archive.read("caption-manifest.json"))["files"][0]["status"]
            == "failed"
        )


def test_caption_manifest_image_is_safe_with_txt_sidecars(api, sdk):
    http, _, _ = api
    response = http.post(
        "/api/caption",
        data={"target_id": "Captioning", "trigger_word": "TOK"},
        files=[image("caption-manifest.png")],
    )
    assert response.status_code == 200
    with ZipFile(BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {
            "caption-manifest.txt",
            "caption-manifest.json",
        }


def test_existing_json_is_not_a_folder_caption(api):
    http, start, _ = api
    response = http.post(
        "/api/caption",
        data={
            "target_id": "Captioning",
            "trigger_word": "TOK",
            "existing_captions": '["image.json"]',
        },
        files=[image()],
    )
    assert response.status_code == 422
    start.assert_not_awaited()


def test_bmp_is_not_accepted_by_single_image_prompt_endpoint(api):
    http, start, _ = api
    _, (_, content, content_type) = image("image.bmp", "BMP")
    response = http.post(
        "/api/generate",
        data={"target_id": "Captioning", "mode": "image", "image_generator": "krea2"},
        files={"image": ("image.bmp", content, content_type)},
    )
    assert response.status_code == 422
    start.assert_not_awaited()
