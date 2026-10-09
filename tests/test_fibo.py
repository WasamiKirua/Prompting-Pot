import asyncio
import base64
import importlib
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import httpx
import pytest
from fastapi.testclient import TestClient
from openai import APIConnectionError, APIStatusError, APITimeoutError
from PIL import Image

from prompting_pot import fibo

backend = importlib.import_module("prompting_pot.app")


def image_bytes():
    buffer = BytesIO()
    Image.new("RGB", (3, 2), "red").save(buffer, "JPEG")
    return buffer.getvalue()


@pytest.fixture
def sdk(monkeypatch):
    client = Mock()
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(
                        content='{"general":{"description":"A scene"},"objects":[]}'
                    ),
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
    monkeypatch.setattr(fibo, "AsyncOpenAI", factory)
    return client, factory


def call():
    return asyncio.run(
        fibo.analyze_image(
            base_url="http://localhost:8080/v1", model="Fibo", image=image_bytes()
        )
    )


def test_reference_payload(sdk):
    client, factory = sdk
    assert call() == {"general": {"description": "A scene"}, "objects": []}
    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == "Fibo"
    assert len(request["messages"]) == 1
    user = request["messages"][0]
    assert user["role"] == "user"
    assert user["content"][1] == {"type": "text", "text": "<caption><full>"}
    url = user["content"][0]["image_url"]["url"]
    assert url.startswith("data:image/png;base64,")
    with Image.open(BytesIO(base64.b64decode(url.split(",")[1]))) as image:
        assert image.format == "PNG" and image.size == (3, 2) and image.mode == "RGB"
    assert request["max_tokens"] == 4096
    assert request["temperature"] == 1.0 and request["top_p"] == 0.95
    assert request["stop"] == ["<|im_end|>", "<|endoftext|>"]
    assert request["extra_body"] == {
        "skip_special_tokens": False,
        "special": True,
        "chat_template_kwargs": {"enable_thinking": False},
        "top_k": 20,
        "min_p": 0.0,
    }
    assert "response_format" not in request
    assert factory.call_args.kwargs["max_retries"] == 0
    assert factory.call_args.kwargs["http_client"].is_closed


def test_recursive_native_token_conversion():
    color = {"red": "<color_255>", "green": "<color_42>", "blue": "<color_0>"}
    box = {
        "top_left": {"x": "<coord_1.5>", "y": "<coord_2.0>"},
        "bottom_right": {"x": 50, "y": 90},
    }
    raw = {
        "general": {"color_palette": [color]},
        "objects": [
            {
                "kind": "person",
                "name": "A person",
                "skin_color": color,
                "face_bbox": box,
                "keypoints": [
                    {"name": "head", "point": {"x": "<coord_5.5>", "y": "<coord_10.0>"}}
                ],
                "clothing": [
                    {
                        "kind": "object",
                        "name": "shirt",
                        "colors": [color],
                        "bounding_boxes": [box],
                    }
                ],
                "children": [
                    {
                        "kind": "text",
                        "name": "sign",
                        "lines": [{"text": "HELLO", "bounding_boxes": [box]}],
                    }
                ],
            }
        ],
    }
    result = fibo.parse_scene(json.dumps(raw))
    person = result["objects"][0]
    assert result["general"]["color_palette"][0] == {"red": 255, "green": 42, "blue": 0}
    assert person["clothing"][0]["bounding_boxes"][0]["top_left"] == {
        "x": 1.5,
        "y": 2.0,
    }
    assert person["face_bbox"]["top_left"]["x"] == 1.5
    assert person["keypoints"][0]["point"]["x"] == 5.5
    assert (
        person["children"][0]["lines"][0]["bounding_boxes"][0]["top_left"]["y"] == 2.0
    )
    assert "<coord_" not in json.dumps(result) and "<color_" not in json.dumps(result)


@pytest.mark.parametrize(
    "text",
    [
        "not JSON",
        "{}",
        "[]",
        '{"general":{},"general":{}}',
        '{"general":{"aesthetic_score":NaN}}',
        '{"general":{"aesthetic_score":11}}',
        '{"general":{"color_palette":[{"red":"<color_bad>","green":0,"blue":0}]}}',
        '{"general":',
    ],
)
def test_invalid_native_json(text):
    with pytest.raises(RuntimeError, match="invalid native"):
        fibo.parse_scene(text)


def test_reference_stop_tokens_and_think_prefix():
    assert fibo.parse_scene('<think>trace</think>\n{"general":{}}<|im_end|>') == {
        "general": {}
    }


@pytest.mark.parametrize(
    "choices",
    [
        [],
        [SimpleNamespace(message=SimpleNamespace(content=None), finish_reason="stop")],
        [
            SimpleNamespace(
                message=SimpleNamespace(content='{"general":{}}'),
                finish_reason="length",
            )
        ],
    ],
)
def test_bad_completion(sdk, choices):
    sdk[0].chat.completions.create.return_value.choices = choices
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
    request = httpx.Request("POST", "http://localhost:8080/v1/chat/completions")
    errors = {
        "connection": APIConnectionError(request=request),
        "timeout": APITimeoutError(request=request),
        "status": APIStatusError(
            "bad", response=httpx.Response(500, request=request), body=None
        ),
    }
    sdk[0].chat.completions.create.side_effect = errors[kind]
    with pytest.raises(RuntimeError, match=match):
        call()


@pytest.fixture
def api(tmp_path, monkeypatch):
    (tmp_path / "targets.json").write_text(
        json.dumps(
            [
                {"id": name, "name": name, "config": f"{name}.yaml", "vision": True}
                for name in ("Fibo", "GemmaUnseen", "Captioning")
            ]
        )
    )
    monkeypatch.setattr(backend, "CONFIG_DIR", tmp_path)
    start = AsyncMock(return_value="http://localhost:8080/v1")
    stop = AsyncMock()
    monkeypatch.setattr(backend, "ensure_llama_swap", start)
    monkeypatch.setattr(backend.llama_swap, "stop", stop)
    return TestClient(backend.app), start, stop


def upload():
    return {"image": ("image.jpg", image_bytes(), "image/jpeg")}


def test_endpoint_routes_fibo_and_cleans_up(api, sdk):
    http, start, stop = api
    response = http.post(
        "/api/generate",
        data={"target_id": "Fibo", "mode": "image", "image_generator": "ideogram4"},
        files=upload(),
    )
    assert response.status_code == 200
    assert response.json()["output"] == {
        "general": {"description": "A scene"},
        "objects": [],
    }
    start.assert_awaited_once()
    assert start.call_args.args[0].config_path.name == "Fibo.yaml"
    stop.assert_awaited_once()
    assert not backend.llama_swap.lock.locked()


def test_invalid_response_is_502_and_stops_model(api, sdk):
    http, _, stop = api
    sdk[0].chat.completions.create.return_value.choices[0].message.content = "bad JSON"
    response = http.post(
        "/api/generate",
        data={"target_id": "Fibo", "mode": "image", "image_generator": "ideogram4"},
        files=upload(),
    )
    assert response.status_code == 502
    stop.assert_awaited_once()
    assert not backend.llama_swap.lock.locked()


@pytest.mark.parametrize(
    "target,mode,generator,description",
    [
        ("Fibo", "image", "krea2", ""),
        ("Fibo", "image", "", ""),
        ("Fibo", "image", "ideogram4", "change the background"),
        ("Fibo", "text", "", "A flower"),
        ("GemmaUnseen", "image", "ideogram4", ""),
        ("Captioning", "image", "ideogram4", "change the background"),
    ],
)
def test_invalid_combinations_do_not_launch(api, target, mode, generator, description):
    http, start, stop = api
    response = http.post(
        "/api/generate",
        data={
            "target_id": target,
            "mode": mode,
            "image_generator": generator,
            "description": description,
        },
        files=upload() if mode == "image" else None,
    )
    assert response.status_code == 422
    start.assert_not_awaited()
    stop.assert_not_awaited()


def test_fibo_not_allowed_in_folders(api):
    http, start, stop = api
    response = http.post(
        "/api/caption",
        data={"target_id": "Fibo", "caption_style": "ideogram4"},
        files=[("images", ("image.jpg", image_bytes(), "image/jpeg"))],
    )
    assert response.status_code == 422
    start.assert_not_awaited()
    stop.assert_not_awaited()


def test_model_capabilities(api):
    http, _, _ = api
    models = {entry["id"]: entry for entry in http.get("/api/targets").json()}
    assert models["Fibo"]["image_generators"] == ["ideogram4"]
    assert models["Fibo"]["text"] is False and models["Fibo"]["captioning"] is False
    assert models["Fibo"]["analysis_only"] is True
    assert "ideogram4" not in models["GemmaUnseen"]["image_generators"]
