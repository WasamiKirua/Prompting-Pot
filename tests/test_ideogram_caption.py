import asyncio
import importlib
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from threading import Thread
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from prompting_pot import captioning
from prompting_pot.ideogram_caption import (
    IDEOGRAM_CAPTION_SCHEMA,
    serialize_ideogram_caption,
)
from prompting_pot.prompts import (
    IDEOGRAM4_CAPTION_PROMPT,
    QWEN_IMAGE_2_1_CAPTION_PROMPT,
    build_caption_request,
)

backend = importlib.import_module("prompting_pot.app")


def caption():
    return {
        "high_level_description": "A café sign above a red table.",
        "style_description": {
            "aesthetics": "warm",
            "lighting": "soft daylight",
            "photo": "wide framing",
            "medium": "photograph",
            "color_palette": ["#AA1122"],
        },
        "compositional_deconstruction": {
            "background": "A cream wall.",
            "elements": [
                {"type": "obj", "bbox": [100, 200, 600, 800], "desc": "A red table."},
                {
                    "type": "text",
                    "text": "CAFÉ",
                    "desc": "Dark sans-serif letters above the table.",
                },
            ],
        },
    }


def test_canonical_photo_caption_and_utf8():
    value = caption()
    style = value["style_description"]
    value["style_description"] = dict(reversed(list(style.items())))
    raw = serialize_ideogram_caption(json.dumps(value))
    assert "café" in raw and "\\u00" not in raw
    assert ": " not in raw
    parsed = json.loads(raw)
    assert list(parsed) == [
        "high_level_description",
        "style_description",
        "compositional_deconstruction",
    ]
    assert list(parsed["style_description"]) == [
        "aesthetics",
        "lighting",
        "photo",
        "medium",
        "color_palette",
    ]
    assert parsed == caption()


def test_photographic_medium_cannot_use_art_style():
    value = caption()
    value["style_description"].pop("photo")
    value["style_description"]["art_style"] = "illustrated"
    with pytest.raises(ValueError, match="Photographs must use photo"):
        serialize_ideogram_caption(json.dumps(value))


def test_canonical_non_photo_and_optional_fields():
    value = caption()
    value["style_description"] = {
        "art_style": "flat illustration",
        "medium": "illustration",
        "lighting": "flat tones",
        "aesthetics": "minimal",
    }
    value["compositional_deconstruction"]["elements"][0].pop("bbox")
    value["compositional_deconstruction"]["elements"][1]["color_palette"] = ["#FFFFFF"]
    value["compositional_deconstruction"]["elements"][1] = dict(
        reversed(list(value["compositional_deconstruction"]["elements"][1].items()))
    )
    parsed = json.loads(serialize_ideogram_caption(json.dumps(value)))
    assert list(parsed["style_description"]) == [
        "aesthetics",
        "lighting",
        "medium",
        "art_style",
    ]
    assert list(parsed["compositional_deconstruction"]["elements"][1]) == [
        "type",
        "text",
        "desc",
        "color_palette",
    ]


@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.pop("compositional_deconstruction"),
        lambda c: c.update(extra="unknown"),
        lambda c: c["style_description"].update(art_style="both"),
        lambda c: c["style_description"].update(color_palette=["#aa1122"]),
        lambda c: c["style_description"].update(color_palette=["#AA1122"] * 17),
        lambda c: c["compositional_deconstruction"]["elements"][0].update(
            bbox=[900, 100, 100, 900]
        ),
        lambda c: c["compositional_deconstruction"]["elements"][0].update(
            bbox=[0, 900, 1000, 100]
        ),
        lambda c: c["compositional_deconstruction"]["elements"][0].update(
            bbox=[0, 0, 1001, 1000]
        ),
        lambda c: c["compositional_deconstruction"]["elements"][0].update(
            bbox=[True, 0, 1000, 1000]
        ),
        lambda c: c["compositional_deconstruction"]["elements"][0].update(
            bbox=[0, 0, 1000]
        ),
        lambda c: c["compositional_deconstruction"]["elements"][1].pop("text"),
        lambda c: c["compositional_deconstruction"]["elements"][0].update(
            text="not an obj key"
        ),
    ],
)
def test_invalid_schema_or_geometry_is_rejected(change):
    value = caption()
    change(value)
    with pytest.raises(ValueError):
        serialize_ideogram_caption(json.dumps(value))


@pytest.mark.parametrize(
    "raw",
    [
        "not JSON",
        "```json\n{}\n```",
        "[]",
        '{"high_level_description":"one","high_level_description":"two"}',
    ],
)
def test_invalid_or_ambiguous_json_is_rejected(raw):
    with pytest.raises(ValueError):
        serialize_ideogram_caption(raw)


@pytest.fixture
def sdk(monkeypatch):
    client = Mock()
    choice = SimpleNamespace(
        message=SimpleNamespace(content=json.dumps(caption())), finish_reason="stop"
    )
    client.chat.completions.create = AsyncMock(
        return_value=SimpleNamespace(choices=[choice])
    )
    factory = Mock()
    context = Mock()
    context.__aenter__ = AsyncMock(return_value=client)

    async def close(*args):
        await factory.call_args.kwargs["http_client"].aclose()

    context.__aexit__ = AsyncMock(side_effect=close)
    factory.return_value = context
    monkeypatch.setattr(captioning, "AsyncOpenAI", factory)
    return client


def test_real_sdk_serializes_schema_to_local_fake_api():
    received = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            received.append(
                json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            )
            body = json.dumps(
                {
                    "id": "test",
                    "object": "chat.completion",
                    "created": 0,
                    "model": "Captioning",
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": json.dumps(caption()),
                            },
                            "finish_reason": "stop",
                        }
                    ],
                }
            ).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    with HTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            output = asyncio.run(
                captioning.generate_caption(
                    base_url=f"http://127.0.0.1:{server.server_port}/v1",
                    model="Captioning",
                    system_prompt=IDEOGRAM4_CAPTION_PROMPT,
                    instruction="caption",
                    image=b"image",
                    content_type="image/png",
                    caption_style="ideogram4",
                )
            )
        finally:
            server.shutdown()
            thread.join(timeout=5)
    assert json.loads(output) == caption()
    assert (
        received[0]["response_format"]["json_schema"]["schema"]
        == IDEOGRAM_CAPTION_SCHEMA
    )
    assert received[0]["max_tokens"] == 4096
    assert received[0]["reasoning"] == "off"


def test_ideogram_schema_request_and_budget(sdk):
    prompt, instruction = build_caption_request("ideogram4")
    output = asyncio.run(
        captioning.generate_caption(
            base_url="http://local:8080/v1",
            model="Captioning",
            system_prompt=prompt,
            instruction=instruction,
            image=b"image",
            content_type="image/png",
            caption_style="ideogram4",
        )
    )
    assert json.loads(output) == caption()
    args = sdk.chat.completions.create.call_args.kwargs
    assert args["response_format"]["type"] == "json_schema"
    assert args["response_format"]["json_schema"]["schema"] == IDEOGRAM_CAPTION_SCHEMA
    assert args["max_tokens"] == 4096
    assert args["extra_body"] == {"reasoning": "off"}
    assert args["messages"][0]["content"] == IDEOGRAM4_CAPTION_PROMPT


def test_truncated_json_is_rejected_even_if_parseable(sdk):
    sdk.chat.completions.create.return_value.choices[0].finish_reason = "length"
    with pytest.raises(RuntimeError, match="truncated"):
        asyncio.run(
            captioning.generate_caption(
                base_url="http://local:8080/v1",
                model="Captioning",
                system_prompt=IDEOGRAM4_CAPTION_PROMPT,
                instruction="caption",
                image=b"image",
                content_type="image/png",
                caption_style="ideogram4",
            )
        )


@pytest.fixture
def api(tmp_path, monkeypatch):
    (tmp_path / "targets.json").write_text(
        json.dumps(
            [
                {
                    "id": "Captioning",
                    "name": "Captioning",
                    "config": "caption.yaml",
                    "vision": True,
                },
            ]
        )
    )
    monkeypatch.setattr(backend, "CONFIG_DIR", tmp_path)
    start = AsyncMock(return_value="http://local:8080/v1")
    stop = AsyncMock()
    monkeypatch.setattr(backend, "ensure_llama_swap", start)
    monkeypatch.setattr(backend.llama_swap, "stop", stop)
    return TestClient(backend.app), start, stop


def files(*names):
    buffer = BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, "PNG")
    return [("images", (name, buffer.getvalue(), "image/png")) for name in names]


def test_single_image_qwen_ideogram_reuses_validated_caption_path(api, sdk):
    http, start, stop = api
    response = http.post(
        "/api/generate",
        data={
            "target_id": "Captioning",
            "mode": "image",
            "image_generator": "ideogram4",
        },
        files=[("image", files("scene.png")[0][1])],
    )
    assert response.status_code == 200
    assert json.loads(response.json()["output"]) == caption()
    request = sdk.chat.completions.create.call_args.kwargs
    assert request["model"] == "Captioning"
    assert request["messages"][0]["content"] == IDEOGRAM4_CAPTION_PROMPT
    assert request["response_format"]["type"] == "json_schema"
    assert request["max_tokens"] == 4096
    start.assert_awaited_once()
    stop.assert_awaited_once()
    assert not backend.llama_swap.lock.locked()


def test_single_image_invalid_qwen_json_is_502_and_stops(api, sdk):
    http, _, stop = api
    sdk.chat.completions.create.return_value.choices[0].message.content = "{}"
    response = http.post(
        "/api/generate",
        data={
            "target_id": "Captioning",
            "mode": "image",
            "image_generator": "ideogram4",
        },
        files=[("image", files("scene.png")[0][1])],
    )
    assert response.status_code == 502
    stop.assert_awaited_once()
    assert not backend.llama_swap.lock.locked()


def test_single_image_qwen_rejects_changes_before_start(api):
    http, start, stop = api
    response = http.post(
        "/api/generate",
        data={
            "target_id": "Captioning",
            "mode": "image",
            "image_generator": "ideogram4",
            "description": "make it blue",
        },
        files=[("image", files("scene.png")[0][1])],
    )
    assert response.status_code == 422
    start.assert_not_awaited()
    stop.assert_not_awaited()


def test_single_image_qwen_text_uses_target_prompt_without_json_settings(api, sdk):
    http, start, stop = api
    text = "A café with warm window light and a red wooden table."
    sdk.chat.completions.create.return_value.choices[0].message.content = text
    response = http.post(
        "/api/generate",
        data={
            "target_id": "Captioning",
            "mode": "image",
            "image_generator": "qwen-image-2.1",
        },
        files=[("image", files("image.png")[0][1])],
    )
    assert response.status_code == 200
    assert response.json()["output"] == text
    args = sdk.chat.completions.create.call_args.kwargs
    assert set(args) == {"model", "messages"}
    assert args["messages"][0]["content"] == QWEN_IMAGE_2_1_CAPTION_PROMPT
    start.assert_awaited_once()
    stop.assert_awaited_once()


def test_single_image_qwen_text_rejects_changes(api):
    http, start, _ = api
    response = http.post(
        "/api/generate",
        data={
            "target_id": "Captioning",
            "mode": "image",
            "image_generator": "qwen-image-2.1",
            "description": "change colors",
        },
        files=[("image", files("image.png")[0][1])],
    )
    assert response.status_code == 422
    start.assert_not_awaited()
