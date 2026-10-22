import base64
import copy
import json
from io import BytesIO
from zipfile import ZipFile

import pytest
import test_caption_inference as caption_fixtures

from prompting_pot import text_generation
from prompting_pot.moss_caption import serialize_moss_caption
from prompting_pot.prompts import MOSS_CAPTION_PROMPT, MOSS_SYSTEM_PROMPT

api = caption_fixtures.api
sdk = caption_fixtures.sdk
image = caption_fixtures.image


def valid_caption():
    return {
        "caption": {
            "long": 'A sign reads "茶\nOPEN" above a blue table.',
            "medium": 'A sign reading "茶 OPEN" above a table.',
            "short": "A sign above a blue table.",
        },
        "watermark": [],
        "borders": ["none"],
    }


def test_moss_unicode_and_line_breaks_are_preserved():
    value = valid_caption()
    assert json.loads(serialize_moss_caption(json.dumps(value))) == value
    value["watermark"] = [{"box_2d": [0, 10, 30, 40], "label": "© artist"}]
    value["borders"] = ["left", "up"]
    assert json.loads(serialize_moss_caption(json.dumps(value))) == value


@pytest.mark.parametrize(
    "kind",
    [
        "empty",
        "missing",
        "unknown",
        "border-mix",
        "border-repeat",
        "border-invalid",
        "reversed-box",
        "out-of-range",
        "bool-box",
        "empty-label",
        "layout-coordinates",
        "malformed",
        "duplicate",
    ],
)
def test_invalid_moss_json_is_rejected(kind):
    value = copy.deepcopy(valid_caption())
    if kind == "empty":
        value["caption"]["long"] = "   "
    elif kind == "missing":
        del value["caption"]["short"]
    elif kind == "unknown":
        value["items"] = []
    elif kind == "border-mix":
        value["borders"] = ["none", "left"]
    elif kind == "border-repeat":
        value["borders"] = ["left", "left"]
    elif kind == "border-invalid":
        value["borders"] = ["top"]
    elif kind in {
        "reversed-box",
        "out-of-range",
        "bool-box",
        "empty-label",
        "layout-coordinates",
    }:
        box = {
            "reversed-box": [100, 0, 50, 10],
            "out-of-range": [0, 0, 1001, 1000],
            "bool-box": [False, 0, 30, 40],
            "empty-label": [0, 0, 30, 40],
            "layout-coordinates": [-512, -512, 512, 512],
        }[kind]
        value["watermark"] = [
            {"box_2d": box, "label": "   " if kind == "empty-label" else "mark"}
        ]
    raw = (
        "not JSON"
        if kind == "malformed"
        else '{"caption":{},"caption":{}}' if kind == "duplicate" else json.dumps(value)
    )
    with pytest.raises(ValueError):
        serialize_moss_caption(raw)


@pytest.mark.parametrize("model", ["Captioning", "GemmaUnseen"])
@pytest.mark.parametrize("mode", ["image", "text"])
def test_moss_generation_uses_prose_and_allows_changes(
    api, sdk, monkeypatch, model, mode
):
    http, client_start, _ = api
    client, factory = sdk
    monkeypatch.setattr(text_generation, "AsyncOpenAI", factory)
    data = {
        "target_id": model,
        "mode": mode,
        "image_generator": "moss",
        "description": "Make the background blue.",
    }
    response = http.post(
        "/api/generate",
        data=data,
        files=[("image", image()[1])] if mode == "image" else None,
    )
    assert response.status_code == 200
    request = client.chat.completions.create.call_args.kwargs
    assert request["messages"][0]["content"].startswith(MOSS_SYSTEM_PROMPT)
    assert "response_format" not in request
    assert request["model"] == model
    if mode == "image":
        assert request["messages"][1]["content"][0]["text"] == data["description"]
    client_start.assert_awaited_once()


@pytest.mark.parametrize("model", ["Captioning", "GemmaUnseen"])
@pytest.mark.parametrize("stream", [False, True])
def test_moss_folder_json_both_models_progress_and_skips(api, sdk, model, stream):
    http, start, stop = api
    client, _ = sdk
    client.chat.completions.create.return_value.choices[0].message.content = json.dumps(
        valid_caption()
    )
    response = http.post(
        "/api/caption",
        headers={"Accept": "application/x-ndjson"} if stream else {},
        data={
            "target_id": model,
            "caption_generator": "moss",
            "caption_style": "moss",
            "existing_captions": '["kept.json"]',
        },
        files=[image("kept.png"), image("new.png")],
    )
    assert response.status_code == 200
    if stream:
        events = [json.loads(line) for line in response.text.splitlines()]
        assert [
            event["processed"] for event in events if event["type"] == "progress"
        ] == [1, 2]
        raw = base64.b64decode(events[-1]["archive"])
    else:
        raw = response.content
    with ZipFile(BytesIO(raw)) as archive:
        assert set(archive.namelist()) == {"new.json", "caption-manifest.json"}
        assert json.loads(archive.read("new.json")) == valid_caption()
    request = client.chat.completions.create.call_args.kwargs
    assert request["model"] == model
    assert request["messages"][0]["content"] == MOSS_CAPTION_PROMPT
    assert request["response_format"]["json_schema"]["name"] == "moss_caption"
    assert request["max_tokens"] == (2000 if model == "GemmaUnseen" else 4096)
    assert request["extra_body"] == {"reasoning": "off", "chat_template_kwargs": {"enable_thinking": False}}
    start.assert_awaited_once()
    stop.assert_awaited_once()


@pytest.mark.parametrize("failure", ["invalid", "truncated"])
def test_moss_failed_outputs_do_not_create_sidecars(api, sdk, failure):
    http, _, _ = api
    client, _ = sdk
    choice = client.chat.completions.create.return_value.choices[0]
    choice.message.content = (
        "{}" if failure == "invalid" else json.dumps(valid_caption())
    )
    choice.finish_reason = "length" if failure == "truncated" else "stop"
    response = http.post(
        "/api/caption",
        data={
            "target_id": "Captioning",
            "caption_generator": "moss",
            "caption_style": "moss",
        },
        files=[image()],
    )
    assert response.headers["X-failed"] == "1"
    with ZipFile(BytesIO(response.content)) as archive:
        assert archive.namelist() == ["caption-manifest.json"]


@pytest.mark.parametrize(
    "model,style,trigger",
    [
        ("Fibo", "moss", ""),
        ("Captioning", "character", ""),
        ("GemmaUnseen", "moss", "TOK"),
    ],
)
def test_moss_incompatible_folder_inputs_rejected(api, model, style, trigger):
    http, start, _ = api
    response = http.post(
        "/api/caption",
        data={
            "target_id": model,
            "caption_generator": "moss",
            "caption_style": style,
            "trigger_word": trigger,
        },
        files=[image()],
    )
    assert response.status_code == 422
    start.assert_not_awaited()
