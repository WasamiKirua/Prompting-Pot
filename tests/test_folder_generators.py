import base64
import json
from io import BytesIO
from zipfile import ZipFile

import pytest
import test_caption_inference as caption_fixtures

from prompting_pot.prompts import build_folder_caption_request

api = caption_fixtures.api
sdk = caption_fixtures.sdk
image = caption_fixtures.image


@pytest.mark.parametrize("generator", ["krea2", "zimage", "qwen-image-2.1", "anima"])
@pytest.mark.parametrize("model", ["Captioning", "GemmaUnseen"])
@pytest.mark.parametrize("style", ["character", "style"])
def test_prose_folder_targets(api, sdk, generator, model, style):
    http, _, _ = api
    client, _ = sdk
    if style == "style":
        client.chat.completions.create.return_value.choices[0].message.content = (
            "TOK style depicts a woman wearing a jacket."
        )
    response = http.post(
        "/api/caption",
        data={
            "target_id": model,
            "caption_generator": generator,
            "caption_style": style,
            "trigger_word": "TOK",
        },
        files=[image()],
    )
    assert response.status_code == 200
    assert response.headers["X-captioned"] == "1"
    request = client.chat.completions.create.call_args.kwargs
    assert (
        request["messages"][0]["content"]
        == build_folder_caption_request(generator, style, "TOK")[0]
    )
    assert "never comma-separated tags" in request["messages"][0]["content"]
    with ZipFile(BytesIO(response.content)) as archive:
        assert "image.txt" in archive.namelist()


def scene():
    return {
        "high_level_description": "A red boat.",
        "style_description": {
            "aesthetics": "calm",
            "lighting": "sunlight",
            "photo": "wide framing",
            "medium": "photograph",
        },
        "compositional_deconstruction": {
            "background": "A lake.",
            "elements": [{"type": "obj", "desc": "A red boat."}],
        },
    }


@pytest.mark.parametrize("stream", [False, True])
def test_ideogram_folder_json_skip_and_progress(api, sdk, stream):
    http, start, stop = api
    client, _ = sdk
    client.chat.completions.create.return_value.choices[0].message.content = json.dumps(
        scene()
    )
    response = http.post(
        "/api/caption",
        headers={"Accept": "application/x-ndjson"} if stream else {},
        data={
            "target_id": "Captioning",
            "caption_generator": "ideogram4",
            "caption_style": "ideogram4",
            "existing_captions": '["kept.JSON"]',
        },
        files=[image("kept.png"), image("new.png")],
    )
    assert response.status_code == 200
    if stream:
        events = [json.loads(line) for line in response.text.splitlines()]
        assert [
            event["processed"] for event in events if event["type"] == "progress"
        ] == [1, 2]
        data = base64.b64decode(events[-1]["archive"])
        assert events[-1]["counts"] == {"captioned": 1, "skipped": 1, "failed": 0}
    else:
        data = response.content
        assert response.headers["X-skipped"] == "1"
    with ZipFile(BytesIO(data)) as archive:
        assert set(archive.namelist()) == {"new.json", "caption-manifest.json"}
        assert json.loads(archive.read("new.json")) == scene()
    assert (
        client.chat.completions.create.call_args.kwargs["response_format"]["type"]
        == "json_schema"
    )
    start.assert_awaited_once()
    stop.assert_awaited_once()


@pytest.mark.parametrize(
    "model,trigger,style,name",
    [
        ("GemmaUnseen", "", "ideogram4", "image.png"),
        ("Fibo", "", "ideogram4", "image.png"),
        ("Captioning", "TOK", "ideogram4", "image.png"),
        ("Captioning", "", "character", "image.png"),
        ("Captioning", "", "ideogram4", "caption-manifest.png"),
    ],
)
def test_ideogram_constraints_before_startup(api, model, trigger, style, name):
    http, start, _ = api
    response = http.post(
        "/api/caption",
        data={
            "target_id": model,
            "caption_generator": "ideogram4",
            "caption_style": style,
            "trigger_word": trigger,
        },
        files=[image(name)],
    )
    assert response.status_code == 422
    start.assert_not_awaited()


def test_invalid_json_is_not_written(api, sdk):
    http, _, _ = api
    client, _ = sdk
    client.chat.completions.create.return_value.choices[0].message.content = "{}"
    response = http.post(
        "/api/caption",
        data={
            "target_id": "Captioning",
            "caption_generator": "ideogram4",
            "caption_style": "ideogram4",
        },
        files=[image()],
    )
    assert response.headers["X-failed"] == "1"
    with ZipFile(BytesIO(response.content)) as archive:
        assert archive.namelist() == ["caption-manifest.json"]


def test_all_existing_json_skips_startup(api):
    http, start, _ = api
    response = http.post(
        "/api/caption",
        data={
            "target_id": "Captioning",
            "caption_generator": "ideogram4",
            "caption_style": "ideogram4",
            "existing_captions": '["image.json"]',
        },
        files=[image()],
    )
    assert response.headers["X-skipped"] == "1"
    start.assert_not_awaited()
