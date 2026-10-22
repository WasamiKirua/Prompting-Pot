import importlib
import json
from io import BytesIO
from zipfile import ZipFile

import pytest
from fastapi import HTTPException, UploadFile
from fastapi.testclient import TestClient
from PIL import Image

from prompting_pot.captioning import (
    MAX_CAPTION_IMAGE_BYTES,
    validate_batch,
    validate_image,
)
from prompting_pot.prompts import (
    FOLDER_CAPTION_STYLES,
    TARGET_PROMPTS,
    build_caption_request,
    build_prompt,
)

backend = importlib.import_module("prompting_pot.app")


@pytest.fixture
def client(tmp_path, monkeypatch):
    (tmp_path / "targets.json").write_text(
        json.dumps(
            [
                {
                    "id": "vision",
                    "name": "Vision",
                    "config": "vision.yaml",
                    "vision": True,
                },
                {"id": "text", "name": "Text", "config": "text.yaml", "vision": False},
            ]
        )
    )
    monkeypatch.setattr(backend, "CONFIG_DIR", tmp_path)
    monkeypatch.setattr(backend, "CAPTION_MODEL_ID", "vision")

    async def ready(target):
        return "http://localhost:8080/v1"

    monkeypatch.setattr(backend, "ensure_llama_swap", ready)
    return TestClient(backend.app)


def png():
    buffer = BytesIO()
    Image.new("RGB", (2, 2)).save(buffer, format="PNG")
    return buffer.getvalue()


def files(*names):
    return [("images", (name, png(), "image/png")) for name in names]


def test_caption_zip_and_skip(client, monkeypatch):
    calls = []

    async def start(target):
        calls.append("start")
        return "http://localhost:8080/v1"

    async def infer(**kwargs):
        assert kwargs["mode"] == "caption"
        assert kwargs["prompt"] == FOLDER_CAPTION_STYLES["character"]
        assert kwargs["description"] == build_caption_request("character", "TOK")[1]
        assert kwargs["image_content_type"] == "image/png"
        calls.append("infer")
        return "  TOK stands beside a tiny café.  "

    monkeypatch.setattr(backend, "ensure_llama_swap", start)
    monkeypatch.setattr(backend, "generate_prompt", infer)
    response = client.post(
        "/api/caption",
        data={
            "target_id": "vision",
            "trigger_word": "TOK",
            "existing_captions": '["kept.txt"]',
        },
        files=files("kept.png", "new.png", "other.png"),
    )
    assert response.status_code == 200
    assert response.headers["X-captioned"] == "2"
    assert response.headers["X-skipped"] == "1"
    assert calls == ["start", "infer", "infer"]
    with ZipFile(BytesIO(response.content)) as archive:
        assert set(archive.namelist()) == {
            "new.txt",
            "other.txt",
            "caption-manifest.json",
        }
        assert (
            archive.read("new.txt").decode("utf-8")
            == "TOK stands beside a tiny café.\n"
        )
        manifest = json.loads(archive.read("caption-manifest.json"))
        assert manifest["counts"] == {"captioned": 2, "skipped": 1, "failed": 0}


def test_all_skipped_without_starting_model(client):
    response = client.post(
        "/api/caption",
        data={
            "target_id": "vision",
            "trigger_word": "TOK",
            "existing_captions": '["IMAGE.TXT"]',
        },
        files=files("image.png"),
    )
    assert response.status_code == 200
    assert response.headers["X-skipped"] == "1"


def test_caption_inference_failure_is_reported(client, monkeypatch):
    async def infer(**kwargs):
        raise RuntimeError("Caption model unavailable")

    monkeypatch.setattr(backend, "generate_caption", infer)
    response = client.post(
        "/api/caption",
        data={"target_id": "vision", "trigger_word": "TOK"},
        files=files("image.png"),
    )
    assert response.status_code == 200
    assert response.headers["X-failed"] == "1"
    with ZipFile(BytesIO(response.content)) as archive:
        manifest = json.loads(archive.read("caption-manifest.json"))
        assert manifest["files"][0]["error"] == "Caption model unavailable"


@pytest.mark.parametrize(
    "names",
    [
        ("../image.png",),
        ("folder/image.png",),
        ("bad.gif",),
        ("photo.jpg", "photo.png"),
        ("PHOTO.png", "photo.png"),
    ],
)
def test_unsafe_or_ambiguous_names(client, names):
    response = client.post(
        "/api/caption",
        data={"target_id": "vision", "trigger_word": "TOK"},
        files=files(*names),
    )
    assert response.status_code == 422


@pytest.mark.parametrize("existing", ["not json", "{}", "[1]", '["../image.txt"]'])
def test_invalid_existing_names(client, existing):
    response = client.post(
        "/api/caption",
        data={
            "target_id": "vision",
            "trigger_word": "TOK",
            "existing_captions": existing,
        },
        files=files("image.png"),
    )
    assert response.status_code == 422


@pytest.mark.parametrize("target", ["missing", "text"])
def test_caption_requires_vision(client, target):
    response = client.post(
        "/api/caption", data={"target_id": target}, files=files("image.png")
    )
    assert response.status_code == 422


def test_invalid_image_and_json_output_reported_per_file(client, monkeypatch):
    async def start(target):
        return "http://localhost:8080/v1"

    async def infer(**kwargs):
        return {"caption": "not plain text"}

    monkeypatch.setattr(backend, "ensure_llama_swap", start)
    monkeypatch.setattr(backend, "generate_prompt", infer)
    response = client.post(
        "/api/caption",
        data={"target_id": "vision", "trigger_word": "TOK"},
        files=[("images", ("bad.png", b"bad", "image/png")), *files("valid.png")],
    )
    assert response.status_code == 200
    assert response.headers["X-failed"] == "2"
    with ZipFile(BytesIO(response.content)) as archive:
        assert archive.namelist() == ["caption-manifest.json"]


@pytest.mark.parametrize("extra", [0, 1])
def test_folder_image_25mb_boundary(extra):
    content = png()
    content += b"\0" * (MAX_CAPTION_IMAGE_BYTES + extra - len(content))
    if extra:
        with pytest.raises(HTTPException, match="25 MB"):
            validate_image(content, "image/png", caption=True)
    else:
        validate_image(content, "image/png", caption=True)
        with pytest.raises(HTTPException, match="10 MB"):
            validate_image(content, "image/png")


@pytest.mark.parametrize("count", [4, 5])
def test_folder_batch_total_stays_100mb(count):
    images = [
        UploadFile(file=BytesIO(), filename=f"{i}.png", size=MAX_CAPTION_IMAGE_BYTES)
        for i in range(count)
    ]
    if count == 4:
        validate_batch(images, [])
    else:
        with pytest.raises(HTTPException, match="100 MB"):
            validate_batch(images, [])


def test_batch_accepts_250_images(client):
    names = [f"{i}.png" for i in range(250)]
    response = client.post(
        "/api/caption",
        data={
            "target_id": "vision",
            "trigger_word": "TOK",
            "existing_captions": json.dumps([f"{i}.txt" for i in range(250)]),
        },
        files=files(*names),
    )
    assert response.status_code == 200
    assert response.headers["X-skipped"] == "250"


def test_batch_image_count_limit(client):
    response = client.post(
        "/api/caption",
        data={"target_id": "vision", "trigger_word": "TOK"},
        files=files(*(f"{i}.png" for i in range(251))),
    )
    assert response.status_code == 422


def test_prompt_overrides_preserve_json_braces(monkeypatch):
    monkeypatch.setitem(
        TARGET_PROMPTS, "json-model", {"text": '{"prompt": "{description}"}'}
    )
    assert build_prompt("json-model", "text", "a flower") == '{"prompt": "a flower"}'
    assert build_prompt("other", "text", "a flower").endswith("a flower")
