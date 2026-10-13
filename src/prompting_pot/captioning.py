"""Sequential caption batches for browser-selected, top-level images."""

import base64
import json
from collections.abc import Awaitable, Callable
from io import BytesIO
from pathlib import PurePosixPath
from typing import Any
from zipfile import ZIP_DEFLATED, ZipFile

from fastapi import HTTPException, UploadFile
from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    DefaultAsyncHttpxClient,
)
from PIL import Image, UnidentifiedImageError

from .ideogram_caption import IDEOGRAM_CAPTION_SCHEMA, serialize_ideogram_caption

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_CAPTION_IMAGE_BYTES = 25 * 1024 * 1024
MAX_BATCH_BYTES = 100 * 1024 * 1024
MAX_BATCH_IMAGES = 150
IMAGE_TYPES = {
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".webp": "image/webp",
}


CAPTION_IMAGE_TYPES = {**IMAGE_TYPES, ".bmp": "image/bmp"}


async def generate_caption(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    instruction: str,
    image: bytes,
    content_type: str,
    caption_style: str = "character",
) -> str:
    """Use server defaults for text; constrain and validate Ideogram JSON output."""
    options: dict[str, Any] = {}
    if caption_style == "ideogram4":
        options = {
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "ideogram4_caption",
                    "schema": IDEOGRAM_CAPTION_SCHEMA,
                },
            },
            "max_tokens": 4096,
            "extra_body": {"reasoning": "off"},
        }
    encoded = base64.b64encode(image).decode("ascii")
    try:
        async with AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key="llama",
            timeout=300.0,
            max_retries=0,
            http_client=DefaultAsyncHttpxClient(trust_env=False),
        ) as client:
            result = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{content_type};base64,{encoded}"
                                },
                            },
                            {"type": "text", "text": instruction},
                        ],
                    },
                ],
                **options,
            )
    except APITimeoutError as exc:
        raise RuntimeError("Caption inference timed out after 300 seconds.") from exc
    except APIConnectionError as exc:
        raise RuntimeError(f"Could not connect to llama-swap at {base_url}.") from exc
    except APIStatusError as exc:
        raise RuntimeError(
            f"llama-swap returned HTTP {exc.status_code} while captioning."
        ) from exc
    try:
        content = result.choices[0].message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError(
            "llama-swap returned an unexpected caption response."
        ) from exc
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The caption model returned no caption.")
    if caption_style == "ideogram4":
        if result.choices[0].finish_reason == "length":
            raise RuntimeError(
                "Ideogram JSON caption was truncated; increase the output/context budget."
            )
        return serialize_ideogram_caption(content)
    return content.strip()


def validate_image(content: bytes, content_type: str, *, caption: bool = False) -> None:
    """Validate size, file type and decodability before passing bytes to inference."""
    allowed = CAPTION_IMAGE_TYPES if caption else IMAGE_TYPES
    if content_type not in allowed.values():
        message = (
            "Use a PNG, JPEG, WebP, or BMP image."
            if caption
            else "Use a PNG, JPEG, or WebP image."
        )
        raise HTTPException(422, message)
    maximum = MAX_CAPTION_IMAGE_BYTES if caption else MAX_IMAGE_BYTES
    if not content or len(content) > maximum:
        raise HTTPException(
            422,
            f"Image must be nonempty and no larger than {maximum // (1024 * 1024)} MB.",
        )
    try:
        with Image.open(BytesIO(content)) as uploaded:
            if Image.MIME.get(uploaded.format or "") != content_type:
                raise HTTPException(422, "Image content does not match its file type.")
            uploaded.verify()
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        raise HTTPException(
            422, "The image file is invalid or too large to decode."
        ) from exc


def caption_filename(filename: str, suffix: str = ".txt") -> str:
    """Accept flat filenames only; never trust uploaded paths for ZIP entries."""
    if not filename or filename in {".", ".."} or any(c in filename for c in "/\\\x00"):
        raise HTTPException(422, "Upload top-level filenames, not folder paths.")
    path = PurePosixPath(filename)
    if path.suffix.lower() not in CAPTION_IMAGE_TYPES:
        raise HTTPException(
            422, "Captioning accepts PNG, JPEG, WebP, and BMP images only."
        )
    name = path.with_suffix(suffix).name
    if name.casefold() == "caption-manifest.json":
        raise HTTPException(
            422,
            "Rename caption-manifest images to avoid colliding with the ZIP manifest.",
        )
    return name


def validate_batch(
    images: list[UploadFile], existing: list[str], suffix: str = ".txt"
) -> None:
    """Reject ambiguous caption names and bound batch size before loading a model."""
    if not 1 <= len(images) <= MAX_BATCH_IMAGES:
        raise HTTPException(422, "Select between 1 and 150 top-level images.")
    if any(
        PurePosixPath(name).suffix.lower() != suffix
        or any(c in name for c in "/\\\x00")
        for name in existing
    ):
        raise HTTPException(422, f"Existing captions must be flat {suffix} filenames.")
    names = [caption_filename(image.filename or "", suffix) for image in images]
    if len({name.casefold() for name in names}) != len(names):
        raise HTTPException(
            422,
            "Images share a caption filename (for example photo.jpg and photo.png). Rename them first.",
        )
    if any((image.size or 0) > MAX_CAPTION_IMAGE_BYTES for image in images):
        raise HTTPException(422, "Each caption image must be no larger than 25 MB.")
    if sum(image.size or 0 for image in images) > MAX_BATCH_BYTES:
        raise HTTPException(422, "A batch must be no larger than 100 MB.")


async def caption_batch(
    images: list[UploadFile],
    existing: list[str],
    infer: Callable[[bytes, str], Awaitable[str]],
    *,
    suffix: str = ".txt",
    on_progress: Callable[[dict[str, Any]], Awaitable[None]] | None = None,
) -> tuple[bytes, dict[str, int]]:
    """Produce UTF-8 sidecars and a per-file manifest in a downloadable ZIP.

    Existing caption names are skipped, case-insensitively. Invalid images and
    inference failures are reported per file; integration stubs abort the batch.
    Upload streams are closed even when inference fails.
    """
    validate_batch(images, existing, suffix)
    skipped = {name.casefold() for name in existing}
    counts = {"captioned": 0, "skipped": 0, "failed": 0}
    results: list[dict[str, str]] = []
    archive = BytesIO()
    total = 0

    async def report(filename: str) -> None:
        if on_progress is not None:
            await on_progress(
                {
                    "type": "progress",
                    "processed": sum(counts.values()),
                    "total": len(images),
                    "filename": filename,
                    "counts": counts.copy(),
                }
            )

    try:
        with ZipFile(archive, "w", ZIP_DEFLATED) as zip_file:
            for image in images:
                filename = image.filename or ""
                sidecar = caption_filename(filename, suffix)
                if sidecar.casefold() in skipped:
                    counts["skipped"] += 1
                    results.append({"image": filename, "status": "skipped"})
                    await report(filename)
                    continue
                try:
                    content = await image.read(MAX_CAPTION_IMAGE_BYTES + 1)
                    total += len(content)
                    if total > MAX_BATCH_BYTES:
                        raise HTTPException(
                            413, "A batch must be no larger than 100 MB."
                        )
                    content_type = CAPTION_IMAGE_TYPES[
                        PurePosixPath(filename).suffix.lower()
                    ]
                    validate_image(content, content_type, caption=True)
                    caption = await infer(content, content_type)
                    if not isinstance(caption, str) or not caption.strip():
                        raise ValueError(
                            "The caption model must return nonempty plain text."
                        )
                    if suffix == ".json":
                        caption = serialize_ideogram_caption(caption)
                    zip_file.writestr(sidecar, caption.strip() + "\n")
                    counts["captioned"] += 1
                    results.append(
                        {"image": filename, "caption": sidecar, "status": "captioned"}
                    )
                except NotImplementedError:
                    raise
                except HTTPException as exc:
                    if exc.status_code == 413:
                        raise
                    counts["failed"] += 1
                    results.append(
                        {
                            "image": filename,
                            "status": "failed",
                            "error": str(exc.detail),
                        }
                    )
                except (ValueError, RuntimeError, OSError) as exc:
                    counts["failed"] += 1
                    results.append(
                        {"image": filename, "status": "failed", "error": str(exc)}
                    )
                await report(filename)
            zip_file.writestr(
                "caption-manifest.json",
                json.dumps({"counts": counts, "files": results}, indent=2),
            )
    finally:
        for image in images:
            await image.close()
    return archive.getvalue(), counts
