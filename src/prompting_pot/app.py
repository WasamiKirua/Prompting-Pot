"""HTTP endpoints, local model lifecycle, and image-to-prompt inference."""
import json
import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles

from .captioning import (
    MAX_IMAGE_BYTES, caption_batch, caption_filename, generate_caption,
    validate_batch, validate_image,
)
from .prompts import (
    FOLDER_CAPTION_STYLES, FIBO_CAPTION_PROMPT, IMAGE_GENERATORS,
    build_caption_request, build_image_prompt, build_prompt,
)
from .fibo import analyze_image
from .gemma_unseen import generate_from_image
from .llama_swap import LlamaSwap

ROOT = Path(os.environ.get("PROMPTING_POT_ROOT", Path(__file__).resolve().parents[2]))
CONFIG_DIR = ROOT / "config"
STATIC_DIR = Path(__file__).parent / "static"
CAPTION_MODEL_ID = "Captioning"


@dataclass(frozen=True)
class Target:
    """A local prompt-writing model, identified by its llama-swap API name."""

    id: str
    name: str
    config_path: Path
    vision: bool


def load_targets() -> list[Target]:
    entries = json.loads((CONFIG_DIR / "targets.json").read_text())
    targets = []
    seen = set()
    for entry in entries:
        path = (CONFIG_DIR / entry["config"]).resolve()
        if not path.is_relative_to(CONFIG_DIR.resolve()) or path.suffix not in {".yaml", ".yml"}:
            raise ValueError("Configurations must be YAML files inside config/.")
        if entry["id"] in seen or not isinstance(entry["vision"], bool):
            raise ValueError("Target IDs must be unique and vision must be boolean.")
        seen.add(entry["id"])
        targets.append(Target(entry["id"], entry["name"], path, entry["vision"]))
    return targets


llama_swap = LlamaSwap()


async def ensure_llama_swap(target: Target) -> str:
    """Start/reuse llama-swap and await readiness inside an inference session."""
    return await llama_swap.start(target.config_path)


@asynccontextmanager
async def inference_session(target: Target) -> AsyncIterator[str]:
    """Keep the lifecycle lock until the request or entire batch is finished."""
    async with llama_swap.lock:
        try:
            try:
                base_url = await ensure_llama_swap(target)
            except (RuntimeError, OSError) as exc:
                raise HTTPException(503, str(exc)) from exc
            yield base_url
        finally:
            await llama_swap.stop()


@asynccontextmanager
async def lifespan(application: FastAPI) -> AsyncIterator[None]:
    """Clean up any remaining application-owned process on server shutdown."""
    try:
        yield
    finally:
        async with llama_swap.lock:
            await llama_swap.stop()


def supported_image_generators(target: Target) -> list[str]:
    """Expose compatible single-image generators for each local vision model."""
    if target.id == CAPTION_MODEL_ID:
        return ["ideogram4", "qwen-image-2.1"]
    return ["ideogram4"] if target.id == "Fibo" else [key for key in IMAGE_GENERATORS if key != "ideogram4"]


async def generate_prompt(
    *, base_url: str, target: Target, mode: str, description: str, prompt: str,
    image: bytes | None, image_content_type: str | None,
    image_generator: str | None = None,
    caption_style: str = "character",
) -> str | dict[str, Any] | list[Any]:
    """Call the selected local vision model for image prompts or captions."""
    if target.id == "Fibo":
        if mode != "image" or image_generator != "ideogram4" or image is None or description.strip():
            raise ValueError("Fibo requires an as-is image analysis for Ideogram 4.")
        return await analyze_image(base_url=base_url, model=target.id, image=image)
    if mode == "image" and target.id == CAPTION_MODEL_ID and image_generator in {"ideogram4", "qwen-image-2.1"}:
        caption_style = image_generator
        prompt, description = build_caption_request(caption_style)
        mode = "caption"
    if mode == "caption":
        if image is None or image_content_type is None:
            raise ValueError("Captioning requires an image.")
        if target.id == "GemmaUnseen":
            return await generate_from_image(
                base_url=base_url, model=target.id, system_prompt=prompt,
                description=description, image=image, content_type=image_content_type,
                image_generator="krea2",
            )
        return await generate_caption(
            base_url=base_url, model=target.id, system_prompt=prompt,
            instruction=description, image=image, content_type=image_content_type,
            caption_style=caption_style,
        )
    if mode != "image":
        raise NotImplementedError(f"Local {mode} inference is not connected yet.")
    if image is None or image_content_type is None or image_generator is None or image_generator not in IMAGE_GENERATORS:
        raise ValueError("Image mode requires an image and a supported image generator.")
    return await generate_from_image(
        base_url=base_url, model=target.id, system_prompt=prompt,
        description=description, image=image, content_type=image_content_type,
        image_generator=image_generator,
    )


app = FastAPI(title="Prompting Pot", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index():
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/api/targets")
async def get_targets():
    return [{"id": t.id, "name": t.name, "vision": t.vision,
             "captioning": t.id in {CAPTION_MODEL_ID, "GemmaUnseen"} and t.vision,
             "image_generators": supported_image_generators(t),
             "analysis_only": t.id in {"Fibo", CAPTION_MODEL_ID}, "text": t.id != "Fibo"} for t in load_targets()]


@app.get("/api/image-generators")
async def get_image_generators():
    return [{"id": key, "name": name} for key, name in IMAGE_GENERATORS.items()]


@app.post("/api/generate")
async def generate(
    target_id: str = Form(...), mode: str = Form(...),
    description: str = Form("", max_length=20000), image: UploadFile | None = File(None),
    image_generator: str = Form(""),
):
    target = next((t for t in load_targets() if t.id == target_id), None)
    if target is None:
        raise HTTPException(422, "Select a registered target model.")
    if mode not in {"image", "text"}:
        raise HTTPException(422, "Mode must be image or text.")
    description = description.strip()
    if target.id == "Fibo" and (mode != "image" or description):
        raise HTTPException(422, "Fibo supports single-image as-is analysis only, without change instructions.")
    if target.id == CAPTION_MODEL_ID and mode == "image" and description:
        raise HTTPException(422, "Qwen caption analysis describes the image as-is, without change instructions.")
    content = None
    content_type = None
    if mode == "image":
        if image_generator not in IMAGE_GENERATORS:
            raise HTTPException(422, "Select a supported target image generator.")
        if image_generator not in supported_image_generators(target):
            raise HTTPException(422, "Select a compatible model for the chosen image generator.")
        if not target.vision:
            raise HTTPException(422, "This target does not support image input.")
        if image is None:
            raise HTTPException(422, "An image is required.")
        content_type = image.content_type
        content = await image.read(MAX_IMAGE_BYTES + 1)
        await image.close()
        validate_image(content, content_type or "")
    elif not description:
        raise HTTPException(422, "Describe the image you want to generate.")
    elif image is not None:
        raise HTTPException(422, "Text mode does not accept an image.")
    try:
        async with inference_session(target) as base_url:
            output = await generate_prompt(base_url=base_url, target=target, mode=mode,
                                           description=description,
                                           prompt=(FIBO_CAPTION_PROMPT if target.id == "Fibo" else
                                                   build_caption_request(image_generator)[0] if mode == "image" and target.id == CAPTION_MODEL_ID else
                                                   build_image_prompt(image_generator) if mode == "image"
                                                   else build_prompt(target.id, mode, description)), image=content,
                                           image_content_type=content_type,
                                           image_generator=image_generator if mode == "image" else None)
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(502, str(exc)) from exc
    return {"output": output}


@app.post("/api/caption")
async def caption_folder(
    target_id: str = Form(...), images: list[UploadFile] = File(...),
    existing_captions: str = Form("[]", max_length=100000),
    caption_style: str = Form("character"), trigger_word: str = Form("", max_length=200),
) -> Response:
    """Caption uploaded images; return a ZIP, never write to a client-supplied path."""
    target = next((t for t in load_targets() if t.id == target_id), None)
    if target is None or target.id not in {CAPTION_MODEL_ID, "GemmaUnseen"} or not target.vision:
        raise HTTPException(422, "Captioning requires registered Qwen3.5 Captioning (Captioning) or GemmaUnseen with vision support.")
    if caption_style not in FOLDER_CAPTION_STYLES:
        raise HTTPException(422, "Krea2 folder captioning supports Character or Style only.")
    if not trigger_word.strip() or '\n' in trigger_word or '\r' in trigger_word:
        raise HTTPException(422, "Krea2 captions require a nonempty, single-line trigger word.")
    trigger_word = trigger_word.strip()
    caption_prompt, instruction = build_caption_request(caption_style, trigger_word)
    try:
        existing = json.loads(existing_captions)
    except json.JSONDecodeError as exc:
        raise HTTPException(422, "Existing captions must be a JSON list of filenames.") from exc
    if not isinstance(existing, list) or not all(isinstance(name, str) for name in existing):
        raise HTTPException(422, "Existing captions must be a JSON list of filenames.")
    validate_batch(images, existing)
    try:
        # No model startup is needed when every image already has a caption.
        skipped = {name.casefold() for name in existing}
        pending = any(caption_filename(image.filename or '').casefold() not in skipped for image in images)

        async def infer(content: bytes, content_type: str) -> str:
            output = await generate_prompt(
                base_url=base_url, target=target, mode="caption", description=instruction,
                prompt=caption_prompt, image=content,
                image_content_type=content_type, caption_style=caption_style,
            )
            if not isinstance(output, str):
                raise ValueError("The caption model must return plain prose.")
            output = output.strip()
            if not output.startswith(trigger_word + ' '):
                raise ValueError("The caption must begin with the exact trigger word followed by a space and prose.")
            return output

        if pending:
            async with inference_session(target) as base_url:
                archive, counts = await caption_batch(images, existing, infer)
        else:
            # caption_batch skips every image without invoking infer().
            archive, counts = await caption_batch(images, existing, infer)
    except NotImplementedError as exc:
        raise HTTPException(501, str(exc)) from exc
    finally:
        for image in images:
            await image.close()
    return Response(
        archive, media_type="application/zip",
        headers={"Content-Disposition": 'attachment; filename="captions.zip"',
                 **{f"X-{key}": str(value) for key, value in counts.items()}},
    )
