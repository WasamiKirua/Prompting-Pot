"""Single-image Fibo analysis using the supplied llama.cpp request settings."""

import base64
import json
from io import BytesIO
from typing import Any, NoReturn

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    DefaultAsyncHttpxClient,
)
from PIL import Image

from .fibo_schema import SceneDescription
from .prompts import FIBO_CAPTION_PROMPT


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_constant(value: str) -> NoReturn:
    raise ValueError(f"Nonfinite JSON number: {value}")


def parse_scene(text: str) -> dict[str, Any]:
    """Validate native JSON and convert trained coordinate/color tokens to numbers."""
    text = text.strip()
    for token in ("<|im_end|>", "<|endoftext|>"):
        text = text.removesuffix(token).strip()
    text = text.split("</think>")[-1].strip()
    try:
        raw = json.loads(
            text, object_pairs_hook=_unique_object, parse_constant=_reject_constant
        )
        scene = SceneDescription.model_validate(raw)
        scene.post_process()  # type: ignore[no-untyped-call]
        result = scene.model_dump(exclude_none=True)
        # Reject nonfinite numbers, including malformed coordinate tokens.
        json.dumps(result, allow_nan=False)
    except (ValueError, TypeError, RecursionError) as exc:
        raise RuntimeError("Fibo returned invalid native scene-analysis JSON.") from exc
    return result


async def analyze_image(*, base_url: str, model: str, image: bytes) -> dict[str, Any]:
    """Send one direct-answer request, with no SDK retries or constrained decoding."""
    with Image.open(BytesIO(image)) as source:
        buffer = BytesIO()
        source.convert("RGB").save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    try:
        async with AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key="local",
            timeout=3600.0,
            max_retries=0,
            http_client=DefaultAsyncHttpxClient(trust_env=False),
        ) as client:
            response = await client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "image_url", "image_url": {"url": url}},
                            {"type": "text", "text": FIBO_CAPTION_PROMPT},
                        ],
                    }
                ],
                n=1,
                temperature=1.0,
                top_p=0.95,
                max_tokens=4096,
                stop=["<|im_end|>", "<|endoftext|>"],
                extra_body={
                    "skip_special_tokens": False,
                    "special": True,
                    "chat_template_kwargs": {"enable_thinking": False},
                    "top_k": 20,
                    "min_p": 0.0,
                },
            )
    except APITimeoutError as exc:
        raise RuntimeError("Fibo inference timed out after 3600 seconds.") from exc
    except APIConnectionError as exc:
        raise RuntimeError(f"Could not connect to llama-swap at {base_url}.") from exc
    except APIStatusError as exc:
        raise RuntimeError(
            f"Fibo inference returned HTTP {exc.status_code}. Check the YAML and server logs."
        ) from exc
    try:
        choice = response.choices[0]
        content = choice.message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError("Fibo returned an unexpected model response.") from exc
    if choice.finish_reason == "length":
        raise RuntimeError(
            "Fibo reached its 4096-token output limit; no partial JSON was returned."
        )
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("Fibo returned no scene analysis.")
    return parse_scene(content)
