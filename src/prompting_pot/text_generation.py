"""Text-only prompt generation and native Fibo scene expansion."""

from typing import Any

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    DefaultAsyncHttpxClient,
)
from openai.types.chat import ChatCompletionMessageParam

from .fibo import parse_scene
from .ideogram_caption import IDEOGRAM_CAPTION_SCHEMA, serialize_ideogram_caption
from .prompts import build_text_prompt


async def generate_from_text(
    *,
    base_url: str,
    model: str,
    description: str,
    image_generator: str,
) -> str | dict[str, Any]:
    """Expand a scene description without an image; validate JSON before returning."""
    options: dict[str, Any] = {
        "temperature": 0.7,
        "max_tokens": 2000,
        "extra_body": {"reasoning": "off"},
    }
    messages: list[ChatCompletionMessageParam]
    if model == "Fibo":
        messages = [
            {
                "role": "user",
                "content": "<expand><full>\n<aspect_ratio>1.00</aspect_ratio>\n"
                + description,
            }
        ]
        options = {
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 4096,
            "stop": ["<|im_end|>", "<|endoftext|>"],
            "extra_body": {
                "skip_special_tokens": False,
                "special": True,
                "chat_template_kwargs": {"enable_thinking": False},
                "top_k": 20,
                "min_p": 0.0,
            },
        }
    else:
        messages = [
            {"role": "system", "content": build_text_prompt(image_generator)},
            {"role": "user", "content": description},
        ]
        if image_generator == "ideogram4":
            options = {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "ideogram4_prompt",
                        "schema": IDEOGRAM_CAPTION_SCHEMA,
                    },
                },
                "max_tokens": 4096,
                "extra_body": {"reasoning": "off"},
            }
    try:
        async with AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key="local",
            timeout=3600.0 if model == "Fibo" else 300.0,
            max_retries=0,
            http_client=DefaultAsyncHttpxClient(trust_env=False),
        ) as client:
            response = await client.chat.completions.create(
                model=model, messages=messages, **options
            )
    except APITimeoutError as exc:
        raise RuntimeError("Text inference timed out.") from exc
    except APIConnectionError as exc:
        raise RuntimeError(f"Could not connect to llama-swap at {base_url}.") from exc
    except APIStatusError as exc:
        raise RuntimeError(f"Text inference returned HTTP {exc.status_code}.") from exc
    try:
        choice = response.choices[0]
        content = choice.message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError("The model returned an unexpected text response.") from exc
    if choice.finish_reason == "length":
        raise RuntimeError(
            "Text generation reached its output limit; no partial prompt was returned."
        )
    if not isinstance(content, str) or not content.strip():
        raise RuntimeError("The model returned no prompt.")
    if model == "Fibo":
        return parse_scene(content)
    if image_generator == "ideogram4":
        return serialize_ideogram_caption(content)
    return content.strip()
