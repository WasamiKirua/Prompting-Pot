"""Async image-to-prompt calls adapted from the user's krea2_prompt.py."""

import base64

from openai import (
    APIConnectionError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    DefaultAsyncHttpxClient,
)
from openai.types.chat import ChatCompletionContentPartParam

TEMPERATURE = 0.7
MAX_TOKENS = 2000
REQUEST_TIMEOUT = 300.0


def build_user_content(
    description: str,
    image: bytes,
    content_type: str,
    image_generator: str,
) -> list[ChatCompletionContentPartParam]:
    """Combine requested changes (or as-is recreation) with the reference image."""
    text = description.strip()
    if not text or text.lower() in {
        "as-is",
        "as is",
        "recreate as-is",
        "recreate as is",
    }:
        text = f"Recreate this reference image as-is as a {image_generator} prompt."
    encoded = base64.b64encode(image).decode("ascii")
    return [
        {"type": "text", "text": text},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{content_type};base64,{encoded}"},
        },
    ]


async def generate_from_image(
    *,
    base_url: str,
    model: str,
    system_prompt: str,
    description: str,
    image: bytes,
    content_type: str,
    image_generator: str,
) -> str:
    """Send a local multimodal chat request and preserve the returned format.

    The model is a llama-swap model ID; image_generator selects instructions,
    not another inference server. No retry or remote fallback is performed.
    """
    try:
        async with AsyncOpenAI(
            base_url=base_url.rstrip("/"),
            api_key="local",
            timeout=REQUEST_TIMEOUT,
            max_retries=0,
            http_client=DefaultAsyncHttpxClient(trust_env=False),
        ) as client:
            result = await client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": build_user_content(
                            description,
                            image,
                            content_type,
                            image_generator,
                        ),
                    },
                ],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
                extra_body={"reasoning": "off"},
            )
    except APITimeoutError as exc:
        raise RuntimeError("llama-swap inference timed out after 300 seconds.") from exc
    except APIConnectionError as exc:
        raise RuntimeError(f"Could not connect to llama-swap at {base_url}.") from exc
    except APIStatusError as exc:
        raise RuntimeError(
            f"llama-swap returned HTTP {exc.status_code}. Check the model ID, YAML and server logs."
        ) from exc

    try:
        choice = result.choices[0]
        content = choice.message.content
    except (AttributeError, IndexError, TypeError) as exc:
        raise RuntimeError("llama-swap returned an unexpected model response.") from exc
    if not isinstance(content, str) or not content.strip():
        reasoning = getattr(choice.message, "reasoning_content", None)
        if choice.finish_reason == "length" and reasoning:
            raise RuntimeError(
                "The model used its output budget for reasoning without producing a prompt. "
                "Increase MAX_TOKENS or disable reasoning in the model configuration."
            )
        raise RuntimeError(
            f"llama-swap returned no prompt (finish_reason={choice.finish_reason!r})."
        )
    return content.strip()
