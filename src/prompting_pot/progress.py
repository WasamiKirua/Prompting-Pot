"""Request-scoped caption progress; no persistent jobs or downloadable artifacts."""

import asyncio
import base64
import json
import logging
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import suppress
from typing import Any

from fastapi import HTTPException
from fastapi.responses import StreamingResponse

ProgressCallback = Callable[[dict[str, Any]], Awaitable[None]]
BatchRunner = Callable[
    [ProgressCallback | None], Awaitable[tuple[bytes, dict[str, int]]]
]


def caption_progress_response(run: BatchRunner) -> StreamingResponse:
    """Stream NDJSON updates and the final ZIP; cancel work on disconnect."""

    async def events() -> AsyncIterator[bytes]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue(maxsize=1)

        async def produce() -> None:
            try:
                archive, counts = await run(queue.put)
                await queue.put(
                    {
                        "type": "complete",
                        "counts": counts,
                        "archive": base64.b64encode(archive).decode("ascii"),
                    }
                )
            except HTTPException as exc:
                await queue.put({"type": "error", "message": str(exc.detail)})
            except (RuntimeError, ValueError, OSError) as exc:
                await queue.put({"type": "error", "message": str(exc)})
            except Exception:
                logging.getLogger(__name__).exception("Caption progress request failed")
                await queue.put(
                    {"type": "error", "message": "Captioning failed unexpectedly."}
                )

        task = asyncio.create_task(produce())
        try:
            yield (
                json.dumps({"type": "stage", "message": "Preparing your batch…"}) + "\n"
            ).encode()
            while True:
                event = await queue.get()
                yield (json.dumps(event) + "\n").encode()
                if event["type"] in {"complete", "error"}:
                    break
        finally:
            task.cancel()
            with suppress(asyncio.CancelledError):
                await task

    return StreamingResponse(
        events(),
        media_type="application/x-ndjson",
        headers={"Cache-Control": "no-store", "X-Accel-Buffering": "no"},
    )
