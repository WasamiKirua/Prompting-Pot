"""Own one local llama-swap process; never terminate an external server."""

import asyncio
import logging
import os
import signal
from pathlib import Path

import httpx

BASE_URL = "http://0.0.0.0:8080/v1"
IP = "0.0.0.0"
STARTUP_TIMEOUT = 30.0
SHUTDOWN_TIMEOUT = 10.0
logger = logging.getLogger(__name__)


async def is_llama_swap_ready(base_url: str = BASE_URL) -> bool:
    """Check API readiness, not process ownership or which YAML it loaded."""
    try:
        async with httpx.AsyncClient(timeout=2.0, trust_env=False) as client:
            response = await client.get(f"{base_url.rstrip('/')}/models")
            response.raise_for_status()
            data = response.json()
            return isinstance(data, dict) and isinstance(data.get("data"), list)
    except (httpx.HTTPError, ValueError):
        return False


async def port_is_occupied() -> bool:
    """Avoid launching a second server when an unready service owns port 8080."""
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(f"{IP}", 8080), timeout=2.0
        )
    except ConnectionRefusedError:
        return False
    except OSError as exc:
        raise RuntimeError(
            "Cannot determine whether llama-swap port 8080 is free."
        ) from exc
    writer.close()
    await writer.wait_closed()
    return True


class LlamaSwap:
    """Serialize complete inference sessions and retain the process we own."""

    def __init__(self) -> None:
        self.lock = asyncio.Lock()
        self.process: asyncio.subprocess.Process | None = None

    async def start(self, config_path: Path) -> str:
        """Reuse a ready external API, or launch and await our local process.

        Caller holds `lock` until inference and cleanup have completed. An
        external API's selected configuration cannot be inferred from readiness.
        """
        if await is_llama_swap_ready():
            return BASE_URL
        if await port_is_occupied():
            raise RuntimeError(
                "Port 8080 is occupied but its API is not ready. "
                "Check the existing service rather than starting another llama-swap."
            )
        if not config_path.is_file():
            raise RuntimeError(
                f"llama-swap configuration does not exist: {config_path}"
            )
        self.process = await asyncio.create_subprocess_exec(
            "llama-swap",
            "--listen",
            f"{IP}:8080",
            "--config",
            str(config_path),
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
            start_new_session=(os.name == "posix"),
        )
        print(f"llama-swap PID: {self.process.pid}", flush=True)
        try:
            deadline = asyncio.get_running_loop().time() + STARTUP_TIMEOUT
            while asyncio.get_running_loop().time() < deadline:
                if self.process.returncode is not None:
                    raise RuntimeError(
                        f"llama-swap exited during startup (code {self.process.returncode}). "
                        "Check the YAML and executable arguments."
                    )
                if await is_llama_swap_ready():
                    if self.process.returncode is not None:
                        raise RuntimeError("llama-swap exited during startup.")
                    return BASE_URL
                await asyncio.sleep(0.25)
            raise RuntimeError("llama-swap did not become ready within 30 seconds.")
        except BaseException:
            # Includes request cancellation: never leave an owned startup behind.
            await self.stop()
            raise

    async def stop(self) -> None:
        """Request graceful shutdown, force termination after 10s, then reap.

        On Linux, signal the session's process group so model subprocesses also
        receive the shutdown request. Only processes launched here are owned.
        """
        process = self.process
        if process is None:
            return

        def send(sig: signal.Signals) -> None:
            try:
                if os.name == "posix":
                    os.killpg(process.pid, sig)
                elif process.returncode is None:
                    if sig == signal.SIGTERM:
                        process.terminate()
                    else:
                        process.kill()
            except ProcessLookupError:
                pass

        send(signal.SIGTERM)
        try:
            await asyncio.wait_for(process.wait(), timeout=SHUTDOWN_TIMEOUT)
        except TimeoutError:
            send(signal.SIGKILL)
            await process.wait()
        self.process = None
        logger.info("Stopped llama-swap PID: %s", process.pid)
