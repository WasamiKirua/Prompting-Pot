# Pinned upstream CUDA build: includes llama-server and its runtime libraries.
FROM ghcr.io/astral-sh/uv:0.12.23@sha256:61d393e44e249f2e4b526b6c7ddcecce245946826e608e11c93ad4f5bba55b21 AS uv
FROM ghcr.io/ggml-org/llama.cpp:server-cuda@sha256:fff6185edd2fbc4093aa5970bf6db53ed11c283e3a1c52a3e244cf27047264f4

USER root
RUN apt-get update \
    && apt-get install -y --no-install-recommends python3 python3-venv ca-certificates curl \
    && rm -rf /var/lib/apt/lists/*

COPY --from=uv /uv /usr/local/bin/uv

# Ship the same llama-swap release as the existing local installation.
ARG TARGETARCH
RUN case "$TARGETARCH" in \
      amd64) checksum=871b3ed7891f8ce057428f5a34d38298e06c36c6848c1136e05308eb4303af50 ;; \
      arm64) checksum=0daefbe0c37a32e193e2fb575ccd29787f46baf63ad691f887e513de913613c4 ;; \
      *) echo "Unsupported architecture: $TARGETARCH" >&2; exit 1 ;; \
    esac \
    && curl -fsSL "https://github.com/mostlygeek/llama-swap/releases/download/v262/llama-swap_262_linux_${TARGETARCH}.tar.gz" -o /tmp/llama-swap.tar.gz \
    && echo "$checksum  /tmp/llama-swap.tar.gz" | sha256sum -c - \
    && tar -xzf /tmp/llama-swap.tar.gz -C /usr/local/bin llama-swap \
    && chmod 755 /usr/local/bin/llama-swap \
    && rm /tmp/llama-swap.tar.gz

WORKDIR /app
ENV PROMPTING_POT_ROOT=/app \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_PYTHON_DOWNLOADS=never \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:/app:/usr/local/bin:${PATH}"

COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --python /usr/bin/python3
COPY src ./src
COPY config ./config
RUN uv sync --frozen --no-dev --no-editable --python /usr/bin/python3 \
    && uv cache clean \
    && useradd --create-home --uid 10001 app

USER app
EXPOSE 8000
# The parent image checks llama-server; our public service is FastAPI instead.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD ["/app/.venv/bin/python", "-c", "import urllib.request; opener = urllib.request.build_opener(urllib.request.ProxyHandler({})); opener.open('http://127.0.0.1:8000/api/targets', timeout=3).close()"]

# Override the upstream llama-server entrypoint; let the app own its lifecycle.
ENTRYPOINT []
CMD ["uvicorn", "prompting_pot.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--timeout-graceful-shutdown", "15"]
