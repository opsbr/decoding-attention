FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

ENV UV_LINK_MODE=copy
ENV UV_COMPILE_BYTECODE=1

RUN useradd -ms /bin/bash appuser
USER appuser

WORKDIR /app
RUN \
  --mount=type=cache,target=/home/appuser/.cache/uv,uid=1000,gid=1000 \
  --mount=type=bind,source=uv.lock,target=uv.lock \
  --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
  uv sync --locked --no-install-project --no-dev

COPY --chown=appuser:appuser . /app
RUN \
  --mount=type=cache,target=/home/appuser/.cache/uv,uid=1000,gid=1000 \
  uv sync --locked --no-dev

RUN mkdir -p /home/appuser/.cache/huggingface && chown appuser:appuser /home/appuser/.cache/huggingface
VOLUME [ "/home/appuser/.cache/huggingface" ]

ENTRYPOINT []
CMD ["uv", "run", "streamlit", "run", "main.py", "--server.port=8501", "--server.address=0.0.0.0"]

EXPOSE 8501
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health
