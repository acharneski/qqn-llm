FROM nvidia/cuda:12.4.1-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    XLA_PYTHON_CLIENT_MEM_FRACTION=0.92 \
    JAX_COMPILATION_CACHE_DIR=/tmp/jax_cache

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.12 python3.12-venv python3-pip git curl ca-certificates awscli \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml ./
COPY qqn-jax ./qqn-jax
COPY src ./src
COPY configs ./configs

RUN python3.12 -m venv /opt/venv \
    && /opt/venv/bin/pip install --no-cache-dir --upgrade pip \
    && /opt/venv/bin/pip install --no-cache-dir -e ./qqn-jax -e ".[cuda]"

ENV PATH="/opt/venv/bin:${PATH}"

ENTRYPOINT ["qqn-llm-train"]
