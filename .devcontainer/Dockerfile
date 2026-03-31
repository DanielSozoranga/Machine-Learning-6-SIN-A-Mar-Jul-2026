FROM python:3.12-slim

# They are no longer needed after installing the packages, and deleting t

RUN apt-get update && apt-get install -y \
    curl \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

# Avoid git push errors on the container
RUN git lfs install --system

# downloads and installs curl
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# ENV defines an environment variable that persists in all subsequen
# and within the running container.

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

RUN uv venv .venv

COPY requirements.txt .

RUN uv pip install --python .venv/bin/python -r requirements.txt

ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:$PATH"
