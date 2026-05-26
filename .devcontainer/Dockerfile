FROM python:3.12-slim

RUN apt-get update && apt-get install -y \
    curl \
    git \
    git-lfs \
    && rm -rf /var/lib/apt/lists/*

RUN git lfs install --system

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

RUN uv venv ds

COPY requirements.txt .

RUN uv pip install --python ds/bin/python -r requirements.txt

RUN ds/bin/python -m ipykernel install \
        --prefix=/usr/local \
        --name ds \
        --display-name "Python 3.12 (ds)"

ENV VIRTUAL_ENV=/app/ds
ENV PATH="/app/ds/bin:$PATH"
