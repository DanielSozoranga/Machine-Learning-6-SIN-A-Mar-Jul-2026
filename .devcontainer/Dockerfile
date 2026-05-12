# syntax=docker/dockerfile:1
#
# DevContainer — Mid-Term Practical Evaluation, Machine Learning (6 SIN-A)
# Patrón del taller de clase: WORKDIR /app, venv `ds` expuesto vía VIRTUAL_ENV
# para que VS Code lo detecte automáticamente como kernel en el selector de Jupyter.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy

RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY --from=ghcr.io/astral-sh/uv:0.5.5 /uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml README.md ./

# Crear venv ds + instalar deps + registrar kernel para VS Code Jupyter.
RUN uv venv ds && \
    uv pip install --python ds/bin/python --no-cache \
        "pandas>=2.2,<4.0" \
        "numpy>=1.26,<3.0" \
        "scikit-learn>=1.5,<2.0" \
        "matplotlib>=3.9,<4.0" \
        "seaborn>=0.13,<1.0" \
        "jupyter>=1.1,<2.0" \
        "ipykernel>=6.29,<7.0" \
        "nbformat>=5.10,<6.0" && \
    /app/ds/bin/python -m ipykernel install \
        --name=ds \
        --display-name="Python 3.12 (ds)" \
        --prefix=/usr/local

ENV VIRTUAL_ENV=/app/ds
ENV PATH="/app/ds/bin:$PATH"

COPY . /app/

CMD ["sleep", "infinity"]
