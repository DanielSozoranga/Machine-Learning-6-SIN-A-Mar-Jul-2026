# syntax=docker/dockerfile:1
#
# DevContainer — Mid-Term Practical Evaluation, Machine Learning (6 SIN-A)
# Sigue el patrón del workshop de la clase: WORKDIR /app, venv `ds` expuesto
# vía VIRTUAL_ENV para que VS Code lo detecte como kernel en el selector de Jupyter.

FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy

# git + ca-certificates requeridos por UV para resolver paquetes HTTPS.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/*

# UV pinneado a 0.5.5 para reproducibilidad del resolver.
COPY --from=ghcr.io/astral-sh/uv:0.5.5 /uv /usr/local/bin/uv

WORKDIR /app

# Capa cacheable de dependencias.
COPY pyproject.toml README.md ./

# Crear venv ds + instalar deps dentro.
# --python ds/bin/python es obligatorio en Docker: no hay venv "activo"
# entre capas RUN, así que hay que apuntar explícitamente al binario.
RUN uv venv ds && \
    uv pip install --python ds/bin/python --no-cache \
        "pandas>=2.2,<4.0" \
        "numpy>=1.26,<3.0" \
        "scikit-learn>=1.5,<2.0" \
        "matplotlib>=3.9,<4.0" \
        "seaborn>=0.13,<1.0" \
        "jupyter>=1.1,<2.0" \
        "ipykernel>=6.29,<7.0" \
        "nbformat>=5.10,<6.0"

# Código fuente (capa separada → editar notebooks no invalida capa de deps).
COPY . /app/

# Exposición del venv al entorno del container:
#   VIRTUAL_ENV → VS Code Python escanea esta variable y reconoce el venv
#                 como "ds (Python 3.12.13)" en el selector de kernel.
#   PATH        → resuelve `python`, `pip`, `jupyter` al binario del venv
#                 automáticamente, sin necesidad de activación manual.
ENV VIRTUAL_ENV=/app/ds
ENV PATH="/app/ds/bin:$PATH"

CMD ["sleep", "infinity"]
