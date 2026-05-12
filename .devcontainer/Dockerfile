# syntax=docker/dockerfile:1
#
# DevContainer — Mid-Term Practical Evaluation, Machine Learning (6 SIN-A)
# Patrón del taller de clase: WORKDIR /app, venv `.DS` expuesto vía VIRTUAL_ENV
# para que VS Code lo detecte automáticamente como kernel en el selector de Jupyter.

FROM python:3.12-slim

# ENV de Python y de instalación:
#   PYTHONUNBUFFERED        → stdout/stderr sin buffer (logs en tiempo real).
#   PYTHONDONTWRITEBYTECODE → no genera .pyc en disco.
#   DEBIAN_FRONTEND         → suprime prompts interactivos de apt-get.
#   UV_LINK_MODE=copy       → fuerza a UV a copiar en lugar de hardlink
#                             (necesario en Codespaces / Docker Desktop).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy

# Dependencias del sistema.
# git + ca-certificates requeridos por UV para resolver paquetes HTTPS.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git build-essential \
    && rm -rf /var/lib/apt/lists/*

# UV como binario estático desde la imagen oficial de Astral.
# Versión pinneada para reproducibilidad del resolver.
COPY --from=ghcr.io/astral-sh/uv:0.5.5 /uv /usr/local/bin/uv

WORKDIR /app

# Capa de dependencias (cacheable).
# Copiar manifiestos antes que el código permite reusar esta capa
# en builds posteriores cuando solo cambian los notebooks.
COPY pyproject.toml README.md ./

# Instalación en tres pasos atómicos:
#   1. uv venv .DS       → crea el venv aislado en /app/.DS (nombre del taller).
#   2. uv pip install    → instala dentro del venv con --python explícito
#                          (obligatorio en Docker: no hay un `source activate`
#                           que persista entre capas RUN).
RUN uv venv .DS && \
    uv pip install --python .DS/bin/python --no-cache \
        "pandas>=2.2,<4.0" \
        "numpy>=1.26,<3.0" \
        "scikit-learn>=1.5,<2.0" \
        "matplotlib>=3.9,<4.0" \
        "seaborn>=0.13,<1.0" \
        "jupyter>=1.1,<2.0" \
        "ipykernel>=6.29,<7.0" \
        "nbformat>=5.10,<6.0"

# Código fuente (capa separada → no invalida la capa de deps al editar notebooks).
COPY . /app/

# Exposición del venv al entorno del container:
#   VIRTUAL_ENV → VS Code Python escanea esta variable y reconoce el venv
#                 como "Virtual Env" en el selector de kernel de Jupyter.
#   PATH        → resuelve `python`, `jupyter` al binario del venv sin
#                 necesidad de activación manual.
ENV VIRTUAL_ENV=/app/.DS
ENV PATH="/app/.DS/bin:$PATH"

# Jupyter Lab en el puerto 8888 sin autenticación (Codespaces ya gestiona auth).
CMD ["jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", "--allow-root", \
     "--ServerApp.token=", "--ServerApp.password="]
