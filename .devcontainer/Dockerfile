# syntax=docker/dockerfile:1
#
# DevContainer para el notebook de regresión lineal y logística.
# Patrón UV + flag --python: instala las dependencias dentro de un venv
# aislado del Python del sistema, requisito para reproducibilidad estricta.

FROM python:3.12-slim AS base

# Variables de entorno de Python y de la instalación no interactiva:
#   PYTHONUNBUFFERED         → stdout/stderr sin buffer (logs en tiempo real).
#   PYTHONDONTWRITEBYTECODE  → no genera archivos .pyc en disco.
#   DEBIAN_FRONTEND          → suprime prompts interactivos de apt-get.
#   UV_LINK_MODE=copy        → fuerza a UV a copiar (no hardlink) en filesystems
#                              donde el hardlink falla (Codespaces, Docker Desktop).
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy

# Dependencias del sistema. `git` y `ca-certificates` son requeridos por UV
# para resolver paquetes desde repositorios HTTPS. `curl` queda disponible
# para diagnósticos. La limpieza de /var/lib/apt/lists reduce el tamaño
# de la imagen final.
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git \
    && rm -rf /var/lib/apt/lists/*

# UV se trae como binario estático desde la imagen oficial de Astral.
# Versión pinneada para garantizar reproducibilidad del resolver.
COPY --from=ghcr.io/astral-sh/uv:0.5.5 /uv /usr/local/bin/uv

WORKDIR /workspace

# Capa de dependencias (cacheable). Copiar primero los manifiestos permite
# que Docker reutilice esta capa en builds posteriores cuando solo cambian
# los notebooks o el README. El README.md se incluye porque pyproject.toml
# lo referencia como `readme = "README.md"` y hatchling lo valida al build.
COPY pyproject.toml uv.lock README.md ./

# Resolución e instalación de dependencias en tres pasos atómicos:
#   1. uv export --frozen          → genera requirements.txt desde uv.lock,
#                                     fallando si lock y pyproject divergen.
#      --extra bonus               → incluye jax + optax + polars (sección bonus).
#   2. uv venv ds                  → crea un venv aislado en /workspace/ds.
#   3. uv pip install --python ... → instala dentro del venv recién creado.
#                                     El flag --python es obligatorio: sin él,
#                                     UV resuelve al Python del sistema porque
#                                     en Docker no hay un venv "activo" entre
#                                     capas RUN. Sin este flag, las dependencias
#                                     quedan instaladas fuera de /workspace/ds.
RUN uv export --frozen --no-hashes --extra bonus -o /tmp/requirements.txt && \
    uv venv ds && \
    uv pip install --python ds/bin/python --no-cache --requirement /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# Código fuente del proyecto.
COPY notebooks ./notebooks

# Registro del kernel de Jupyter para que VS Code lo detecte automáticamente
# al abrir un .ipynb. El nombre `ds-final-ml` debe coincidir con el kernel
# referenciado en notebooks/main.ipynb (campo metadata.kernelspec).
RUN ds/bin/python -m ipykernel install --sys-prefix --name ds-final-ml \
        --display-name "Python 3.12 (DS Final ML)"

# Comando por defecto: arranca Jupyter Lab sin token ni password, accesible
# en el puerto 8888 forwardeado por el devcontainer.
CMD ["ds/bin/jupyter", "lab", \
     "--ip=0.0.0.0", "--port=8888", \
     "--no-browser", "--allow-root", \
     "--ServerApp.token=", "--ServerApp.password="]
