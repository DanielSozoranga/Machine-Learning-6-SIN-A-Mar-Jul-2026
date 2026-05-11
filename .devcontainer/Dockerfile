# ============================================================================
# Dockerfile — DS Final ML Project (UIDE 6 SIN-A)
#
# Patrón UV + --python flag (workshop pattern).
# ¿Por qué `--python .venv/bin/python` es OBLIGATORIO en Docker?
#   En el host, `uv` detecta automáticamente el venv activo. Dentro de Docker
#   NO hay shell interactiva ni `source activate` durante el RUN; cada RUN es
#   una capa nueva. Sin `--python`, UV resuelve al Python del sistema y las
#   dependencias quedan fuera del venv → el `.venv/` queda vacío y el notebook
#   no puede importarlas en runtime. El flag fuerza la instalación dentro del
#   venv que justamente acabamos de crear.
# ============================================================================
FROM python:3.12-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive \
    UV_LINK_MODE=copy

# --- Sistema base ---
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl ca-certificates git make \
    && rm -rf /var/lib/apt/lists/*

# --- Instalación de UV (bin oficial, único curl externo) ---
COPY --from=ghcr.io/astral-sh/uv:0.5.5 /uv /usr/local/bin/uv

WORKDIR /workspace

# --- Capa de dependencias (cacheable) ---
# Copiamos pyproject.toml, uv.lock y README.md (que el pyproject referencia
# como `readme = "README.md"` — hatchling lo necesita para validar el proyecto
# al instalar `-e .`). El README cambia poco, así que la cache se mantiene
# casi siempre que solo se modifique el notebook.
COPY pyproject.toml uv.lock README.md ./

# Crear venv y resolver desde el lockfile pinneado.
# `--frozen` exige que uv.lock coincida con pyproject.toml (reproducibilidad estricta).
# `--extra bonus` incluye jax + optax + polars para que el notebook corra completo
# (incluyendo la sección bonus) al primer "Run All".
#
# NOTA: exportamos primero a un archivo en lugar de usar process substitution
# `<(...)` porque el RUN de Docker corre con /bin/sh (no bash), y `<(...)`
# es sintaxis exclusiva de bash. Hacerlo así también es más cacheable
# (cada paso es atómico y Docker puede reaprovechar capas).
RUN uv export --frozen --no-hashes --extra bonus -o /tmp/requirements.txt && \
    uv venv .venv && \
    uv pip install --python .venv/bin/python --no-cache --requirement /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# --- Código del proyecto ---
COPY notebooks ./notebooks

# Registrar el kernel del venv para que Jupyter lo encuentre desde VS Code.
RUN .venv/bin/python -m ipykernel install --sys-prefix --name ds-final-ml --display-name "Python 3.12 (DS Final ML)"

# Por defecto: levantar Jupyter Lab apuntando al venv.
CMD [".venv/bin/jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--ServerApp.token=", "--ServerApp.password="]
