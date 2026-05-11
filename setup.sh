#!/usr/bin/env bash
# ============================================================================
# setup.sh — Instalación de un solo comando (alternativa al Makefile)
#
# Uso:
#   bash setup.sh
#
# Esto es por si alguien NO usa el DevContainer y quiere reproducir el setup
# en su propia máquina con un único comando. Equivalente a `make all`.
# ============================================================================
set -euo pipefail

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  DS Final ML — Setup automático"
echo "════════════════════════════════════════════════════════════"

# 1. Verificar / instalar uv
if ! command -v uv >/dev/null 2>&1; then
    echo "→ Instalando uv (Astral)..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi
echo "✅ uv $(uv --version | cut -d' ' -f2)"

# 2. Crear venv + instalar dependencias (incluye [bonus])
echo "→ Resolviendo dependencias desde uv.lock (128 paquetes pinneados)..."
uv venv .venv
uv export --frozen --no-hashes --extra bonus -o /tmp/ds_requirements.txt
uv pip install --python .venv/bin/python --no-cache --requirement /tmp/ds_requirements.txt
rm -f /tmp/ds_requirements.txt
echo "✅ Venv listo en .venv/"

# 3. Registrar kernel Jupyter
.venv/bin/python -m ipykernel install --user --name ds-final-ml --display-name "Python 3.12 (DS Final ML)"
echo "✅ Kernel 'Python 3.12 (DS Final ML)' registrado"

# 4. Ejecutar el notebook end-to-end
echo "→ Ejecutando notebook (esto tarda ~1-2 min)..."
.venv/bin/python -m jupyter nbconvert \
    --to notebook \
    --execute notebooks/main.ipynb \
    --output main.ipynb \
    --ExecutePreprocessor.timeout=600 \
    >/dev/null 2>&1
echo "✅ Notebook ejecutado con outputs guardados"

echo ""
echo "════════════════════════════════════════════════════════════"
echo "  ✅ TODO LISTO"
echo "════════════════════════════════════════════════════════════"
echo "  Abre notebooks/main.ipynb para ver los resultados."
echo ""
