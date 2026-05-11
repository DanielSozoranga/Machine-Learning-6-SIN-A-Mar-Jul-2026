# ============================================================================
# Makefile — DS Final ML (UIDE 6 SIN-A)
#
# Conveniencia para ejecutar TODO el proyecto con un solo comando: `make all`.
# Útil para sanity check antes de entregar y para la defensa en clase.
#
# El flujo oficial del rubric sigue siendo Codespaces + "Reopen in Container"
# + "Run All" en el notebook. Este Makefile es complementario.
# ============================================================================

# Python del venv (se crea con `make install`)
PY := .venv/bin/python
UV := uv

.PHONY: help all install lock notebook clean

help:  ## Muestra esta ayuda
	@echo "Targets disponibles:"
	@echo "  make all       → install + ejecuta el notebook end-to-end (un solo comando)"
	@echo "  make install   → crea .venv y resuelve dependencias desde uv.lock"
	@echo "  make lock      → regenera uv.lock desde pyproject.toml"
	@echo "  make notebook  → ejecuta notebooks/main.ipynb con outputs guardados"
	@echo "  make clean     → borra .venv y caches"

all: install notebook  ## Hace todo: instala y corre el notebook
	@echo ""
	@echo "✅ Listo. Abre notebooks/main.ipynb para ver outputs."

install:  ## Crea el venv e instala dependencias desde uv.lock (con bonus)
	@command -v $(UV) >/dev/null || (echo "❌ Falta uv. Instálalo: curl -LsSf https://astral.sh/uv/install.sh | sh" && exit 1)
	$(UV) venv .venv
	$(UV) export --frozen --no-hashes --extra bonus -o /tmp/ds_requirements.txt
	$(UV) pip install --python $(PY) --requirement /tmp/ds_requirements.txt
	@rm -f /tmp/ds_requirements.txt
	$(PY) -m ipykernel install --user --name ds-final-ml --display-name "Python 3.12 (DS Final ML)"

lock:  ## Regenera uv.lock (cuando cambia pyproject.toml)
	$(UV) lock

notebook:  ## Ejecuta el notebook top-to-bottom (guarda outputs in-place)
	$(PY) -m jupyter nbconvert \
		--to notebook \
		--execute notebooks/main.ipynb \
		--output main.ipynb \
		--ExecutePreprocessor.timeout=600

clean:  ## Borra el venv y caches
	rm -rf .venv .pytest_cache .ruff_cache **/__pycache__ **/.ipynb_checkpoints
