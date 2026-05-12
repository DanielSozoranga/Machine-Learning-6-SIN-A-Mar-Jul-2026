# Mid-Term Practical Evaluation — Machine Learning (6 SIN-A)

**Universidad Internacional del Ecuador — Escuela de Ciencias de la Computación**
**Autor:** Daniel Sozoranga
**Docente:** Ing. Jonathan E. Tito O., MSc.
**Dataset:** Wine Quality (Red Wine) — 1 599 filas, 11 features + target `quality`

---

## Cómo abrir el entorno

1. Aceptar la invitación de GitHub Classroom.
2. Click en **Code → Codespaces → Create codespace on main**.
3. Esperar ~60 s a que el DevContainer compile (venv `ds` se crea automáticamente).
4. Abrir `notebooks/exam.ipynb` desde el árbol de archivos.
5. Ejecutar la primera celda (Task 1). Si imprime versiones → listo.

## Estructura del repositorio

```
.
├── .devcontainer/        # Config Codespaces — no modificar
│   ├── Dockerfile        # Python 3.12-slim + UV + venv DS
│   └── devcontainer.json
├── data/
│   └── wine_quality.csv  # 1 599 filas, 11 features + 'quality'
├── notebooks/
│   └── exam.ipynb        # Notebook principal con las 11 tareas
├── src/
│   └── scale.py          # Módulo de escalado (Task 2)
├── pyproject.toml
└── README.md
```

## Cómo hacer commit (el profe revisa los timestamps)

```bash
# Al terminar Part A
git add . && git commit -m "Part A: env check, bug fix, pandas indexing"

# Al terminar Part B
git add . && git commit -m "Part B: EDA, regression pipeline, metrics"

# Al terminar Part C
git add . && git commit -m "Part C: logistic regression, class imbalance"

# Entrega final
git add . && git commit -m "Final exam submission" && git push
```

"Comit 6"