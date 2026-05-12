# Linear & Logistic Regression on California Housing

**Daniel Sozoranga** · Machine Learning · Prof. Ing. Jonathan E. Tito O., MSc. · Branch `p2`

---

## Dataset

California Housing (`sklearn.datasets.fetch_california_housing`) — 20,640 distritos censales × 8 features socioeconómicas.

| Target | Tipo | Definición |
|---|---|---|
| `MedHouseVal` | Continuo | Precio mediano por distrito (×100,000 USD) |
| `expensive` | Binario | `1 si MedHouseVal > mediana, else 0` |

El split por la mediana produce clases balanceadas (~50/50), lo que aísla el efecto neto de la regularización L2 sin necesidad de `class_weight`.

---

## Stack y justificación

**Pandas · Scikit-learn · Matplotlib/Seaborn.**

- **Scikit-learn** para preprocesamiento y modelado. El patrón `ColumnTransformer → Pipeline` previene data leakage por diseño: el `StandardScaler` aprende media y desviación únicamente del `fit` sobre `X_train`, y el `transform` sobre `X_test` reusa esos parámetros.
- **`LogisticRegression(penalty="l2", C=1.0)`** satisface la regularización L2 exigida sin implementación manual, dejando el bonus libre para una reimplementación independiente desde cero.
- **Pandas** como interfaz tabular: integración nativa con sklearn (`as_frame=True`), seaborn y matplotlib.
- **Matplotlib + Seaborn** cubren las tres visualizaciones de EDA requeridas: `histplot` (target), `boxplot` (features), `heatmap` (correlación de Pearson).

---

## Cómo ejecutar

### Opción A — GitHub Codespaces (recomendada)

1. `Code → Codespaces → Create codespace on p2`
2. Espera ~2 minutos al build del devcontainer.
3. Abre `notebooks/main.ipynb`, selecciona el kernel `Python 3.12 (DS Final ML)` y ejecuta `Run All`.

### Opción B — Docker local

```bash
git clone -b p2 https://github.com/DanielSozoranga/DS_RegresionesJAX.git
cd DS_RegresionesJAX
docker build -f .devcontainer/Dockerfile -t ds-final-ml . && \
docker run --rm -p 8888:8888 -v "$PWD":/app ds-final-ml
```

---

## Reproducibilidad

| Mecanismo | Implementación |
|---|---|
| Dependencias pinneadas | `uv.lock` con 128 paquetes resueltos (`--frozen`) |
| Patrón de instalación en Docker | `uv pip install --python ds/bin/python --requirement ...` |
| Aleatoriedad controlada | `random_state = 42` para `train_test_split` y el PRNG de JAX |
| Entorno aislado | DevContainer construido desde `python:3.12-slim` |

El flag `--python ds/bin/python` es obligatorio en Docker: cada `RUN` es una capa nueva sin shell interactiva, así que sin `--python` UV resolvería al Python del sistema y las dependencias quedarían fuera del venv.

---

## Resultados (Test set, 4,128 distritos)

| Modelo | Métrica | Valor |
|---|---|---|
| **Linear Regression** | R² | 0.7806 |
|  | MAE | 0.4232 (≈ $42,318 USD) |
|  | RMSE | 0.5302 (≈ $53,018 USD) |
| **Logistic Regression L2** (`C=1.0`) | Accuracy | 0.8413 |
|  | Precision | 0.8452 |
|  | Recall | 0.8358 |
|  | F1 | 0.8404 |
|  | ROC-AUC | 0.9282 |
| **Bonus — JAX desde cero** | Diff vs sklearn | 0.00000 |
| **Bonus — Ecuación Normal** | Coincidencia triple sklearn/GD/Normal | R² idéntico a 4 decimales |
| **Bonus — Benchmark JIT** | Speedup XLA medido | ≈ 11× |

---

## Estructura del repositorio

```
.
├── .devcontainer/
│   ├── Dockerfile             # Imagen Python 3.12 + UV + venv aislado
│   └── devcontainer.json      # Configuración de VS Code / Codespaces
├── notebooks/
│   └── main.ipynb             # Deliverable principal — 7 secciones + bonus
├── data/
│   └── california_housing.csv # Fallback offline (mismo schema que el oficial)
├── scripts/
│   └── build_notebook.py      # Regeneración programática del notebook
├── pyproject.toml             # Dependencias del proyecto + extra [bonus]
├── uv.lock                    # 128 paquetes pinneados
├── .gitignore
└── README.md
```

---

## Mapeo al rubric

| Sección del rubric | Ubicación en el notebook |
|---|---|
| 1. Problem Definition | §1 |
| 2. EDA — 3 visualizaciones | §2.1 (target), §2.2 (features), §2.3 (correlación) |
| 3. Preprocessing Pipeline | §3 — `ColumnTransformer` + `StandardScaler` dentro de `Pipeline` |
| 4. Train / Test Split | §4 — 80/20, `random_state=42`, estratificado por `expensive` |
| 5. Linear Regression | §5 — R², MAE, RMSE + plot de coeficientes + scatter predicho vs real |
| 6. Logistic Regression L2 | §6 — métricas completas + matriz de confusión + curva ROC + comparación de pesos |
| 7. Discussion | §7 — outliers, multicolinealidad, overfitting |
| Bonus (rubric §3) | §🎁.1 Polars LazyFrame · §🎁.2 Ecuación Normal · §🎁.3 Benchmark JIT |

---

## AI Usage Disclosure

Asistente utilizado: **Claude (Anthropic)**.

| Tarea | Tipo de asistencia |
|---|---|
| Boilerplate del `Dockerfile` y `pyproject.toml` | Generación inicial del patrón UV |
| Esqueleto de las celdas markdown del notebook | Borrador inicial de cada sección |
| Plantilla del bonus JAX | Generación de gradientes con `jax.grad` + Adam + `@jax.jit` |
| Justificación de stack (4 oraciones) | Borrador inicial |

**Decisiones propias (no asistidas):** elección del dataset, decisión de aplicar L2, interpretación de coeficientes, contenido de la sección Discussion, y elección de las herramientas del bonus.
