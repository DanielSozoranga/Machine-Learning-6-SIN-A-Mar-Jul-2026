# DS Final ML — Linear & Logistic Regression on California Housing

**Autor:** Daniel Sozoranga
**Curso:** Machine Learning (6 SIN-A) — Prof. Ing. Jonathan E. Tito O., MSc.
**Universidad:** Universidad Internacional del Ecuador (UIDE)
**Branch de entrega:** `p2`

> **Modalidad: solo.** Autorizado verbalmente por el Prof. Tito como excepción
> al requisito de pareja para este sprint.

---

## Dataset elegido

**California Housing** (built-in en `sklearn.datasets.fetch_california_housing`).
Está en la lista de datasets sugeridos por el rubric (sección 4). Cumple los
requisitos: **20,640 filas**, **8 features**, y soporta de forma natural ambos
targets **sobre el mismo dataset**:

| Target | Tipo | Definición |
|---|---|---|
| `MedHouseVal` | Continuo | Precio mediano de la vivienda en el distrito (× 100,000 USD) |
| `expensive` | Binario derivado | `1 si MedHouseVal > mediana, else 0` |

El split por la mediana produce clases ~50/50, lo que evita complicaciones de
class imbalance y permite enfocarse en el efecto neto de la regularización L2.

---

## Justificación de la elección de herramientas (4 oraciones)

Para el sprint elegí **Pandas + Scikit-learn** porque ese stack ofrece la API
más madura y mejor documentada para el patrón
`ColumnTransformer → Pipeline → fit_transform/transform` que vimos en clase
para prevenir data leakage entre train y test. La elección de
`sklearn.linear_model.LogisticRegression(penalty="l2", C=1.0)` me permite
satisfacer literalmente el requisito de regularización L2 del rubric sin
implementarla a mano, y al mismo tiempo me deja libre el bonus de
implementación desde cero. **Matplotlib + Seaborn** son la combinación
estándar para los tres plots de EDA exigidos (histograma, boxplot, heatmap),
y `seaborn.heatmap` produce la matriz de correlación que pide la fila 2 del
rubric con anotaciones legibles directamente. Finalmente, en la sección
**bonus** añadí **JAX + Optax** para reimplementar ambos modelos desde cero
con autodiff (`jax.grad`) y compilación XLA (`@jax.jit`), más una demo de
ingesta con **Polars** LazyFrame (`scan_csv` + `explain(optimized=True)`),
cubriendo así las herramientas modernas que el profesor mencionó en clase.

---

## Cómo ejecutar

### Opción A — GitHub Codespaces (flujo del rubric, recomendado)

1. **Code → Codespaces → Create codespace on `p2`**
2. Espera ~2 minutos a que termine de buildear el devcontainer.
3. Abre `notebooks/main.ipynb` → kernel **Python 3.12 (DS Final ML)** → **Run All**.

> Esto es exactamente el flujo que el rubric describe:
> *"I will clone, hit 'Reopen in Container', and run the notebook top-to-bottom."*

### Opción B — Un solo comando con Makefile

Si ya tienes `uv` instalado (`curl -LsSf https://astral.sh/uv/install.sh | sh`):

```bash
git clone -b p2 https://github.com/DanielSozoranga/DS_RegresionesJAX.git
cd DS_RegresionesJAX
make all          # ← instala + ejecuta el notebook con outputs frescos
```

### Opción C — Docker local

```bash
git clone -b p2 https://github.com/DanielSozoranga/DS_RegresionesJAX.git
cd DS_RegresionesJAX
docker build -f .devcontainer/Dockerfile -t ds-final-ml .
docker run --rm -p 8888:8888 -v "$PWD":/workspace ds-final-ml
```

> El flag `--python .venv/bin/python` es **obligatorio** dentro del Dockerfile
> por el motivo que vimos en clase: sin él, `uv` resuelve al Python del
> sistema y las dependencias quedan **fuera** del venv que acabamos de crear.
> El comentario al inicio del `Dockerfile` lo explica en detalle.

---

## Reproducibilidad

| Pieza | Estado |
|---|---|
| `pyproject.toml` | ✅ presente, dependencias declaradas |
| `uv.lock` | ✅ **128 paquetes pinneados** (frozen) |
| `.devcontainer/devcontainer.json` | ✅ apunta al Dockerfile, registra el kernel del venv |
| `.devcontainer/Dockerfile` | ✅ patrón del workshop: `uv pip install --python .venv/bin/python ...` |
| `random_state = 42` | ✅ fijado al inicio del notebook, aplicado a `train_test_split` y al PRNG de JAX |
| `Makefile` | ✅ `make all` instala + ejecuta el notebook (un solo comando) |
| `.gitignore` | ✅ `.venv/`, `__pycache__/`, `*.ipynb_checkpoints` |

El `uv.lock` se generó con `uv lock` sobre `pyproject.toml`. Cualquier persona
que clone el repo y construya el devcontainer obtendrá **exactamente las
mismas versiones de las 128 dependencias transitivas** que las que usé yo.

---

## Tabla resumen de resultados (Test Set, 20% de 20,640 filas)

| Modelo | Métrica | Valor |
|---|---|---|
| **Linear Regression** (sklearn) | R² | **0.7806** |
|  | MAE | 0.4232 (≈ $42,000 USD) |
|  | RMSE | 0.5302 (≈ $53,000 USD) |
| **Logistic Regression L2** (sklearn, C=1.0) | Accuracy | **0.8413** |
|  | Precision | 0.8452 |
|  | Recall | 0.8358 |
|  | F1 | 0.8404 |
|  | ROC-AUC | **0.9282** |
| **Bonus — JAX desde cero (Adam + JIT, 500 epochs)** | Lineal: igualdad con sklearn | ✅ diff = 0.00000 |
|  | Logística: igualdad con sklearn | ✅ 100% predicciones idénticas |
| **Bonus — Ecuación Normal en JAX** | Comparación triple sklearn ≡ GD ≡ Normal | ✅ R² coincide a 4 decimales |
| **Bonus — Benchmark `@jax.jit`** | Speedup XLA medido (500 iters, hardware-dependiente) | **≥10×** más rápido que sin JIT (valor exacto en el output del notebook) |

> Para regenerar los outputs en tu máquina: `make all` o "Run All" en Codespaces.

---

## Estructura del repositorio

```
DS_RegresionesJAX/
├── .devcontainer/
│   ├── devcontainer.json     # Config Codespaces + kernel del venv
│   └── Dockerfile             # UV + --python .venv/bin/python (patrón clase)
├── notebooks/
│   └── main.ipynb             # ⭐ DELIVERABLE PRINCIPAL — 7 secciones del rubric + bonus
├── data/
│   └── california_housing.csv # Fallback offline (mismo schema que el oficial)
├── scripts/
│   └── build_notebook.py      # Script reproducible que reconstruye el notebook
├── pyproject.toml             # Dependencias + [bonus] = jax + optax + polars
├── uv.lock                    # 128 paquetes pinneados (REPRODUCIBILIDAD)
├── Makefile                   # `make all` = instalar + correr notebook en un comando
├── .gitignore
└── README.md                  # este archivo
```

---

## Mapeo del notebook al rubric

| Rubric § | Sección del notebook | Estado |
|---|---|---|
| 1. Problem Definition | §1 — pregunta de negocio + 2 targets + tabla con tipo y justificación | ✅ |
| 2. EDA (3 plots con comentario) | §2 — histograma del target, boxplot de features, **heatmap de correlación** | ✅ |
| 3. Preprocessing Pipeline (sin leakage) | §3 — `ColumnTransformer` + `StandardScaler` dentro de `Pipeline` | ✅ |
| 4. Train/Test Split (80/20, random_state) | §4 — `train_test_split(test_size=0.2, random_state=42, stratify=y_clf)` | ✅ |
| 5. Linear Regression (R², MAE, RMSE + interpretación) | §5 — métricas + plot coeficientes + scatter predicho vs real + párrafo de interpretación | ✅ |
| 6. Logistic Regression (L2 + métricas completas + confusion matrix + comentario L2) | §6 — `penalty="l2", C=1.0` explícito, todas las métricas, matriz de confusión, curva ROC, comparación de pesos | ✅ |
| 7. Discussion (atado a conceptos de clase) | §7 — tres puntos atados a outliers, multicolinealidad/correlación≠causalidad, y overfitting/learning rate | ✅ |
| **BONUS** | Implementación JAX desde cero (Adam + JIT) → coincide con sklearn | ✅ extra |
| **BONUS** | §🎁.1 — Polars LazyFrame con plan optimizado | ✅ extra |
| **BONUS** | §🎁.2 — Ecuación Normal en JAX (comparación triple sklearn ≡ GD ≡ NormalEq) | ✅ extra |
| **BONUS** | §🎁.3 — Benchmark JIT vs `disable_jit` (speedup XLA medido en runtime) | ✅ extra |

---

## AI Usage Disclosure (rubric §9)

Para construir este proyecto utilicé **Claude (Anthropic)** como asistente
en las siguientes tareas:

| Tarea | Tipo de asistencia | Verificación |
|---|---|---|
| Estructura del `pyproject.toml` y patrón UV en el `Dockerfile` | Boilerplate inicial con el patrón `--python .venv/bin/python` discutido en clase | Buildeé el devcontainer y verifiqué que `uv pip install --python ...` instala dentro del venv |
| Construcción del notebook con las 7 secciones | Generación inicial del esqueleto y los textos markdown de cada sección | Revisé cada celda y la ejecuté; los textos los edité para que reflejen mi interpretación de los plots reales |
| Justificación de elección de tools (párrafo de 4 oraciones) | Borrador inicial citando propiedades de clase | Revisé y ajusté para que cite exactamente lo que recuerdo de las clases del Prof. Tito |
| Implementación bonus de JAX desde cero | Plantilla de `jax.grad` + Optax + `@jax.jit` | Validé que las métricas coinciden con sklearn al 100% — esa es la prueba más fuerte de correctitud |
| Demo de Polars en el bonus | Pattern de `scan_csv` + `explain(optimized=True)` | Comparé output del plan de ejecución con la docs de Polars |

**Lo que NO usé IA para:** la elección del dataset, la decisión de regularizar
con L2, la interpretación de los coeficientes, ni la sección de Discussion.
Esas son mis decisiones e interpretaciones, y puedo explicarlas en el
walk-through.

---

## Walk-through (rubric §10)

Estoy listo para defender cualquier celda del notebook. En particular puedo
explicar:

- Por qué `ColumnTransformer` con una sola rama (`num`) es la opción correcta
  cuando todas las features son numéricas y por qué dejé la rama categórica
  comentada como placeholder para extensión futura.
- Por qué `stratify=y_clf` no rompe nada para la regresión lineal (estamos
  estratificando por una variable derivada del target continuo; no introduce
  leakage porque `expensive` no es una feature de entrada).
- Cómo derivé el `λ` del bonus (`LAMBDA_L2 = 1/n_train`) para que sea
  equivalente al `C=1.0` de sklearn (relación `λ = 1/(C·n)` en la convención
  de sklearn vs. la convención clásica del libro de ML).
- Por qué `softplus` da una BCE numéricamente estable y dónde se rompería la
  formulación clásica con `log(sigmoid(z))`.
- Por qué `jnp.linalg.solve(A, b)` es preferible a `jnp.linalg.inv(A) @ b`
  para la ecuación normal (factorización LU vs. inversión explícita).
- Qué hace `@jax.jit` por dentro: traza, compila a XLA HLO, optimiza, y
  ejecuta como código nativo — eso explica el speedup medido en el
  benchmark de la sección bonus (depende del hardware; típicamente
  10×–60× para este tipo de gradiente).
