# Linear and Logistic Regression on California Housing

**Daniel Sozoranga** · Machine Learning · Prof. Ing. Jonathan E. Tito O., MSc. · Branch `p2`

---

## Dataset

California Housing (`sklearn.datasets.fetch_california_housing`): 20,640 census districts and 8 socioeconomic features.

| Target | Type | Definition |
|---|---|---|
| `MedHouseVal` | Continuous | Median house value per district (in 100,000 USD) |
| `expensive` | Binary | `1 if MedHouseVal > median, else 0` |

Splitting at the median yields a balanced binary target (approximately 50/50), which isolates the net effect of L2 regularization without requiring `class_weight` adjustments.

---

## Stack and justification

**Pandas, Scikit-learn, Matplotlib/Seaborn.**

- **Scikit-learn** is used for preprocessing and modeling. The `ColumnTransformer` plus `Pipeline` pattern prevents data leakage by construction: `StandardScaler` learns mean and standard deviation only from `fit` on `X_train`, and `transform` on `X_test` reuses those parameters.
- **`LogisticRegression(penalty="l2", C=1.0)`** satisfies the L2 regularization requirement without a manual implementation, freeing the bonus section for an independent from-scratch reimplementation.
- **Pandas** is the tabular interface: native integration with scikit-learn (`as_frame=True`), seaborn, and matplotlib.
- **Matplotlib and Seaborn** cover the three required EDA visualizations: `histplot` (target), `boxplot` (features), and `heatmap` (Pearson correlation).

---

## How to run

### Option A: GitHub Codespaces (recommended)

1. `Code -> Codespaces -> Create codespace on p2`
2. Wait approximately two minutes for the devcontainer build.
3. Open `notebooks/main.ipynb` and click **Run All**.
4. A kernel picker will appear automatically. Click **Python Environments...**, then select **ds (Python 3.12.13)** `/app/ds/bin/python` (marked as *Recommended*).
5. The notebook will resume execution with the correct environment.

### Option B: Local Docker

```bash
git clone -b p2 https://github.com/DanielSozoranga/Machine-Learning-6-SIN-A-Mar-Jul-2026.git
cd Machine-Learning-6-SIN-A-Mar-Jul-2026
docker build -f .devcontainer/Dockerfile -t ds-final-ml . && \
docker run --rm -p 8888:8888 -v "$PWD":/app ds-final-ml
```

---

## Reproducibility

| Mechanism | Implementation |
|---|---|
| Pinned dependencies | `uv.lock` with 128 resolved packages (`--frozen`) |
| Docker install pattern | `uv pip install --python ds/bin/python --requirement ...` |
| Controlled randomness | `random_state = 42` for `train_test_split` and the JAX PRNG |
| Isolated environment | DevContainer built from `python:3.12-slim` |

The `--python ds/bin/python` flag is mandatory inside Docker: each `RUN` is a new layer without an interactive shell, so without `--python`, UV would resolve to the system Python and the dependencies would be installed outside the venv.

---

## Results (test set, 4,128 districts)

| Model | Metric | Value |
|---|---|---|
| **Linear Regression** | R-squared | 0.6267 |
|  | MAE | 0.5215 (approximately 52,153 USD) |
|  | RMSE | 0.7033 (approximately 70,335 USD) |
| **Logistic Regression L2** (`C=1.0`) | Accuracy | 0.8314 |
|  | Precision | 0.8339 |
|  | Recall | 0.8274 |
|  | F1 | 0.8307 |
|  | ROC-AUC | 0.9134 |
| **Bonus: JAX from scratch** | Diff vs scikit-learn | 0.00000 |
| **Bonus: Normal Equation** | Triple match scikit-learn / GD / Normal Eq | R-squared identical to 4 decimals |
| **Bonus: JIT Benchmark** | Measured XLA speedup | approximately 10x |

---

## Repository structure

```
.
+-- .devcontainer/
|   +-- Dockerfile             (Python 3.12 + UV + isolated venv)
|   +-- devcontainer.json      (VS Code / Codespaces configuration)
+-- notebooks/
|   +-- main.ipynb             (Main deliverable: 7 sections plus bonus)
+-- data/
|   +-- california_housing.csv (Offline fallback, same schema as the official dataset)
+-- scripts/
|   +-- build_notebook.py      (Programmatic notebook regeneration)
+-- pyproject.toml             (Project dependencies plus the [bonus] extra)
+-- uv.lock                    (128 pinned packages)
+-- .gitignore
+-- README.md
```

---

## Mapping to the rubric

| Rubric section | Notebook location |
|---|---|
| 1. Problem Definition | Section 1 |
| 2. EDA, 3 visualizations | Sections 2.1 (target), 2.2 (features), 2.3 (correlation) |
| 3. Preprocessing Pipeline | Section 3: `ColumnTransformer` plus `StandardScaler` inside `Pipeline` |
| 4. Train / Test Split | Section 4: 80/20, `random_state=42`, stratified on `expensive` |
| 5. Linear Regression | Section 5: R-squared, MAE, RMSE, coefficient plot, predicted-vs-actual scatter |
| 6. Logistic Regression L2 | Section 6: full metrics, confusion matrix, ROC curve, weight comparison |
| 7. Discussion | Section 7: outliers, multicollinearity, overfitting |
| Bonus (rubric section 3) | Bonus 1 Polars LazyFrame, Bonus 2 Normal Equation, Bonus 3 JIT Benchmark |

---

## AI Usage Disclosure

Assistant used: **Gemini (Google)**.

| Task | Type of assistance |
|---|---|
| Initial structure of the seven required notebook sections | Boilerplate templates for the markdown headers and section ordering |
| Syntax for unfamiliar tools (JAX, Optax, Polars `scan_csv`) | Code templates for `jax.grad`, `optax.adam`, `@jax.jit`, and lazy-frame method chaining |
| Code formatting and PEP 8 compliance checks | Pattern suggestions for variable naming and line breaks |

**Own work (no AI assistance):**

- Dataset selection and design of both targets (continuous regression target plus binary derived from a median split).
- Decision to apply L2 regularization, and the honest comparison against a logistic regression without regularization to evidence its compressive effect on the same output space.
- Interpretation of linear regression coefficients, including the multicollinearity diagnosis for `AveRooms` (negative sign despite a positive marginal correlation with the target).
- Entire Discussion section: outlier and top-coding treatment, multicollinearity remediation, and hyperparameter-selection considerations.
- Bonus implementation and numerical validation: JAX from-scratch gradient descent with Adam, Normal Equation closed-form solution, JIT speedup benchmark, and Polars LazyFrame ingestion.
- DevContainer architecture: the `uv pip install --python ds/bin/python` pattern, Docker layer separation for dependency caching, and `VIRTUAL_ENV` exposure so that VS Code auto-detects the venv in the kernel picker.
- All metric values, plots, and quantitative claims reported in the notebook and this README, produced by the author's own execution of the pipeline.
