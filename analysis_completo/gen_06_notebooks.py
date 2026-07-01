"""
Genera los tres notebooks de control de efecto de finca:
  06a_Residualizacion_VitD_Calcio.ipynb
  06b_GLMM_Intercepto_Aleatorio_Finca.ipynb
  06c_Importancia_Condicional_Finca_RF.ipynb
"""
import json

def nb(cells):
    return {
        "cells": cells,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python", "version": "3.10.0"},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }

def md(cid, text):
    return {
        "cell_type": "markdown",
        "id": cid,
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }

def code(cid, src):
    return {
        "cell_type": "code",
        "execution_count": None,
        "id": cid,
        "metadata": {},
        "outputs": [],
        "source": src.splitlines(keepends=True),
    }

def save(cells, path):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb(cells), f, indent=1, ensure_ascii=False)
    print(f"  -> {path}  ({len(cells)} celdas)")


# ═══════════════════════════════════════════════════════════════════════════
# NOTEBOOK 06a — Residualizacion de VitD y Calcio por finca
# ═══════════════════════════════════════════════════════════════════════════

A = []

A.append(md("md_title", """# Notebook 06a — Residualizacion de VitD y Calcio por finca
### Tuberculosis bovina · Target: Lesiones_TB

**Pregunta:** Un animal con *mas VitD que la media de su propia finca* ¿tiene menor
probabilidad de TB? ¿Existe un efecto individual de VitD/Calcio una vez que se elimina
la senal de "a que finca pertenece"?

**Estrategia — residualizacion intra-finca:**

```
VitD_res   = VitD   - media(VitD   | finca)
Calcio_res = Calcio - media(Calcio | finca)
```

Los residuales son ortogonales a la media de finca: el RF no puede usar la finca
de forma implicita a traves de VitD/Calcio.
Se conserva toda la N (n=103 tras `drop_sparse_rows`).

**Preprocesamiento:** identico al Nb 02c.
- `drop_sparse_rows`: elimina filas con >4 NaN en MODEL_FEATURES.
- Los 1-2 NaN restantes se imputan (mediana) dentro del pipeline CV.
"""))

A.append(code("cell_setup", """\
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             brier_score_loss, matthews_corrcoef, recall_score)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import shap
import tb_utils as tb

PALETTE = tb.set_plot_style()
tb.set_seeds(tb.SEED)

DATA = "../BD.csv"
df   = tb.clean(tb.load_raw(DATA))

# --- Identico al Nb 02c ---
d = tb.target_subset(df, "Lesiones_TB")
d = tb.drop_sparse_rows(d)          # elimina filas con >4 NaN; n=103
y = d["Lesiones_TB"].astype(int).values
g = d[tb.GROUP_COL].values
EXPLOTS = sorted(d[tb.GROUP_COL].unique())
EXPL_PAL = dict(zip(EXPLOTS, PALETTE[:len(EXPLOTS)]))

print(f"n={len(y)}, prevalencia={y.mean():.3f}")
print(f"Explotaciones: {EXPLOTS}")
"""))

A.append(md("md_resid", "## 2. Residualizacion intra-finca de VitD y Calcio"))

A.append(code("cell_residualize", """\
# Calcular medias de finca (nanmean, ignorando NaN) y residuales
for feat in ["VITAMINA_D", "CALCIO"]:
    farm_mean = d.groupby(tb.GROUP_COL)[feat].transform("mean")
    d[feat + "_res"] = d[feat] - farm_mean

# Descriptivos de medias de finca y residuales
print("Medias de finca:")
print(d.groupby(tb.GROUP_COL)[["VITAMINA_D", "CALCIO"]].mean().round(3))
print()

print("Residuales — estadisticos globales (media ~0):")
for feat_res in ["VITAMINA_D_res", "CALCIO_res"]:
    mu  = round(float(d[feat_res].mean()), 4)
    lo  = round(float(d[feat_res].min()),  2)
    hi  = round(float(d[feat_res].max()),  2)
    nan = int(d[feat_res].isna().sum())
    print(f"  {feat_res}: media={mu}, rango=[{lo}, {hi}], NaN={nan}")

print()
print("Media de residuales por finca (debe ser ~0 en cada una):")
for feat_res in ["VITAMINA_D_res", "CALCIO_res"]:
    fm = d.groupby(tb.GROUP_COL)[feat_res].mean().round(4)
    print(f"  {feat_res}: {fm.to_dict()}")

print()
print("NaN en residuales son heredados de NaN en la variable original.")
print("Se imputaran con mediana dentro del pipeline CV (igual que en 02c).")
"""))

A.append(code("cell_resid_plot", """\
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

pairs = [("VITAMINA_D", "VITAMINA_D_res", "Vitamina D"),
         ("CALCIO",     "CALCIO_res",     "Calcio")]

for row_i, (feat_raw, feat_res, pretty) in enumerate(pairs):
    # Variable cruda
    ax = axes[row_i][0]
    sns.boxplot(data=d, x=tb.GROUP_COL, y=feat_raw, order=EXPLOTS,
                palette=EXPL_PAL, ax=ax)
    sns.stripplot(data=d, x=tb.GROUP_COL, y=feat_raw, order=EXPLOTS,
                  color="black", alpha=0.3, size=3, jitter=True, ax=ax)
    ax.set_title(f"{pretty} cruda (incluye efecto de finca)")
    ax.set_xlabel("Explotacion"); ax.set_ylabel(feat_raw)

    # Residual intra-finca, coloreado por estado TB
    ax2 = axes[row_i][1]
    hue_vals = d["Lesiones_TB"].map({0: "TB-", 1: "TB+"})
    sns.boxplot(data=d, x=tb.GROUP_COL, y=feat_res, order=EXPLOTS,
                palette=EXPL_PAL, ax=ax2, boxprops=dict(alpha=0.4))
    sns.stripplot(data=d, x=tb.GROUP_COL, y=feat_res, order=EXPLOTS,
                  hue=hue_vals,
                  palette={"TB-": PALETTE[2], "TB+": PALETTE[0]},
                  alpha=0.65, size=4, jitter=True, ax=ax2)
    ax2.axhline(0, color="black", lw=0.9, ls="--")
    ax2.set_title(f"{pretty} residual intra-finca\n(puntos: TB+/TB-)")
    ax2.set_xlabel("Explotacion"); ax2.set_ylabel(feat_res)
    if row_i == 0:
        ax2.legend(title="Estado TB", fontsize=8, loc="upper right")
    else:
        ax2.get_legend().remove() if ax2.get_legend() else None

plt.suptitle("VitD y Calcio: crudos vs residuales intra-finca", fontsize=12)
plt.tight_layout()
plt.savefig("figures/fig_06a_residuales.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

A.append(md("md_rf", """\
## 3. Random Forest con residuales intra-finca (CV por animal, 5x10)

Se sustituyen VITAMINA_D y CALCIO por sus residuales. El resto de features
(EDAD, serologia, RAZA2) se mantienen sin cambio.
La variable de explotacion **no entra en el modelo en ninguna forma**.
"""))

A.append(code("cell_rf_setup", """\
# Features del modelo residualizado
MODEL_RES = [
    "VITAMINA_D_res", "CALCIO_res",
    "PIROPLASMA_Q_log", "EDAD", "PIROPLASMA", "THEILERIA", "ANAPLASMA", "RAZA2",
]
NUM_RES = ["VITAMINA_D_res", "CALCIO_res",
           "PIROPLASMA_Q_log", "EDAD", "PIROPLASMA", "THEILERIA", "ANAPLASMA"]
CAT_RES = ["RAZA2"]

X_res = d[MODEL_RES].copy()

def make_rf(n_estimators=100, min_samples_leaf=5):
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=None,
        min_samples_leaf=min_samples_leaf, max_features="sqrt",
        class_weight="balanced", random_state=tb.SEED, n_jobs=-1, oob_score=True,
    )

def make_pipe_res(feature_list, n_estimators=100):
    num_f = [f for f in feature_list if f in NUM_RES]
    cat_f = [f for f in feature_list if f in CAT_RES]
    prep  = tb.make_preprocessor(numeric=num_f, categorical=cat_f, scale=False)
    return Pipeline([("prep", prep), ("clf", make_rf(n_estimators=n_estimators))])

CV_OUTER = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=tb.SEED)
CV_SEL   = RepeatedStratifiedKFold(n_splits=5, n_repeats=3,  random_state=tb.SEED)

def cv_metrics(feature_list, X, y, cv, n_estimators=100, detailed=False):
    Xs = X[list(feature_list)]
    pipe = make_pipe_res(feature_list, n_estimators)
    rows = []
    for tr, te in cv.split(Xs, y):
        if len(np.unique(y[te])) < 2: continue
        pf = clone(pipe); pf.fit(Xs.iloc[tr], y[tr])
        p  = pf.predict_proba(Xs.iloc[te])[:, 1]
        pr = (p >= 0.5).astype(int)
        rows.append(dict(
            prauc=average_precision_score(y[te], p),
            roc=roc_auc_score(y[te], p),
            brier=brier_score_loss(y[te], p),
            mcc=matthews_corrcoef(y[te], pr),
            sens=recall_score(y[te], pr, pos_label=1, zero_division=0),
            spec=recall_score(y[te], pr, pos_label=0, zero_division=0),
        ))
    df_r = pd.DataFrame(rows)
    return df_r if detailed else (df_r.prauc.mean(), df_r.prauc.std())

base_m, base_s = cv_metrics(MODEL_RES, X_res, y, CV_SEL)
print(f"Baseline RF residualizado: PR-AUC = {base_m:.3f} +/- {base_s:.3f}")
print(f"Linea base prevalencia: {y.mean():.3f}")
"""))

A.append(code("cell_feature_sel", """\
# Importancia de permutacion -> orden de eliminacion
full_pipe = make_pipe_res(MODEL_RES).fit(X_res, y)
pi = permutation_importance(full_pipe, X_res, y, scoring="average_precision",
                             n_repeats=50, random_state=tb.SEED, n_jobs=-1)
imp_df = pd.DataFrame({
    "feature": MODEL_RES,
    "imp_mean": pi.importances_mean,
    "imp_std":  pi.importances_std,
}).sort_values("imp_mean", ascending=False).reset_index(drop=True)

fig, ax = plt.subplots(figsize=(8, 4))
order = imp_df.sort_values("imp_mean")
ax.barh(order.feature, order.imp_mean, xerr=order.imp_std,
        color=PALETTE[0], capsize=3, alpha=0.85)
ax.axvline(0, color="k", lw=0.8)
ax.set_xlabel("Delta PR-AUC por permutacion")
ax.set_title("Importancia de permutacion — RF residualizado")
plt.tight_layout()
plt.savefig("figures/fig_06a_perm_imp_full.png", dpi=150, bbox_inches="tight")
plt.show()

# Eliminacion hacia atras
elim_order = imp_df.sort_values("imp_mean", ascending=True)["feature"].tolist()
current = MODEL_RES.copy(); history = []
m0, s0 = cv_metrics(current, X_res, y, CV_SEL)
history.append({"n": len(current), "features": current.copy(),
                "removed": "baseline", "mean": m0, "std": s0})
print(f"  {len(current):2d} features [baseline] PR-AUC={m0:.3f}+/-{s0:.3f}")

for feat in elim_order:
    if feat not in current or len(current) <= 1: break
    test_f = [f for f in current if f != feat]
    m, s = cv_metrics(test_f, X_res, y, CV_SEL)
    delta = m - m0
    history.append({"n": len(test_f), "features": test_f.copy(),
                    "removed": feat, "mean": m, "std": s})
    print(f"  {len(test_f):2d} features [-{feat:25s}] PR-AUC={m:.3f}+/-{s:.3f}  D={delta:+.3f}")
    current = test_f

hist_df = pd.DataFrame(history)
best_mu  = hist_df["mean"].max()
best_sd  = hist_df.loc[hist_df["mean"].idxmax(), "std"]
threshold = best_mu - best_sd
parsimonious = hist_df[hist_df["mean"] >= threshold].sort_values("n").iloc[0]
SELECTED = list(parsimonious.features)
print(f"\nFeatures seleccionados ({len(SELECTED)}): {SELECTED}")
"""))

A.append(code("cell_eval", """\
Xs = X_res[SELECTED].copy()
det = cv_metrics(SELECTED, X_res, y, CV_OUTER, n_estimators=700, detailed=True)
m = det.mean(); s = det.std()

print("Metricas CV por animal (5x10, RF residualizado):")
print(f"  PR-AUC = {m.prauc:.3f} +/- {s.prauc:.3f}   (base = {y.mean():.3f})")
print(f"  ROC    = {m.roc:.3f}   +/- {s.roc:.3f}")
print(f"  Brier  = {m.brier:.3f}  +/- {s.brier:.3f}")
print(f"  MCC    = {m.mcc:.3f}   +/- {s.mcc:.3f}")
print(f"  Sens   = {m.sens:.2f}   +/- {s.sens:.2f}")
print(f"  Spec   = {m.spec:.2f}   +/- {s.spec:.2f}")

fig, axes = plt.subplots(1, 3, figsize=(13, 4))
for ax, col, title in zip(axes, ["prauc", "roc", "mcc"], ["PR-AUC", "ROC-AUC", "MCC"]):
    ax.hist(det[col].dropna(), bins=15, color=PALETTE[0], alpha=0.8, edgecolor="white")
    ax.axvline(det[col].mean(), color=PALETTE[1], lw=2, label=f"Media={det[col].mean():.3f}")
    if col == "prauc":
        ax.axvline(y.mean(), color="grey", ls="--", lw=1.5, label=f"Base={y.mean():.3f}")
    ax.set_xlabel(title); ax.set_title(f"Dist. {title}"); ax.legend(fontsize=8)
plt.suptitle("RF residualizado — Metricas CV por animal (5x10)", y=1.02)
plt.tight_layout()
plt.savefig("figures/fig_06a_metricas.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

A.append(code("cell_shap", """\
FINAL = make_pipe_res(SELECTED, n_estimators=1000).fit(Xs, y)
prep  = FINAL.named_steps["prep"]
clf   = FINAL.named_steps["clf"]
Xt    = prep.transform(Xs)
names = list(prep.get_feature_names_out())

explainer = shap.TreeExplainer(clf)
sv_raw    = explainer.shap_values(Xt)
if isinstance(sv_raw, list):
    sv = sv_raw[1]
elif np.asarray(sv_raw).ndim == 3:
    sv = np.asarray(sv_raw)[:, :, 1]
else:
    sv = np.asarray(sv_raw)
exp_val = (float(explainer.expected_value[1])
           if isinstance(explainer.expected_value, (list, np.ndarray))
           else float(explainer.expected_value))

fig, axes = plt.subplots(1, 2, figsize=(14, max(4, 0.5 * len(names))))
plt.sca(axes[0])
shap.summary_plot(sv, Xt, feature_names=names, show=False, max_display=len(names))
axes[0].set_title("SHAP summary — RF residualizado")

mean_abs = np.abs(sv).mean(0)
si = pd.DataFrame({"feature": names, "mean_abs": mean_abs}).sort_values("mean_abs")
axes[1].barh(si.feature, si.mean_abs, color=PALETTE[0], alpha=0.85)
axes[1].set_xlabel("Media |SHAP|")
axes[1].set_title("Importancia SHAP (RF residualizado)")
plt.tight_layout()
plt.savefig("figures/fig_06a_shap.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

A.append(code("cell_ale", """\
# ALE para los residuales seleccionados
CONT = [f for f in SELECTED if f in NUM_RES and Xs[f].nunique() > 5]
if not CONT:
    CONT = [f for f in SELECTED if f in NUM_RES]

def ale_1d(model, X, feat, bins=10):
    x = X[feat].dropna().values
    q = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(q) < 3: return np.array([]), np.array([])
    eff, Xc = [], X.copy()
    for i in range(len(q) - 1):
        mask = (X[feat] >= q[i]) & (X[feat] <= q[i + 1])
        if not mask.any(): eff.append(0.0); continue
        lo = Xc.loc[mask].copy(); lo[feat] = q[i]
        hi = Xc.loc[mask].copy(); hi[feat] = q[i + 1]
        eff.append((model.predict_proba(hi)[:, 1]
                    - model.predict_proba(lo)[:, 1]).mean())
    ale = np.cumsum(eff); ale -= ale.mean()
    return (q[:-1] + q[1:]) / 2, ale

if CONT:
    n_c = len(CONT)
    fig, axes_ale = plt.subplots(1, n_c, figsize=(5 * n_c, 4))
    if n_c == 1: axes_ale = [axes_ale]
    for ax, feat in zip(axes_ale, CONT):
        cx, ale_v = ale_1d(FINAL, Xs, feat)
        if not len(cx): continue
        ax.plot(cx, ale_v, "o-", color=PALETTE[1], lw=2, ms=5)
        ax.axhline(0, color="grey", lw=0.7, ls="--")
        ax.fill_between(cx, 0, ale_v, where=(ale_v > 0),
                        alpha=0.18, color=PALETTE[0], label="mayor riesgo TB")
        ax.fill_between(cx, 0, ale_v, where=(ale_v < 0),
                        alpha=0.18, color=PALETTE[1], label="menor riesgo TB")
        ax.set_title(f"ALE: {feat}")
        ax.set_xlabel(feat + "\\n(residual intra-finca)")
        ax.legend(fontsize=8)
    plt.suptitle("ALE — efecto intra-finca de VitD/Calcio residual", y=1.02)
    plt.tight_layout()
    plt.savefig("figures/fig_06a_ale.png", dpi=150, bbox_inches="tight")
    plt.show()
"""))

A.append(md("md_synthesis", """\
## 8. Interpretacion comparativa

| Resultado | Interpretacion |
|---|---|
| PR-AUC(06a) ~ PR-AUC(02c) | La senal de VitD/Calcio es individual — no es la finca disfrazada |
| PR-AUC(06a) << PR-AUC(02c) | La mayor parte de la senal en 02c era efecto ecologico (finca) |
| ALE(VitD_res) decreciente | Mas VitD que la media de la finca → menos TB (efecto intra-finca) |
| ALE(VitD_res) plano | No hay efecto individual de VitD una vez controlada la finca |

> Comparar PR-AUC(06a) con Nb 02c (VitD/Calcio crudos, sin control de finca)
> y con Nb 03c (VitD/Calcio crudos + finca como feature explicita).
"""))

save(A, "06a_Residualizacion_VitD_Calcio.ipynb")


# ═══════════════════════════════════════════════════════════════════════════
# NOTEBOOK 06b — GLMM con intercepto aleatorio por finca
# ═══════════════════════════════════════════════════════════════════════════

B = []

B.append(md("md_title", """# Notebook 06b — Modelo mixto (GLMM) con intercepto aleatorio por finca
### Tuberculosis bovina · Target: Lesiones_TB

**Pregunta:** ¿Cuanto es el efecto de VitD y Calcio sobre TB *ajustado* por la finca,
estimado directamente como efecto fijo en un modelo estadistico inferencial?

**Estrategia:** modelo logistico con **finca como efecto aleatorio** (intercepto
aleatorio por finca) + VitD, Calcio, EDAD como efectos fijos:

```
logit(P(TB)) = beta0 + beta_VitD * VitD + beta_Calcio * Calcio + beta_Edad * Edad
               + u_finca    (u_finca ~ N(0, sigma^2_finca))
```

El intercepto aleatorio `u_finca` absorbe toda la heterogeneidad entre fincas,
de modo que `beta_VitD` mide el efecto *dentro de la finca* (separado del entre-fincas).

**Preprocesamiento:** identico al Nb 02c (drop_sparse_rows + caso completo para GLMM).
"""))

B.append(code("cell_setup", """\
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import statsmodels.formula.api as smf
from statsmodels.genmod.generalized_estimating_equations import GEE
from statsmodels.genmod.families import Binomial
from statsmodels.genmod.cov_struct import Exchangeable, Independence
from statsmodels.genmod.bayes_mixed_glm import BinomialBayesMixedGLM
import tb_utils as tb

PALETTE = tb.set_plot_style()
tb.set_seeds(tb.SEED)

DATA = "../BD.csv"
df   = tb.clean(tb.load_raw(DATA))

# --- Identico al Nb 02c ---
d = tb.target_subset(df, "Lesiones_TB")
d = tb.drop_sparse_rows(d)          # n=103
y = d["Lesiones_TB"].astype(int).values
g = d[tb.GROUP_COL].values
EXPLOTS = sorted(d[tb.GROUP_COL].unique())

print(f"n={len(y)}, prevalencia={y.mean():.3f}")
print(f"Explotaciones: {EXPLOTS}")
print(f"n por explotacion: {pd.Series(g).value_counts().to_dict()}")
"""))

B.append(md("md_prep", """\
## 2. Preparacion de la tabla de datos para modelos estadisticos

Los modelos de statsmodels requieren datos completos. Se eliminan las filas
con NaN en las variables predictoras (VitD, Calcio, EDAD).

Las variables continuas se **estandarizan** (z-score) para que los coeficientes
sean comparables entre predictores (beta = cambio en log-OR por 1 SD).
"""))

B.append(code("cell_prep_glm", """\
# Variables de interes
FIXED = ["VITAMINA_D", "CALCIO", "EDAD"]
TARGET = "Lesiones_TB"
GROUP  = tb.GROUP_COL   # "Expl"

# Caso completo (eliminar NaN en predictores fijos)
cols_needed = [TARGET, GROUP] + FIXED
d_glm = d[cols_needed].dropna().copy().reset_index(drop=True)

n_drop = len(d) - len(d_glm)
print(f"n antes de eliminar NaN en predictores: {len(d)}")
print(f"Eliminados por NaN:                    {n_drop}")
print(f"n para modelos estadisticos:            {len(d_glm)}")
print(f"Prevalencia TB en subconjunto:          {d_glm[TARGET].mean():.3f}")

# Estandarizar predictores continuos (z-score)
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
d_std = d_glm.copy()
d_std[["VitD_z", "Calcio_z", "Edad_z"]] = scaler.fit_transform(d_glm[FIXED])

means = dict(zip(FIXED, scaler.mean_))
stds  = dict(zip(FIXED, scaler.scale_))
print()
print("Estandarizacion (media +/- SD):")
for v in FIXED:
    print(f"  {v}: media={means[v]:.3f}, SD={stds[v]:.3f}")
"""))

B.append(md("md_models", """\
## 3. Modelos estadisticos: del mas simple al mas completo

Se ajustan cuatro modelos para comparar como cambia el efecto de VitD/Calcio
al controlar progresivamente por la finca.

| Modelo | Control de finca | Estimando |
|---|---|---|
| M0 — Logit nulo | Ninguno | Efecto bruto de VitD/Calcio (puede confundirse con finca) |
| M1 — Logit + dummies | Efectos fijos (dummies de finca) | Efecto ajustado por finca (parametrico) |
| M2 — GEE | Correlacion de intercambiabilidad | Efecto poblacional (marginal) |
| M3 — GLMM | Intercepto aleatorio por finca | Efecto condicional a la finca (inferencia) |
"""))

B.append(code("cell_models", """\
# M0: Logit sin control de finca (referencia)
m0 = smf.logit("Lesiones_TB ~ VitD_z + Calcio_z + Edad_z", data=d_std).fit(disp=0)
print("M0 — Logit sin control de finca")
print(m0.summary2().tables[1][["Coef.", "Std.Err.", "z", "P>|z|"]].round(3))
print()

# M1: Logit con dummies de finca (efectos fijos)
m1 = smf.logit("Lesiones_TB ~ VitD_z + Calcio_z + Edad_z + C(Expl)",
               data=d_std).fit(disp=0)
print("M1 — Logit con efectos fijos de finca")
coef_m1 = m1.summary2().tables[1][["Coef.", "Std.Err.", "z", "P>|z|"]].round(3)
print(coef_m1[~coef_m1.index.str.startswith("C(Expl)")])
print()

# M2: GEE (efecto marginal/poblacional)
m2 = GEE.from_formula(
    "Lesiones_TB ~ VitD_z + Calcio_z + Edad_z",
    groups=GROUP,
    data=d_std,
    family=Binomial(),
    cov_struct=Exchangeable(),
).fit(disp=False)
print("M2 — GEE (efecto marginal, intercambiabilidad)")
print(m2.summary().tables[1])
print()

# M3: GLMM (intercepto aleatorio por finca — Bayesian MAP)
try:
    m3 = BinomialBayesMixedGLM.from_formula(
        "Lesiones_TB ~ VitD_z + Calcio_z + Edad_z",
        {"Finca": "0 + C(Expl)"},
        data=d_std,
    )
    r3 = m3.fit_map()
    print("M3 — GLMM (intercepto aleatorio por finca, MAP)")
    print(r3.summary())
    HAS_GLMM = True
except Exception as e:
    print(f"GLMM no disponible: {e}")
    HAS_GLMM = False
"""))

B.append(code("cell_or_table", """\
# Tabla de Odds Ratios comparada entre modelos
def or_ci(model, varnames):
    rows = []
    for v in varnames:
        coef = model.params[v]
        ci   = model.conf_int().loc[v]
        rows.append({
            "Variable": v,
            "OR": round(float(np.exp(coef)), 3),
            "IC95_lo": round(float(np.exp(ci.iloc[0])), 3),
            "IC95_hi": round(float(np.exp(ci.iloc[1])), 3),
            "p":       round(float(model.pvalues[v]), 4),
        })
    return pd.DataFrame(rows)

var_show = ["VitD_z", "Calcio_z", "Edad_z"]
or_m0 = or_ci(m0, var_show); or_m0["Modelo"] = "M0-Logit"
or_m1 = or_ci(m1, var_show); or_m1["Modelo"] = "M1-Logit+FE"
or_m2 = or_ci(m2, var_show); or_m2["Modelo"] = "M2-GEE"

frames = [or_m0, or_m1, or_m2]
labels = ["M0-Logit", "M1-Logit+FE", "M2-GEE"]
colors_m = [PALETTE[0], PALETTE[1], PALETTE[2]]

if HAS_GLMM:
    fep_names = r3.model.fep_names
    nfep = len(fep_names)
    # SEs desde la covarianza del MAP (diagonal de cov_params)
    try:
        full_cov = r3.cov_params()
        fe_se = np.sqrt(np.diag(full_cov)[:nfep])
    except Exception:
        try:
            fe_se = r3.bse[:nfep]
        except Exception:
            fe_se = np.full(nfep, np.nan)
    from scipy.stats import norm as _norm
    r3_rows = []
    for i, v in enumerate(fep_names):
        if v in var_show:
            coef = float(r3.fe_mean[i])
            se   = float(fe_se[i])
            lo   = coef - 1.96 * se
            hi   = coef + 1.96 * se
            pval = 2 * (1 - _norm.cdf(abs(coef / se))) if se > 0 else np.nan
            r3_rows.append({"Variable": v, "OR": round(np.exp(coef), 3),
                             "IC95_lo": round(np.exp(lo), 3),
                             "IC95_hi": round(np.exp(hi), 3), "p": round(pval, 4)})
    or_m3 = pd.DataFrame(r3_rows); or_m3["Modelo"] = "M3-GLMM"
    frames.append(or_m3); labels.append("M3-GLMM"); colors_m.append(PALETTE[3])

all_or = pd.concat(frames, ignore_index=True)
print("Odds Ratios por modelo (1 SD de cambio en la variable):")
display(all_or.pivot_table(index="Variable", columns="Modelo",
                            values="OR").round(3))

# Forest plot
fig, ax = plt.subplots(figsize=(9, 5))
n_models = len(frames)
y_pos = np.arange(len(var_show))
offsets = np.linspace(-0.25, 0.25, n_models)

for j, (df_or, lbl, col) in enumerate(zip(frames, labels, colors_m)):
    for i, v in enumerate(var_show):
        row_v = df_or[df_or["Variable"] == v]
        if row_v.empty: continue
        r = row_v.iloc[0]
        ax.scatter(r.OR, y_pos[i] + offsets[j], color=col, s=60, zorder=5)
        ax.errorbar(r.OR, y_pos[i] + offsets[j],
                    xerr=[[r.OR - r.IC95_lo], [r.IC95_hi - r.OR]],
                    fmt="none", color=col, capsize=4, lw=1.5)

ax.axvline(1, color="grey", lw=0.9, ls="--")
ax.set_yticks(y_pos); ax.set_yticklabels(var_show)
ax.set_xlabel("Odds Ratio (IC 95%)")
ax.set_title("Forest plot de OR: VitD/Calcio/Edad sobre TB\npor modelo (por 1 SD)")
legend_patches = [mpatches.Patch(color=c, label=l)
                  for c, l in zip(colors_m, labels)]
ax.legend(handles=legend_patches, fontsize=9, loc="lower right")
plt.tight_layout()
plt.savefig("figures/fig_06b_forest_or.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

B.append(code("cell_icc", """\
# Intraclass Correlation Coefficient (ICC): varianza explicada por finca
# ICC = sigma2_finca / (sigma2_finca + pi^2/3)
# donde pi^2/3 ~ 3.2899 es la varianza del logistico estandar

# M1 (efectos fijos): ICC implicito por varianza de los coef. de finca
coef_finca = [v for v in m1.params.index if "C(Expl)" in v]
var_finca_m1 = m1.params[coef_finca].var()
icc_m1 = var_finca_m1 / (var_finca_m1 + np.pi**2 / 3)
print(f"ICC aproximada (M1 efectos fijos): {icc_m1:.3f}")

# GEE: alpha de intercambiabilidad ~ correlacion intra-finca
try:
    alpha_gee = m2.cov_struct.dep_params
    print(f"GEE correlacion intra-finca (alpha): {alpha_gee:.3f}")
except Exception:
    pass

if HAS_GLMM:
    # M3: varianza del intercepto aleatorio
    vc_names = r3.model.vcp_names
    vc_mean  = r3.vcp_mean
    sigma2_finca = np.exp(vc_mean[0]) if len(vc_mean) > 0 else np.nan
    icc_glmm = sigma2_finca / (sigma2_finca + np.pi**2 / 3)
    print(f"ICC (M3 GLMM, intercepto aleatorio): {icc_glmm:.3f}")
    print(f"  sigma2_finca = {sigma2_finca:.3f}")
    print()
    print("Interpretacion ICC:")
    print("  ICC > 0.10: la finca explica >10% de la variabilidad en TB")
    print("  -> controlar por finca es necesario para estimar efectos individuales")

# Interceptos aleatorios de finca
print()
print("Medias de TB por finca (cruda vs ajustada M1):")
obs = d_glm.groupby(GROUP)[TARGET].agg(["mean", "count"]).round(3)
obs.columns = ["prev_obs", "n"]
pred_m1 = pd.Series(m1.predict(d_std), index=d_std.index)
pred_by_farm = d_std.assign(pred=pred_m1).groupby(GROUP)["pred"].mean().round(3)
print(pd.concat([obs, pred_by_farm.rename("prev_ajustada_M1")], axis=1))
"""))

B.append(md("md_synthesis", """\
## 6. Interpretacion

### Clave de lectura

| Observacion | Interpretacion |
|---|---|
| OR(VitD_z, M3) < 1 y sig. | VitD protege contra TB *dentro de la finca* (efecto individual) |
| OR(VitD_z) cambia entre M0 y M3 | La finca confunde el efecto de VitD (cambio de confusor) |
| ICC alto (>0.10) | Gran variabilidad entre fincas -> necesario controlar por finca |
| ICC bajo (<0.05) | La finca explica poca varianza -> el efecto no es finca-dependiente |

### Comparacion de los modelos

- **M0 → M1 → M3**: si el OR de VitD cambia mucho, la finca confundia el efecto.
- **M2 (GEE)**: da el efecto marginal (poblacional); puede diferir de M3 si hay
  interaccion entre finca e individuos.
- **M3 (GLMM)**: el mas apropiado para separar "efecto de finca" de "efecto individual".

> **Limitacion:** con n=103 y solo 4 fincas, los parametros del intercepto aleatorio
> tienen alta incertidumbre. El ICC y los efectos aleatorios son estimaciones exploratorias.
"""))

save(B, "06b_GLMM_Intercepto_Aleatorio_Finca.ipynb")


# ═══════════════════════════════════════════════════════════════════════════
# NOTEBOOK 06c — Importancia condicional de VitD/Calcio dentro de cada finca
# ═══════════════════════════════════════════════════════════════════════════

C = []

C.append(md("md_title", """# Notebook 06c — Importancia condicional de VitD/Calcio dentro de cada finca (RF)
### Tuberculosis bovina · Target: Lesiones_TB

**Pregunta:** En un RF entrenado globalmente, ¿aportan VitD/Calcio poder predictivo
*dentro de cada finca por separado*? O dicho de otro modo: si permuto VitD solo
dentro de los animales de la finca VC30D, ¿sube el error de prediccion?

Si la importancia de VitD es alta en todas las fincas → efecto individual real.
Si es alta globalmente pero baja dentro de cada finca → VitD es un proxy de finca.

**Estrategia:**
1. RF global entrenado sin la variable finca.
2. Importancia de permutacion global (referencia).
3. Importancia de permutacion calculada *dentro de cada finca* por separado.
4. SHAP global y SHAP agrupado por finca (comparacion de distribuciones).

**Preprocesamiento:** identico al Nb 02c.
"""))

C.append(code("cell_setup", """\
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold, StratifiedKFold
from sklearn.metrics import (average_precision_score, roc_auc_score,
                             brier_score_loss, matthews_corrcoef, recall_score)
from sklearn.inspection import permutation_importance, PartialDependenceDisplay
import shap
import tb_utils as tb

PALETTE = tb.set_plot_style()
tb.set_seeds(tb.SEED)

DATA = "../BD.csv"
df   = tb.clean(tb.load_raw(DATA))

# --- Identico al Nb 02c ---
d = tb.target_subset(df, "Lesiones_TB")
d = tb.drop_sparse_rows(d)
y = d["Lesiones_TB"].astype(int).values
g = d[tb.GROUP_COL].values
EXPLOTS = sorted(d[tb.GROUP_COL].unique())
EXPL_PAL = dict(zip(EXPLOTS, PALETTE[:len(EXPLOTS)]))

X_all = d[tb.MODEL_FEATURES].copy()
print(f"n={len(y)}, prevalencia={y.mean():.3f}")
print(f"Features: {tb.MODEL_FEATURES}")
print(f"n por finca: {pd.Series(g).value_counts().to_dict()}")
"""))

C.append(md("md_global_rf", "## 2. RF global (sin variable de finca) y seleccion de features"))

C.append(code("cell_global_rf", """\
def make_rf(n_estimators=100, min_samples_leaf=5):
    return RandomForestClassifier(
        n_estimators=n_estimators, max_depth=None,
        min_samples_leaf=min_samples_leaf, max_features="sqrt",
        class_weight="balanced", random_state=tb.SEED, n_jobs=-1, oob_score=True,
    )

def make_pipe(feature_list, n_estimators=100):
    num_f = [f for f in feature_list if f in tb.NUM_FEATURES]
    cat_f = [f for f in feature_list if f in tb.CAT_FEATURES]
    prep  = tb.make_preprocessor(numeric=num_f, categorical=cat_f, scale=False)
    return Pipeline([("prep", prep), ("clf", make_rf(n_estimators=n_estimators))])

CV_OUTER = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=tb.SEED)
CV_SEL   = RepeatedStratifiedKFold(n_splits=5, n_repeats=3,  random_state=tb.SEED)

def cv_metrics(feature_list, X, y, cv, n_estimators=100, detailed=False):
    Xs = X[list(feature_list)]
    pipe = make_pipe(feature_list, n_estimators)
    rows = []
    for tr, te in cv.split(Xs, y):
        if len(np.unique(y[te])) < 2: continue
        pf = clone(pipe); pf.fit(Xs.iloc[tr], y[tr])
        p  = pf.predict_proba(Xs.iloc[te])[:, 1]
        pr = (p >= 0.5).astype(int)
        rows.append(dict(
            prauc=average_precision_score(y[te], p),
            roc=roc_auc_score(y[te], p),
            mcc=matthews_corrcoef(y[te], pr),
        ))
    df_r = pd.DataFrame(rows)
    return df_r if detailed else (df_r.prauc.mean(), df_r.prauc.std())

# Modelo completo como base
full_pipe = make_pipe(tb.MODEL_FEATURES).fit(X_all, y)
print(f"OOB Score RF global: {full_pipe.named_steps['clf'].oob_score_:.3f}")

# Importancia de permutacion global (todas las features)
pi_global = permutation_importance(full_pipe, X_all, y, scoring="average_precision",
                                    n_repeats=50, random_state=tb.SEED, n_jobs=-1)
imp_global = pd.DataFrame({
    "feature":   tb.MODEL_FEATURES,
    "imp_mean":  pi_global.importances_mean,
    "imp_std":   pi_global.importances_std,
}).sort_values("imp_mean", ascending=False)

print()
print("Importancia de permutacion global:")
for _, r in imp_global.iterrows():
    print(f"  {r.feature:25s}: {r.imp_mean:+.4f} +/- {r.imp_std:.4f}")

# Seleccion de features (eliminacion hacia atras)
elim_order = imp_global.sort_values("imp_mean", ascending=True)["feature"].tolist()
current = tb.MODEL_FEATURES.copy(); history = []
m0, s0 = cv_metrics(current, X_all, y, CV_SEL)
history.append({"n": len(current), "features": current.copy(),
                "removed": "baseline", "mean": m0, "std": s0})
for feat in elim_order:
    if feat not in current or len(current) <= 1: break
    test_f = [f for f in current if f != feat]
    m, s = cv_metrics(test_f, X_all, y, CV_SEL)
    history.append({"n": len(test_f), "features": test_f.copy(),
                    "removed": feat, "mean": m, "std": s})
    print(f"  {len(test_f):2d} feat [-{feat:20s}] PR-AUC={m:.3f}+/-{s:.3f}")
    current = test_f

hist_df = pd.DataFrame(history)
best_mu   = hist_df["mean"].max()
threshold = best_mu - hist_df.loc[hist_df["mean"].idxmax(), "std"]
parsimonious = hist_df[hist_df["mean"] >= threshold].sort_values("n").iloc[0]
SELECTED = list(parsimonious.features)
print(f"\nFeatures seleccionados: {SELECTED}")

Xs = X_all[SELECTED].copy()
FINAL = make_pipe(SELECTED, n_estimators=1000).fit(Xs, y)
print(f"OOB Score (features seleccionados): {FINAL.named_steps['clf'].oob_score_:.3f}")
"""))

C.append(md("md_cond_imp", """\
## 3. Importancia de permutacion condicional (dentro de cada finca)

Se toma el RF global (entrenado en todos los animales) y se calcula la importancia
de permutacion usando **unicamente los animales de cada finca**. Esto responde:
"¿permutando VitD solo dentro de VC30D, empeora la prediccion en VC30D?"

Si la respuesta es si -> VitD aporta informacion individual dentro de esa finca.
Si la respuesta es no -> VitD no predice TB mas alla del efecto de finca.
"""))

C.append(code("cell_cond_perm_imp", """\
# Importancia condicional por finca
farm_imp = {}
for farm in EXPLOTS:
    mask = g == farm
    n_farm = mask.sum()
    y_farm = y[mask]
    X_farm = Xs[mask].reset_index(drop=True)
    n_pos  = y_farm.sum(); n_neg = n_farm - n_pos
    if n_farm < 5 or n_pos < 2 or n_neg < 2:
        print(f"  {farm}: n={n_farm} (insuficiente para importancia), se omite")
        continue
    try:
        pi_farm = permutation_importance(
            FINAL, X_farm, y_farm,
            scoring="average_precision",
            n_repeats=30,
            random_state=tb.SEED,
        )
        farm_imp[farm] = {
            f: (float(pi_farm.importances_mean[i]),
                float(pi_farm.importances_std[i]))
            for i, f in enumerate(SELECTED)
        }
        print(f"  {farm} (n={n_farm}, TB+={n_pos}):")
        for f, (m, s) in sorted(farm_imp[farm].items(), key=lambda x: -x[1][0]):
            print(f"    {f:25s}: {m:+.4f} +/- {s:.4f}")
    except Exception as e:
        print(f"  {farm}: error - {e}")

# Importancia global para comparacion
pi_sel = permutation_importance(FINAL, Xs, y, scoring="average_precision",
                                 n_repeats=50, random_state=tb.SEED, n_jobs=-1)
global_imp = {f: (float(pi_sel.importances_mean[i]),
                   float(pi_sel.importances_std[i]))
              for i, f in enumerate(SELECTED)}
print(f"\nImportancia GLOBAL (referencia):")
for f, (m, s) in sorted(global_imp.items(), key=lambda x: -x[1][0]):
    print(f"  {f:25s}: {m:+.4f} +/- {s:.4f}")
"""))

C.append(code("cell_heatmap_imp", """\
# Heatmap: importancia por feature x finca (+ columna global)
farms_ok  = list(farm_imp.keys())
feat_order = sorted(global_imp.keys(), key=lambda f: -global_imp[f][0])

data_heat = pd.DataFrame(index=feat_order, columns=farms_ok + ["GLOBAL"])
for feat in feat_order:
    for farm in farms_ok:
        data_heat.loc[feat, farm] = round(farm_imp[farm][feat][0], 4)
    data_heat.loc[feat, "GLOBAL"] = round(global_imp[feat][0], 4)

data_heat = data_heat.astype(float)

fig, ax = plt.subplots(figsize=(max(7, len(farms_ok) + 3), max(4, len(feat_order) * 0.8)))
sns.heatmap(data_heat, annot=True, fmt=".3f", cmap="RdBu_r",
            center=0, linewidths=0.5, ax=ax,
            cbar_kws={"label": "Delta PR-AUC (permutacion)"})
ax.set_title("Importancia de permutacion condicional por finca vs global\n"
             "(positivo = la feature aporta; negativo = sin efecto o ruido)")
ax.set_ylabel("Feature")
ax.set_xlabel("Finca (+ global)")
plt.tight_layout()
plt.savefig("figures/fig_06c_heatmap_imp_condicional.png", dpi=150, bbox_inches="tight")
plt.show()

# Barplot comparativo
fig, ax = plt.subplots(figsize=(9, max(4, len(feat_order) * 0.7)))
y_pos = np.arange(len(feat_order))
width = 0.8 / (len(farms_ok) + 1)
offsets = np.linspace(-0.35, 0.35, len(farms_ok) + 1)
all_cols = list(EXPL_PAL.values())[:len(farms_ok)] + ["black"]
all_labels = farms_ok + ["GLOBAL"]

for j, (farm_j, col_j) in enumerate(zip(all_labels, all_cols)):
    means_j = [data_heat.loc[f, farm_j] for f in feat_order]
    ax.barh(y_pos + offsets[j], means_j, height=width,
            color=col_j, alpha=0.82, label=farm_j)

ax.axvline(0, color="black", lw=0.8)
ax.set_yticks(y_pos); ax.set_yticklabels(feat_order)
ax.set_xlabel("Delta PR-AUC por permutacion")
ax.set_title("Importancia condicional por finca vs global")
ax.legend(fontsize=8)
plt.tight_layout()
plt.savefig("figures/fig_06c_barplot_imp_condicional.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

C.append(md("md_shap", "## 4. SHAP global y agrupado por finca"))

C.append(code("cell_shap_global", """\
prep  = FINAL.named_steps["prep"]
clf   = FINAL.named_steps["clf"]
Xt    = prep.transform(Xs)
names = list(prep.get_feature_names_out())

explainer = shap.TreeExplainer(clf)
sv_raw    = explainer.shap_values(Xt)
if isinstance(sv_raw, list):
    sv = sv_raw[1]
elif np.asarray(sv_raw).ndim == 3:
    sv = np.asarray(sv_raw)[:, :, 1]
else:
    sv = np.asarray(sv_raw)
exp_val = (float(explainer.expected_value[1])
           if isinstance(explainer.expected_value, (list, np.ndarray))
           else float(explainer.expected_value))

plt.figure(figsize=(9, max(4, 0.5 * len(names))))
shap.summary_plot(sv, Xt, feature_names=names, show=False, max_display=len(names))
plt.title("SHAP summary global (RF sin finca)")
plt.tight_layout()
plt.savefig("figures/fig_06c_shap_global.png", dpi=150, bbox_inches="tight")
plt.show()
"""))

C.append(code("cell_shap_by_farm", """\
# Media |SHAP| por feature dentro de cada finca
shap_farm = {}
for farm in EXPLOTS:
    mask_f = g == farm
    shap_farm[farm] = np.abs(sv[mask_f]).mean(axis=0)

shap_heat = pd.DataFrame(
    {farm: shap_farm[farm] for farm in EXPLOTS},
    index=names,
)
shap_heat["GLOBAL"] = np.abs(sv).mean(axis=0)
shap_heat = shap_heat.sort_values("GLOBAL", ascending=False)

fig, ax = plt.subplots(figsize=(max(7, len(EXPLOTS) + 2), max(4, len(names) * 0.8)))
sns.heatmap(shap_heat, annot=True, fmt=".3f", cmap="YlOrRd",
            linewidths=0.5, ax=ax,
            cbar_kws={"label": "Media |SHAP| (clase TB+)"})
ax.set_title("Importancia SHAP por finca y global\n"
             "(mayor valor = feature mas influyente en esa finca)")
ax.set_xlabel("Finca (+ global)"); ax.set_ylabel("Feature")
plt.tight_layout()
plt.savefig("figures/fig_06c_shap_por_finca.png", dpi=150, bbox_inches="tight")
plt.show()

# Violinplot: distribucion de valores SHAP de VitD/Calcio por finca
vitd_idx  = names.index("VITAMINA_D") if "VITAMINA_D" in names else None
calcio_idx = names.index("CALCIO")    if "CALCIO"    in names else None
feat_idx   = [(n, i) for n, i in [("VITAMINA_D", vitd_idx), ("CALCIO", calcio_idx)]
              if i is not None]

if feat_idx:
    fig, axes_v = plt.subplots(1, len(feat_idx), figsize=(6 * len(feat_idx), 4.5))
    if len(feat_idx) == 1: axes_v = [axes_v]
    for ax_v, (feat_name, fi) in zip(axes_v, feat_idx):
        df_shap = pd.DataFrame({
            "SHAP": sv[:, fi],
            "Finca": g,
            "TB": ["TB+" if yi == 1 else "TB-" for yi in y],
        })
        sns.violinplot(data=df_shap, x="Finca", y="SHAP", order=EXPLOTS,
                       palette=EXPL_PAL, inner="box", ax=ax_v, alpha=0.7, cut=0)
        ax_v.axhline(0, color="black", lw=0.8, ls="--")
        ax_v.set_title(f"SHAP de {feat_name} por finca\n(positivo = mayor riesgo TB)")
        ax_v.set_xlabel("Finca"); ax_v.set_ylabel("Valor SHAP")
        for i_f, farm in enumerate(EXPLOTS):
            n_f = (g == farm).sum()
            ax_v.text(i_f, df_shap[df_shap.Finca == farm].SHAP.min() - 0.01,
                      f"n={n_f}", ha="center", fontsize=8, color="grey")
    plt.suptitle("Distribucion de valores SHAP de VitD/Calcio por finca", fontsize=11)
    plt.tight_layout()
    plt.savefig("figures/fig_06c_shap_violin_finca.png", dpi=150, bbox_inches="tight")
    plt.show()
"""))

C.append(md("md_pdp", """\
## 5. PDP + ICE condicionados a finca

**ICE (Individual Conditional Expectation):** una curva por animal que muestra como
cambia P(TB+) al variar VitD manteniendo el resto constante. Las lineas coloreadas
por finca revelan si los animales de una finca siguen trayectorias distintas.

**PDP por finca:** media de las curvas ICE dentro de cada finca — indica la tendencia
promedio intra-finca, sin el ruido individual.

Interpretacion:
- Lineas de todas las fincas con la misma pendiente → efecto universal de VitD.
- Pendiente opuesta entre fincas → interaccion finca x VitD.
- Curvas planas → VitD no cambia la prediccion dentro de la finca.
"""))

C.append(code("cell_ice_pdp", """\
CONT_SEL = [f for f in SELECTED if f in tb.NUM_FEATURES and Xs[f].nunique() > 5]
if not CONT_SEL:
    CONT_SEL = [f for f in SELECTED if f in tb.NUM_FEATURES]

def ice_pdp_curves(model, X, feat, n_grid=60):
    lo = float(X[feat].quantile(0.02))
    hi = float(X[feat].quantile(0.98))
    grid = np.linspace(lo, hi, n_grid)
    Xc = X.reset_index(drop=True).copy()
    ice = np.zeros((len(Xc), n_grid))
    for j, val in enumerate(grid):
        Xc[feat] = val
        ice[:, j] = model.predict_proba(Xc)[:, 1]
    return grid, ice, ice.mean(axis=0)

if CONT_SEL:
    n_feat = len(CONT_SEL)
    fig, axes_ice = plt.subplots(2, n_feat, figsize=(6 * n_feat, 10),
                                  squeeze=False)

    for ci, feat in enumerate(CONT_SEL):
        grid, ice_mat, pdp_g = ice_pdp_curves(FINAL, Xs, feat)

        # --- Fila 0: ICE global coloreado por finca ---
        ax0 = axes_ice[0][ci]
        for ii in range(len(ice_mat)):
            farm_i = g[ii]
            col_i  = EXPL_PAL.get(farm_i, "grey")
            lw_i   = 0.4 if y[ii] == 0 else 0.9
            al_i   = 0.12 if y[ii] == 0 else 0.35
            ax0.plot(grid, ice_mat[ii], lw=lw_i, alpha=al_i, color=col_i)
        ax0.plot(grid, pdp_g, color="black", lw=2.5, zorder=10, label="PDP global")
        ax0.axhline(y.mean(), color="grey", ls="--", lw=1, label="prevalencia")
        ax0.set_title(f"ICE por finca — {feat}\n(trazo fino=TB-, trazo grueso=TB+)")
        ax0.set_xlabel(feat); ax0.set_ylabel("P(TB+)")
        legend_h = [plt.Line2D([0], [0], color=c, lw=2, label=f)
                    for f, c in EXPL_PAL.items()]
        legend_h += [plt.Line2D([0], [0], color="black", lw=2.5, label="PDP global")]
        ax0.legend(handles=legend_h, fontsize=8, loc="upper right")

        # --- Fila 1: PDP medio por finca ---
        ax1 = axes_ice[1][ci]
        for farm, col_f in EXPL_PAL.items():
            mask_f = g == farm
            n_f = mask_f.sum()
            if n_f < 5: continue
            Xf = Xs[mask_f].reset_index(drop=True)
            _, _, pdp_farm = ice_pdp_curves(FINAL, Xf, feat)
            ax1.plot(grid, pdp_farm, color=col_f, lw=2.2, alpha=0.9,
                     label=f"{farm} (n={n_f})")
        ax1.plot(grid, pdp_g, color="black", lw=2.5, ls="--", label="global", zorder=10)
        ax1.axhline(y.mean(), color="grey", ls=":", lw=1)
        ax1.set_title(f"PDP por finca — {feat}")
        ax1.set_xlabel(feat); ax1.set_ylabel("P(TB+)")
        ax1.legend(fontsize=8)

    plt.suptitle("Partial Dependence + ICE condicionados a finca", fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("figures/fig_06c_ice_pdp.png", dpi=150, bbox_inches="tight")
    plt.show()
else:
    print("No hay features continuas en SELECTED para PDP/ICE.")
"""))

C.append(md("md_ale", """\
## 6. ALE — Accumulated Local Effects global y por finca

El ALE es preferible al PDP cuando hay correlacion entre predictores (que aqui
esperamos, porque finca correlaciona con VitD). El ALE calcula el efecto **local**
de cada feature promediando diferencias dentro de intervalos estrechos, sin
extrapolar fuera de la distribucion observada.

- **Curva negra:** ALE global (todos los animales).
- **Curvas de colores:** ALE calculado solo con los animales de cada finca.

Interpretacion:
- Si el ALE global y los ALE de finca son similares → el efecto de VitD es
  consistente entre fincas.
- Si el ALE de finca es plano pero el global tiene pendiente → la senal global
  viene del efecto entre fincas, no dentro de ellas.
"""))

C.append(code("cell_ale_farm", """\
def ale_1d(model, X, feat, bins=8):
    x = X[feat].dropna().values
    q = np.unique(np.quantile(x, np.linspace(0, 1, bins + 1)))
    if len(q) < 3:
        return np.array([]), np.array([])
    eff, Xc = [], X.copy()
    for ii in range(len(q) - 1):
        mask_b = (X[feat] >= q[ii]) & (X[feat] <= q[ii + 1])
        if not mask_b.any():
            eff.append(0.0); continue
        Xl = Xc.loc[mask_b].copy(); Xl[feat] = q[ii]
        Xh = Xc.loc[mask_b].copy(); Xh[feat] = q[ii + 1]
        eff.append((model.predict_proba(Xh)[:, 1]
                    - model.predict_proba(Xl)[:, 1]).mean())
    ale = np.cumsum(eff); ale -= ale.mean()
    return (q[:-1] + q[1:]) / 2, ale

if CONT_SEL:
    n_feat = len(CONT_SEL)
    fig, axes_ale = plt.subplots(1, n_feat, figsize=(6 * n_feat, 4.5),
                                  squeeze=False)

    for ci, feat in enumerate(CONT_SEL):
        ax = axes_ale[0][ci]

        # ALE global
        cx_g, ale_g = ale_1d(FINAL, Xs, feat, bins=10)
        if len(cx_g):
            ax.fill_between(cx_g, 0, ale_g, where=(ale_g > 0),
                            alpha=0.10, color=PALETTE[0])
            ax.fill_between(cx_g, 0, ale_g, where=(ale_g < 0),
                            alpha=0.10, color=PALETTE[1])
            ax.plot(cx_g, ale_g, color="black", lw=2.5, zorder=10, label="global")

        # ALE por finca
        for farm, col_f in EXPL_PAL.items():
            mask_f = g == farm
            n_f = mask_f.sum()
            if n_f < 8: continue
            Xf = Xs[mask_f].reset_index(drop=True)
            cx_f, ale_f = ale_1d(FINAL, Xf, feat, bins=5)
            if len(cx_f):
                ax.plot(cx_f, ale_f, color=col_f, lw=2, alpha=0.8,
                        ls="--", label=f"{farm} (n={n_f})")

        ax.axhline(0, color="grey", lw=0.8)
        ax.set_title(f"ALE global vs por finca — {feat}")
        ax.set_xlabel(feat); ax.set_ylabel("Efecto ALE en P(TB+)")
        ax.legend(fontsize=8)

    plt.suptitle(
        "ALE: efecto acumulado local de VitD/Calcio\\n"
        "(negro=global, colores=por finca, sombreado=riesgo TB+/-)",
        fontsize=11,
    )
    plt.tight_layout()
    plt.savefig("figures/fig_06c_ale_farm.png", dpi=150, bbox_inches="tight")
    plt.show()
"""))

C.append(md("md_shap_dep", """\
## 7. SHAP dependence condicionado a finca

Los SHAP dependence plots muestran la relacion entre el **valor de la feature** y
su **valor SHAP** (contribucion a la prediccion de TB+). Colorear por finca permite ver:

- Si la nube de puntos de cada finca ocupa regiones distintas → VitD difiere entre fincas.
- Si la pendiente OLS dentro de cada finca es similar → el efecto individual es consistente.
- Si la pendiente intra-finca es plana → VitD no contribuye al modelo una vez fijada la finca.

La tabla de pendientes (b) al final de la celda es el resumen mas directo:
`b < 0` significa mas VitD → SHAP mas negativo → el modelo predice menos TB.
"""))

C.append(code("cell_shap_dep", """\
# Mapear nombres originales a columnas de sv (que usa nombres de preprocesador)
def get_shap_col(feat, names):
    if feat in names: return names.index(feat)
    for i, n in enumerate(names):
        if n.endswith("__" + feat) or n == feat:
            return i
    return None

feat_to_shap = {f: get_shap_col(f, names) for f in CONT_SEL}
CONT_SHAP = [f for f in CONT_SEL if feat_to_shap[f] is not None]

if CONT_SHAP:
    n_feat = len(CONT_SHAP)
    fig, axes_dep = plt.subplots(2, n_feat, figsize=(6 * n_feat, 9),
                                  squeeze=False)

    for ci, feat in enumerate(CONT_SHAP):
        fi   = feat_to_shap[feat]
        xval = Xs[feat].values
        sval = sv[:, fi]

        # --- Fila 0: scatter por finca ---
        ax0 = axes_dep[0][ci]
        for farm, col_f in EXPL_PAL.items():
            mask_f = g == farm
            ax0.scatter(xval[mask_f], sval[mask_f], color=col_f, label=farm,
                        alpha=0.75, s=40, edgecolors="white", lw=0.5, zorder=3)
        ax0.axhline(0, color="black", lw=0.8, ls="--")
        ax0.set_title(f"SHAP dependence — {feat}\\n(coloreado por finca)")
        ax0.set_xlabel(feat); ax0.set_ylabel(f"SHAP valor ({feat})")
        ax0.legend(fontsize=8)

        # --- Fila 1: tendencia OLS intra-finca ---
        ax1 = axes_dep[1][ci]
        x_lo = float(np.nanpercentile(xval, 2))
        x_hi = float(np.nanpercentile(xval, 98))
        x_fit = np.linspace(x_lo, x_hi, 100)
        slopes_summary = {}

        for farm, col_f in EXPL_PAL.items():
            mask_f = g == farm
            xf = xval[mask_f]; sf = sval[mask_f]
            ok = np.isfinite(xf) & np.isfinite(sf)
            if ok.sum() < 4:
                slopes_summary[farm] = np.nan; continue
            ax1.scatter(xf[ok], sf[ok], color=col_f, alpha=0.35, s=25,
                        edgecolors="none")
            p = np.polyfit(xf[ok], sf[ok], 1)
            slopes_summary[farm] = round(float(p[0]), 4)
            ax1.plot(x_fit, np.polyval(p, x_fit), color=col_f, lw=2.2,
                     label=f"{farm} b={p[0]:+.3f}")

        ax1.axhline(0, color="black", lw=0.8, ls="--")
        ax1.set_title(f"Tendencia SHAP intra-finca — {feat}\\nb<0 = mas {feat} -> menos TB")
        ax1.set_xlabel(feat); ax1.set_ylabel(f"SHAP valor ({feat})")
        ax1.legend(fontsize=8)

    plt.suptitle("SHAP dependence: efecto de VitD/Calcio condicionado a finca",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    plt.savefig("figures/fig_06c_shap_dep_finca.png", dpi=150, bbox_inches="tight")
    plt.show()

    # Tabla de pendientes
    print("\nPendiente SHAP intra-finca (b, OLS) por feature y finca:")
    print("  b < 0: mas valor -> SHAP mas negativo -> el modelo ve menos riesgo TB")
    print("  b ~ 0: VitD/Calcio no cambia la prediccion dentro de esa finca")
    print()
    for feat in CONT_SHAP:
        fi   = feat_to_shap[feat]
        xval = Xs[feat].values; sval = sv[:, fi]
        print(f"  {feat}:")
        for farm in EXPLOTS:
            mask_f = g == farm
            xf = xval[mask_f]; sf = sval[mask_f]
            ok = np.isfinite(xf) & np.isfinite(sf)
            if ok.sum() < 4:
                print(f"    {farm}: n insuficiente"); continue
            p = np.polyfit(xf[ok], sf[ok], 1)
            b = float(p[0])
            direction = "protector (b<0)" if b < -0.001 else (
                        "riesgo (b>0)" if b > 0.001 else "neutro")
            print(f"    {farm} (n={ok.sum()}): b={b:+.4f}  [{direction}]")
else:
    print("No hay features continuas seleccionadas con indice SHAP mapeado.")
"""))

C.append(md("md_synthesis", """\
## 8. Interpretacion consolidada

### Clave de lectura del heatmap de importancia condicional

| Patron | Interpretacion |
|---|---|
| Importancia alta en TODAS las fincas | VitD/Calcio predicen TB intra-finca en todas partes — efecto individual real y generalizable |
| Importancia alta solo en VC30D (n grande) | La senal es especifica de VC30D — puede ser efecto real o sesgo por tamano de muestra |
| Importancia alta globalmente, baja en todas las fincas | VitD/Calcio son proxies de la finca — no predicen TB individualmente |
| Importancia negativa dentro de una finca | La permutacion MEJORA la prediccion en esa finca — la feature introduce ruido intra-finca |

### Interpretacion de SHAP por finca

- Si la distribucion de SHAP(VitD) es similar entre fincas → efecto universal.
- Si difiere drasticamente (ej. positivo en VC30D, negativo en WZ72B) → interaccion finca x VitD.

### Comparacion de los tres enfoques (06a, 06b, 06c)

| Enfoque | Pregunta central | Limitacion |
|---|---|---|
| 06a Residualizacion | ¿VitD individual predice TB? | El RF no separa bien efecto finca vs. individuo |
| 06b GLMM | ¿Cual es el OR de VitD ajustado por finca? | n=103 y 4 fincas limita la estimacion del efecto aleatorio |
| 06c Importancia condicional | ¿VitD aporta dentro de cada finca? | n por finca muy pequeno (14-55) para permutacion estable |
"""))

save(C, "06c_Importancia_Condicional_Finca_RF.ipynb")

print("\nListo. Tres notebooks generados correctamente.")
