"""
tb_utils.py
===========
Funciones compartidas para el pipeline de tuberculosis bovina (n=106, 4 explotaciones).
Importado por los tres notebooks para garantizar coherencia (mismo preprocesamiento,
mismas variables, misma validación) y evitar fugas de información (data leakage).

Autoría: pipeline reproducible. Licencia de uso: investigación.
"""
from __future__ import annotations
import warnings
import numpy as np
import pandas as pd
from scipy import stats

# ----------------------------------------------------------------------------- #
# 0. Reproducibilidad
# ----------------------------------------------------------------------------- #
SEED = 42

def set_seeds(seed: int = SEED) -> None:
    """Fija todas las semillas relevantes."""
    import os, random
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    try:
        import xgboost  # noqa: F401
    except Exception:
        pass

# ----------------------------------------------------------------------------- #
# 1. Definiciones de variables (ancladas a la estructura real de BD.csv)
# ----------------------------------------------------------------------------- #
# Valores que cuentan como "positivo" en serología
_POS = {"Positivo", "positivo", "POSITIVO", "Pos", "SI", "Sí", "1", "1.0"}
_NA_TOKENS = ["NA", "na", "ND", "nd", "", " ", "N/A", "n/a"]

# Predictores serológicos binarios (Positivo/Negativo)
BIN_SEROLOGY = ["PIROPLASMA", "THEILERIA", "ANAPLASMA"]
# Continuas
CONTINUOUS = ["VITAMINA_D", "CALCIO", "PIROPLASMA_Q", "EDAD"]
# Categóricas nominales
NOMINAL = ["SEXO", "RAZA"]
GROUP_COL = "Expl"          # explotación (granja) -> validación agrupada
ID_COL = "Ide"

# Conjunto de features para modelado (PIROPLASMA_Q se usa en log; RAZA colapsada)
MODEL_FEATURES = [
    "VITAMINA_D", "CALCIO", "PIROPLASMA_Q_log", "EDAD",
    "PIROPLASMA", "THEILERIA", "ANAPLASMA", "RAZA2",
]
NUM_FEATURES = ["VITAMINA_D", "CALCIO", "PIROPLASMA_Q_log", "EDAD",
                "PIROPLASMA", "THEILERIA", "ANAPLASMA"]
CAT_FEATURES = ["RAZA2"]

# Etiquetas legibles para figuras
PRETTY = {
    "VITAMINA_D": "Vitamina D", "CALCIO": "Calcio",
    "PIROPLASMA_Q": "Carga piroplasma (q)", "PIROPLASMA_Q_log": "log(1+Carga piroplasma)",
    "EDAD": "Edad (años)", "PIROPLASMA": "Piroplasma (+)",
    "THEILERIA": "Theileria spp. (+)", "ANAPLASMA": "Anaplasma spp. (+)",
    "RAZA2": "Raza", "SEXO": "Sexo",
    "Lesiones_TB": "Lesión TB", "Patron_lesiones": "Patrón (generalizado)",
    "Patron_lesiones_3": "Patrón (3 clases)",
    "Score_lesional": "Score lesional", "IDTC": "Intensidad IDTC",
}

# ----------------------------------------------------------------------------- #
# 2. Carga y limpieza
# ----------------------------------------------------------------------------- #
def load_raw(path: str) -> pd.DataFrame:
    """Carga BD.csv probando codificaciones (el archivo real es cp1252)."""
    last_err = None
    for enc in ("utf-8", "cp1252", "latin-1"):
        try:
            df = pd.read_csv(path, sep=";", encoding=enc,
                             na_values=_NA_TOKENS, skipinitialspace=True)
            df.columns = [c.strip() for c in df.columns]
            return df
        except Exception as e:  # pragma: no cover
            last_err = e
    raise RuntimeError(f"No se pudo leer {path}: {last_err}")


def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Tipa y limpia preservando la ausencia explícita (NA) sin imputar aquí.

    - Serología -> 0/1 (NA preservado).
    - Numéricas -> float (NA preservado).
    - PIROPLASMA_Q_log = log1p(PIROPLASMA_Q) por fuerte asimetría y ceros.
    - RAZA2: colapsa la categoría singleton 'Angus' (n=1) en 'Otra'.
    - Define máscaras de subconjuntos para targets condicionales.
    """
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]

    for c in BIN_SEROLOGY:
        def _enc(v):
            if pd.isna(v):
                return np.nan
            if isinstance(v, str):
                return 1.0 if v.strip() in _POS else 0.0
            return float(v)  # ya numérico (idempotente): respeta 0/1 existentes
        df[c] = df[c].apply(_enc)
    for c in ["VITAMINA_D", "CALCIO", "PIROPLASMA_Q", "EDAD", "IDTC",
              "Score_lesional", "Lesiones_TB", "Patron_lesiones"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df["PIROPLASMA_Q_log"] = np.log1p(df["PIROPLASMA_Q"])
    df["RAZA2"] = df["RAZA"].replace({"Angus": "Otra"}).astype("category")
    df["SEXO"] = df["SEXO"].astype("category")
    df[GROUP_COL] = df[GROUP_COL].astype(str)
    
    if "Patron_lesiones" in df.columns:
        df["Patron_lesiones_3"] = df["Patron_lesiones"].fillna(-1).astype(int)

    return df


def drop_sparse_rows(df: pd.DataFrame, features=None, max_nan: int = 4) -> pd.DataFrame:
    """Elimina filas con más de max_nan NaN entre las features del modelo.

    Llámalo después de target_subset, antes de construir X e y.
    """
    features = MODEL_FEATURES if features is None else features
    present = [f for f in features if f in df.columns]
    n_nan = df[present].isna().sum(axis=1)
    mask = n_nan <= max_nan
    dropped = int((~mask).sum())
    if dropped:
        print(f"drop_sparse_rows: {dropped} fila(s) eliminada(s) (>{max_nan} NaN en features). "
              f"n restante = {int(mask.sum())}.")
    return df[mask].reset_index(drop=True)


def target_subset(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Devuelve el subconjunto válido para cada target, respetando su naturaleza.

    Lesiones_TB    -> todo el cohorte (n=106)
    Patron_lesiones-> SOLO animales lesionados (NA estructural = no lesionados); n=65
    Patron_lesiones_3 -> todo el cohorte, 3 clases: -1 (sin lesión), 0 (localizado), 1 (generalizado)
    Score_lesional -> todo el cohorte (0 = sin lesión); ordinal {0,1,2,4,5} (sin 3)
    IDTC           -> todo el cohorte (recuento)
    """
    if target == "Patron_lesiones":
        # -1 indica ausencia de lesión, nos quedamos solo con los animales que sí tienen patrón (0 o 1)
        sub = df[df["Lesiones_TB"] == 1].dropna(subset=["Patron_lesiones"])
        return sub[sub["Patron_lesiones"] != -1].copy()
    if target == "Patron_lesiones_3":
        return df.dropna(subset=["Patron_lesiones_3"]).copy()
    return df.dropna(subset=[target]).copy()


# ----------------------------------------------------------------------------- #
# 3. Preprocesamiento (dentro de CV -> sin fuga de imputación)
# ----------------------------------------------------------------------------- #
def make_preprocessor(numeric=None, categorical=None, scale=True):
    """ColumnTransformer: imputación + (opcional) escalado + one-hot. Se ajusta SOLO en train.

    scale=True  → incluye StandardScaler (necesario para LogReg, SVMs, redes).
    scale=False → omite escalado (innecesario para Random Forest / árboles en general).
    """
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler, OneHotEncoder
    numeric = NUM_FEATURES if numeric is None else numeric
    categorical = CAT_FEATURES if categorical is None else categorical
    num_steps = [("imp", SimpleImputer(strategy="median", add_indicator=False))]
    if scale:
        num_steps.append(("sc", StandardScaler()))
    num_pipe = Pipeline(num_steps)
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("oh", OneHotEncoder(handle_unknown="ignore", drop=None)),
    ])
    return ColumnTransformer(
        [("num", num_pipe, numeric), ("cat", cat_pipe, categorical)],
        remainder="drop", verbose_feature_names_out=False,
    )

# ----------------------------------------------------------------------------- #
# 4. Esquemas de validación cruzada (con SOLO 4 explotaciones)
# ----------------------------------------------------------------------------- #
def outer_cv_optimistic(n_splits=5, n_repeats=10, seed=SEED):
    """CV estratificada repetida que IGNORA la granja -> estimación optimista."""
    from sklearn.model_selection import RepeatedStratifiedKFold
    return RepeatedStratifiedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

def outer_cv_realistic():
    """Leave-One-Farm-Out (4 folds). Estima generalización a NUEVAS granjas."""
    from sklearn.model_selection import LeaveOneGroupOut
    return LeaveOneGroupOut()

def inner_cv_group(n_splits=3):
    from sklearn.model_selection import GroupKFold
    return GroupKFold(n_splits=n_splits)

# ----------------------------------------------------------------------------- #
# 5. Estadística: tamaños de efecto y test bivariados
# ----------------------------------------------------------------------------- #
def cliffs_delta(a, b) -> float:
    """Cliff's delta (no paramétrico). |d|: <0.147 insignif, <0.33 peq, <0.474 med, sino grande."""
    a = np.asarray(a, float); b = np.asarray(b, float)
    a = a[~np.isnan(a)]; b = b[~np.isnan(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    gt = sum((x > b).sum() for x in a)
    lt = sum((x < b).sum() for x in a)
    return (gt - lt) / (len(a) * len(b))

def cramers_v(tab: np.ndarray) -> float:
    tab = np.asarray(tab, float)
    chi2 = stats.chi2_contingency(tab, correction=False)[0]
    n = tab.sum()
    k = min(tab.shape) - 1
    return np.sqrt(chi2 / (n * k)) if (n > 0 and k > 0) else np.nan

def fdr(pvals) -> np.ndarray:
    """Benjamini-Hochberg FDR."""
    from statsmodels.stats.multitest import multipletests
    p = np.asarray(pvals, float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    if ok.sum() > 0:
        out[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return out

def boot_ci_median_diff(x0, x1, n_boot=2000, seed=SEED):
    """IC bootstrap (percentil) de la diferencia de medianas (grupo1 - grupo0)."""
    rng = np.random.default_rng(seed)
    x0 = np.asarray(x0, float); x1 = np.asarray(x1, float)
    x0 = x0[~np.isnan(x0)]; x1 = x1[~np.isnan(x1)]
    diffs = [np.median(rng.choice(x1, len(x1))) - np.median(rng.choice(x0, len(x0)))
             for _ in range(n_boot)]
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))

def bivariate_vs_binary(df: pd.DataFrame, target: str,
                        continuous=None, binary=None, nominal=None) -> pd.DataFrame:
    """Tabla bivariada predictor~target binario, con test adecuado + FDR + efecto.

    Continuas -> Shapiro; si normal: Welch t, si no: Mann-Whitney (+ Cliff's delta).
    Binarias/Nominales -> Fisher (2x2 con celdas<5) o Chi2 (+ Cramér's V).
    """
    continuous = ["VITAMINA_D", "CALCIO", "PIROPLASMA_Q_log", "EDAD"] if continuous is None else continuous
    binary = BIN_SEROLOGY if binary is None else binary
    nominal = ["RAZA2", "SEXO"] if nominal is None else nominal

    sub = df.dropna(subset=[target])
    g0, g1 = sub[sub[target] == 0], sub[sub[target] == 1]
    rows = []
    for c in continuous:
        x0, x1 = g0[c].dropna(), g1[c].dropna()
        if len(x0) < 3 or len(x1) < 3:
            continue
        pooled = sub[c].dropna()
        normal = stats.shapiro(pooled)[1] > 0.05 if len(pooled) >= 3 else False
        if normal:
            stat, p = stats.ttest_ind(x0, x1, equal_var=False); test = "Welch t"
        else:
            stat, p = stats.mannwhitneyu(x0, x1, alternative="two-sided"); test = "Mann-Whitney U"
        lo, hi = boot_ci_median_diff(x0, x1)
        rows.append(dict(variable=c, test=test, n0=len(x0), n1=len(x1),
                         med0=float(x0.median()), med1=float(x1.median()),
                         effect=cliffs_delta(x1, x0), effect_type="Cliff δ",
                         ci_low=lo, ci_high=hi, p=p))
    for c in list(binary) + list(nominal):
        tab = pd.crosstab(sub[c], sub[target])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        if tab.shape == (2, 2) and (tab.values < 5).any():
            _, p = stats.fisher_exact(tab); test = "Fisher exact"
        else:
            _, p, _, _ = stats.chi2_contingency(tab); test = "Chi²"
        rows.append(dict(variable=c, test=test, n0=int(tab.iloc[:, 0].sum()),
                         n1=int(tab.iloc[:, 1].sum()), med0=np.nan, med1=np.nan,
                         effect=cramers_v(tab.values), effect_type="Cramér V",
                         ci_low=np.nan, ci_high=np.nan, p=p))
    res = pd.DataFrame(rows)
    if len(res):
        res["p_fdr"] = fdr(res["p"].values)
        res = res.sort_values("p").reset_index(drop=True)
    return res

def bivariate_vs_multiclass(df: pd.DataFrame, target: str,
                            continuous=None, binary=None, nominal=None) -> pd.DataFrame:
    """Tabla bivariada predictor~target multiclase (>= 3 clases), con test adecuado.
    
    Continuas -> Kruskal-Wallis H-test.
    Binarias/Nominales -> Chi2.
    """
    continuous = ["VITAMINA_D", "CALCIO", "PIROPLASMA_Q_log", "EDAD"] if continuous is None else continuous
    binary = BIN_SEROLOGY if binary is None else binary
    nominal = ["RAZA2", "SEXO"] if nominal is None else nominal

    sub = df.dropna(subset=[target])
    classes = np.sort(sub[target].unique())
    if len(classes) < 3:
        return bivariate_vs_binary(df, target, continuous, binary, nominal)
        
    rows = []
    for c in continuous:
        groups = [sub[sub[target] == k][c].dropna() for k in classes]
        groups = [g for g in groups if len(g) >= 3]
        if len(groups) < 2:
            continue
        
        try:
            stat, p = stats.kruskal(*groups); test = "Kruskal-Wallis"
        except ValueError:
            continue
            
        rows.append(dict(variable=c, test=test, n_groups=len(groups),
                         p=p))
                         
    for c in list(binary) + list(nominal):
        tab = pd.crosstab(sub[c], sub[target])
        if tab.shape[0] < 2 or tab.shape[1] < 2:
            continue
        
        _, p, _, _ = stats.chi2_contingency(tab); test = "Chi²"
        rows.append(dict(variable=c, test=test, n_groups=tab.shape[1],
                         p=p))
                         
    res = pd.DataFrame(rows)
    if len(res):
        res["p_fdr"] = fdr(res["p"].values)
        res = res.sort_values("p").reset_index(drop=True)
    return res


def spearman_table(df: pd.DataFrame, target: str, predictors) -> pd.DataFrame:
    """Spearman rho + IC bootstrap + FDR (para targets ordinales/continuos)."""
    rng = np.random.default_rng(SEED)
    rows = []
    for c in predictors:
        s = df[[c, target]].dropna()
        if len(s) < 5:
            continue
        rho, p = stats.spearmanr(s[c], s[target])
        boots = []
        idx = np.arange(len(s))
        for _ in range(2000):
            b = rng.choice(idx, len(idx))
            boots.append(stats.spearmanr(s.iloc[b, 0], s.iloc[b, 1]).correlation)
        rows.append(dict(variable=c, rho=rho, n=len(s),
                         ci_low=np.nanpercentile(boots, 2.5),
                         ci_high=np.nanpercentile(boots, 97.5), p=p))
    res = pd.DataFrame(rows)
    if len(res):
        res["p_fdr"] = fdr(res["p"].values)
        res = res.sort_values("p").reset_index(drop=True)
    return res

# ----------------------------------------------------------------------------- #
# 6. Causalidad: E-value (VanderWeele & Ding, 2017)
# ----------------------------------------------------------------------------- #
def e_value_rr(rr: float) -> float:
    """E-value para un riesgo relativo (o RR aproximado). Mide robustez a confusión."""
    rr = max(rr, 1.0 / rr)
    return float(rr + np.sqrt(rr * (rr - 1)))

def e_value_ci(rr: float, lo: float, hi: float):
    """E-value puntual y para el límite del IC más cercano al nulo (1)."""
    point = e_value_rr(rr)
    bound = 1.0 if (lo < 1 < hi) else (lo if rr > 1 else hi)
    ev_bound = 1.0 if bound == 1.0 else e_value_rr(bound)
    return point, ev_bound

# ----------------------------------------------------------------------------- #
# 7. Estética de figuras (publication-ready, paleta accesible)
# ----------------------------------------------------------------------------- #
def set_plot_style():
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.25, "legend.frameon": False,
        "font.family": "DejaVu Sans",
    })
    # Paleta accesible (Okabe-Ito)
    return ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]

OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    set_seeds()
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "/mnt/project/BD.csv"
    df = clean(load_raw(path))
    print("OK load+clean:", df.shape)
    print("Lesiones_TB prev:", round(df["Lesiones_TB"].mean(), 3))
    print("Patron subset n:", len(target_subset(df, "Patron_lesiones")))
    print(bivariate_vs_binary(df, "Lesiones_TB").round(3).to_string(index=False))
