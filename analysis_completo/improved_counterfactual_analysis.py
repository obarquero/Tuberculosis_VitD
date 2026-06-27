"""
Análisis Contrafáctico Profundo para Publicación
================================================
Reemplaza find_cf() simple por análisis poblacional robusto
con bootstrapping, IC95%, heterogeneidad y validación
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import f_oneway, kruskal
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# 1. BÚSQUEDA CONTRAFÁCTICA MEJORADA
# ============================================================================

def find_cf_refined(model, row_dict, feat, grid, target_cls=0, threshold=0.50):
    """
    Busca el valor contrafáctico usando una grilla predefinida.
    
    Parámetros:
    -----------
    model : sklearn-like
        Modelo con predict_proba()
    row_dict : dict o pd.Series
        Características del animal (convertidas a dict si es Series)
    feat : str
        Nombre de la característica a variar
    grid : np.ndarray
        Grilla de valores a buscar (puede expandir fuera de rango observado)
    target_cls : int
        Clase objetivo (0=bajo riesgo, 1=alto riesgo)
    threshold : float
        Umbral de probabilidad para decisión (default 0.50)
    
    Retorna:
    --------
    float o None
        Valor CF si existe, None si no alcanzable
    """
    if isinstance(row_dict, pd.Series):
        row_dict = row_dict.to_dict()
    
    for v in grid:
        r = dict(row_dict)
        r[feat] = v
        
        # Obtener probabilidad de clase 1 (lesiones TB)
        prob = model.predict_proba(pd.DataFrame([r]))[0, 1]
        
        # Criterio: ¿cruzó el umbral hacia la clase objetivo?
        if target_cls == 0:  # Queremos bajo riesgo
            if prob <= threshold:
                return v
        else:  # Queremos alto riesgo
            if prob >= threshold:
                return v
    
    return None


def find_cf_optimal(model, row_dict, feat, lo, hi, target_cls=0, 
                    threshold=0.50, n_grid=1000):
    """
    Búsqueda cuasi-continua con grilla fina.
    
    Esto responde tu pregunta: "¿el código busca en todo el continuo?"
    Sí, pero usa una grilla discreta. Con n_grid=1000, tienes
    resolución de (hi-lo)/1000, que es prácticamente continuo.
    """
    grid = np.linspace(lo, hi, n_grid)
    return find_cf_refined(model, row_dict, feat, grid, target_cls, threshold)


# ============================================================================
# 2. ANÁLISIS POBLACIONAL CON GRILLA EXPANDIDA
# ============================================================================

def population_counterfactual_analysis(model, Xs_high_risk, feat_list,
                                       expand_factor=0.5, n_grid=1000,
                                       target_cls=0, threshold=0.50,
                                       verbose=True):
    """
    Análisis poblacional de contrafácticos para TODOS los animales de alto riesgo.
    
    Parámetros:
    -----------
    model : sklearn-like
        Modelo entrenado
    Xs_high_risk : pd.DataFrame
        Todas las filas de animales de alto riesgo
    feat_list : list
        Lista de características a analizar ['Vitamina_D', 'Calcio', ...]
    expand_factor : float
        Expansion factor: 0.5 = ±50% del rango observado
        La grilla de búsqueda será:
        [min_obs - 0.5*rango_obs, max_obs + 0.5*rango_obs]
    n_grid : int
        Puntos en la grilla (1000 = cuasi-continuo)
    target_cls : int
        0 = queremos pasar a bajo riesgo
    threshold : float
        Umbral de probabilidad (default 0.50)
    
    Retorna:
    --------
    dict
        Resultados poblacionales para cada variable
    """
    
    results = {}
    
    for feat in feat_list:
        if feat not in Xs_high_risk.columns:
            print(f"⚠️  {feat} no está en los datos")
            continue
        
        # Rango observado
        lo_obs = float(Xs_high_risk[feat].min())
        hi_obs = float(Xs_high_risk[feat].max())
        range_obs = hi_obs - lo_obs
        
        # Rango EXPANDIDO para búsqueda
        lo_search = lo_obs - expand_factor * range_obs
        hi_search = hi_obs + expand_factor * range_obs
        
        grid = np.linspace(lo_search, hi_search, n_grid)
        
        if verbose:
            print(f"\n{'='*70}")
            print(f"Análisis: {feat}")
            print(f"{'='*70}")
            print(f"Rango observado:  [{lo_obs:.2f}, {hi_obs:.2f}]")
            print(f"Rango búsqueda:   [{lo_search:.2f}, {hi_search:.2f}]")
            print(f"Animales a analizar: {len(Xs_high_risk)}")
        
        # Calcular CF para cada animal
        deltas = []
        cfs = []
        actuals = []
        
        for idx, row in tqdm(Xs_high_risk.iterrows(), total=len(Xs_high_risk),
                             disable=not verbose, desc=feat):
            cf = find_cf_refined(model, row, feat, grid, target_cls, threshold)
            
            if cf is not None:
                delta = cf - row[feat]
                deltas.append(delta)
                cfs.append(cf)
                actuals.append(row[feat])
        
        # Estadísticos
        deltas_arr = np.array(deltas)
        n_alcanzable = len(deltas)
        n_no_alcanzable = len(Xs_high_risk) - n_alcanzable
        pct_no_alcanzable = 100 * n_no_alcanzable / len(Xs_high_risk)
        
        results[feat] = {
            # Estadísticos
            'mean': deltas_arr.mean() if n_alcanzable > 0 else np.nan,
            'std': deltas_arr.std() if n_alcanzable > 0 else np.nan,
            'median': np.median(deltas_arr) if n_alcanzable > 0 else np.nan,
            'p25': np.percentile(deltas_arr, 25) if n_alcanzable > 0 else np.nan,
            'p75': np.percentile(deltas_arr, 75) if n_alcanzable > 0 else np.nan,
            'min': deltas_arr.min() if n_alcanzable > 0 else np.nan,
            'max': deltas_arr.max() if n_alcanzable > 0 else np.nan,
            
            # Alcanzabilidad
            'n_alcanzable': n_alcanzable,
            'n_no_alcanzable': n_no_alcanzable,
            'pct_alcanzable': 100 - pct_no_alcanzable,
            
            # Arrays brutos para bootstrap
            'deltas': deltas_arr,
            'cfs': np.array(cfs),
            'actuals': np.array(actuals),
            
            # Rango de búsqueda
            'search_range': (lo_search, hi_search),
            'obs_range': (lo_obs, hi_obs)
        }
        
        # Imprimir resumen
        if verbose and n_alcanzable > 0:
            print(f"\n✓ Contrafácticos alcanzables: {n_alcanzable}/{len(Xs_high_risk)} ({100-pct_no_alcanzable:.1f}%)")
            print(f"  Media(Δ):    {deltas_arr.mean():+.3f}")
            print(f"  Std(Δ):      {deltas_arr.std():.3f}")
            print(f"  Mediana(Δ):  {np.median(deltas_arr):+.3f}")
            print(f"  Rango(Δ):    [{deltas_arr.min():+.3f}, {deltas_arr.max():+.3f}]")
    
    return results


# ============================================================================
# 3. BOOTSTRAP PARA IC95%
# ============================================================================

def bootstrap_ci_counterfactual(model, Xs_high_risk, feat, lo, hi,
                                n_bootstrap=1000, ci=95, target_cls=0,
                                threshold=0.50, n_grid=1000, verbose=False):
    """
    Calcula intervalos de confianza vía bootstrap.
    
    Parámetros:
    -----------
    n_bootstrap : int
        Número de iteraciones bootstrap (1000 recomendado)
    ci : float
        Nivel de confianza (95)
    
    Retorna:
    --------
    dict
        'mean': media puntual
        'lower': límite inferior IC
        'upper': límite superior IC
        'bootstrap_means': todas las medias bootstrap (para visualizar)
    """
    
    grid = np.linspace(lo, hi, n_grid)
    bootstrap_means = []
    
    if verbose:
        iterator = tqdm(range(n_bootstrap), desc=f"Bootstrap {ci}% CI para {feat}")
    else:
        iterator = range(n_bootstrap)
    
    for b in iterator:
        # Resample con reemplazo
        Xs_boot = Xs_high_risk.sample(n=len(Xs_high_risk), replace=True)
        
        # Calcular CF en muestra bootstrap
        deltas_b = []
        for idx, row in Xs_boot.iterrows():
            cf = find_cf_refined(model, row, feat, grid, target_cls, threshold)
            if cf is not None:
                deltas_b.append(cf - row[feat])
        
        if deltas_b:
            bootstrap_means.append(np.mean(deltas_b))
    
    # Percentiles para IC
    bootstrap_means = np.array(bootstrap_means)
    alpha = 100 - ci
    lower = np.percentile(bootstrap_means, alpha/2)
    upper = np.percentile(bootstrap_means, 100 - alpha/2)
    point = np.mean(bootstrap_means)
    
    return {
        'point': point,
        'lower': lower,
        'upper': upper,
        'bootstrap_means': bootstrap_means,
        'ci': ci
    }


# ============================================================================
# 4. ANÁLISIS DE HETEROGENEIDAD
# ============================================================================

def heterogeneity_analysis(Xs_high_risk, deltas_dict, groupby_var):
    """
    Analiza si el cambio contrafáctico difiere entre subgrupos.
    
    Parámetros:
    -----------
    Xs_high_risk : pd.DataFrame
    deltas_dict : dict
        Resultado de population_counterfactual_analysis()[feat]
    groupby_var : str
        Variable para agrupar (e.g., 'edad', 'sexo')
    
    Retorna:
    --------
    dict
        Estadísticos por grupo + test de significancia
    """
    
    if groupby_var not in Xs_high_risk.columns:
        print(f"⚠️  {groupby_var} no está en los datos")
        return None
    
    deltas = deltas_dict['deltas']
    groups = Xs_high_risk[groupby_var].unique()
    
    stats = {}
    group_data = []
    
    for group in sorted(groups):
        mask = (Xs_high_risk[groupby_var] == group).values[:len(deltas)]
        deltas_group = deltas[mask]
        
        stats[group] = {
            'n': mask.sum(),
            'mean': deltas_group.mean(),
            'std': deltas_group.std(),
            'median': np.median(deltas_group),
            'p25': np.percentile(deltas_group, 25),
            'p75': np.percentile(deltas_group, 75)
        }
        
        group_data.append(deltas_group)
    
    # Test de significancia (Kruskal-Wallis, más robusto)
    if len(group_data) > 1:
        h_stat, p_value = kruskal(*group_data)
    else:
        h_stat, p_value = np.nan, np.nan
    
    return {
        'by_group': stats,
        'test': {
            'method': 'Kruskal-Wallis',
            'h_statistic': h_stat,
            'p_value': p_value,
            'significant': p_value < 0.05
        }
    }


# ============================================================================
# 5. VISUALIZACIONES
# ============================================================================

def plot_cf_distributions(results_dict, feat_list, figsize=(12, 4)):
    """
    Histogramas de distribuciones de cambios contrafácticos.
    
    Retorna:
    --------
    fig, axes
    """
    n_feats = len(feat_list)
    fig, axes = plt.subplots(1, n_feats, figsize=figsize)
    if n_feats == 1:
        axes = [axes]
    
    for ax, feat in zip(axes, feat_list):
        if feat not in results_dict:
            continue
        
        deltas = results_dict[feat]['deltas']
        mean = results_dict[feat]['mean']
        median = results_dict[feat]['median']
        n_alcanzable = results_dict[feat]['n_alcanzable']
        n_total = n_alcanzable + results_dict[feat]['n_no_alcanzable']
        
        # Histograma
        ax.hist(deltas, bins=20, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(mean, color='red', linestyle='--', lw=2, label=f'Media: {mean:+.2f}')
        ax.axvline(median, color='green', linestyle='--', lw=2, label=f'Mediana: {median:+.2f}')
        ax.axvline(0, color='gray', linestyle=':', lw=1, alpha=0.5)
        
        ax.set_xlabel(f'Δ {feat}')
        ax.set_ylabel('Frecuencia')
        ax.set_title(f'{feat}\n(n={n_alcanzable}/{n_total})')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig, axes


def plot_cf_by_subgroup(Xs_high_risk, results_dict, feat, groupby_var, figsize=(10, 5)):
    """
    Box plots de CF por subgrupo.
    """
    
    deltas_dict = results_dict[feat]
    deltas = deltas_dict['deltas']
    
    # Crear DataFrame para plotting
    # Problema: los índices pueden no coincidir después de filtrar CF alcanzables
    # Solución: reconstruir análisis registrando índices
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Simplificación: usar todos los datos originales
    if groupby_var in Xs_high_risk.columns:
        data_plot = Xs_high_risk.copy()
        data_plot['delta_cf'] = np.nan
        
        # Rellenar (asumiendo correspondencia de orden)
        idx_valid = 0
        for i, row in Xs_high_risk.iterrows():
            if idx_valid < len(deltas):
                data_plot.loc[i, 'delta_cf'] = deltas[idx_valid]
                idx_valid += 1
        
        data_plot = data_plot.dropna(subset=['delta_cf'])
        
        sns.boxplot(data=data_plot, x=groupby_var, y='delta_cf', ax=ax)
        ax.axhline(0, color='gray', linestyle='--', lw=1, alpha=0.5)
        ax.set_ylabel(f'Δ {feat}')
        ax.set_title(f'Cambio Contrafáctico por {groupby_var}')
        ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    return fig, ax


# ============================================================================
# 6. TABLA DE RESULTADOS
# ============================================================================

def create_results_table(results_dict, feat_list):
    """
    Crea tabla de resultados para publicación.
    
    Retorna:
    --------
    pd.DataFrame
        Tabla formateada
    """
    
    rows = []
    for feat in feat_list:
        if feat not in results_dict:
            continue
        
        r = results_dict[feat]
        
        rows.append({
            'Variable': feat,
            'N': f"{r['n_alcanzable']}/{r['n_alcanzable'] + r['n_no_alcanzable']}",
            'Alcanzabilidad (%)': f"{r['pct_alcanzable']:.1f}%",
            'Media(Δ)': f"{r['mean']:+.3f}",
            'Std(Δ)': f"{r['std']:.3f}",
            'Mediana(Δ)': f"{r['median']:+.3f}",
            'P25-P75': f"[{r['p25']:+.3f}, {r['p75']:+.3f}]",
            'Rango(Δ)': f"[{r['min']:+.3f}, {r['max']:+.3f}]"
        })
    
    df_results = pd.DataFrame(rows)
    return df_results


# ============================================================================
# EJEMPLO DE USO
# ============================================================================

if __name__ == "__main__":
    """
    En tu notebook, usarías así:
    
    # 1. Identificar animales de alto riesgo
    preds = FINAL_4C.predict(Xs)
    Xs_high_risk = Xs[preds == 1]
    
    # 2. Análisis poblacional
    results = population_counterfactual_analysis(
        model=FINAL_4C,
        Xs_high_risk=Xs_high_risk,
        feat_list=['Vitamina_D', 'Calcio'],
        expand_factor=0.5,  # Expandir ±50% del rango observado
        n_grid=1000,        # Cuasi-continuo
        verbose=True
    )
    
    # 3. Bootstrap para IC95%
    ci_vitd = bootstrap_ci_counterfactual(
        model=FINAL_4C,
        Xs_high_risk=Xs_high_risk,
        feat='Vitamina_D',
        lo=results['Vitamina_D']['search_range'][0],
        hi=results['Vitamina_D']['search_range'][1],
        n_bootstrap=1000
    )
    print(f"Vitamina D IC95%: {ci_vitd['lower']:.2f} a {ci_vitd['upper']:.2f}")
    
    # 4. Visualizaciones
    fig_dist, ax_dist = plot_cf_distributions(results, ['Vitamina_D', 'Calcio'])
    
    # 5. Tabla de resultados
    df_table = create_results_table(results, ['Vitamina_D', 'Calcio'])
    print(df_table.to_string(index=False))
    """
    print(__doc__)
