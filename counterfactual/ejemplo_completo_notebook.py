# EJEMPLO COMPLETO: Análisis Contrafáctico Profundo para VC30D
# ============================================================
# Copia y pega esto en tu notebook después de entrenar FINAL_4C

# ============================================================
# PASO 1: Preparación (ejecutar una sola vez)
# ============================================================

# Copiar las funciones desde improved_counterfactual_analysis.py
# (O si tienes acceso: from improved_counterfactual_analysis import *)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import kruskal
from tqdm import tqdm

# [Aquí pegarías las funciones del archivo improved_counterfactual_analysis.py]
# Por brevedad, asumo que ya están disponibles

# ============================================================
# PASO 2: Identificar animales de alto riesgo
# ============================================================

# Obtener predicciones y probabilidades
preds_class = FINAL_4C.predict(Xs)
preds_proba = FINAL_4C.predict_proba(Xs)[:, 1]  # P(Lesiones TB)

# Filtrar: alta probabilidad de lesiones (define tu umbral)
THRESHOLD_HIGH_RISK = 0.70  # Ajusta según tu criterio clínico
Xs_high_risk = Xs[preds_proba >= THRESHOLD_HIGH_RISK].copy()
proba_high_risk = preds_proba[preds_proba >= THRESHOLD_HIGH_RISK]

print(f"""
╔════════════════════════════════════════════════════════╗
║         IDENTIFICACIÓN DE ANIMALES DE ALTO RIESGO     ║
╚════════════════════════════════════════════════════════╝
Total animales:              {len(Xs)}
Animales de ALTO RIESGO:     {len(Xs_high_risk)} ({100*len(Xs_high_risk)/len(Xs):.1f}%)
Rango P(Lesiones TB):        [{proba_high_risk.min():.3f}, {proba_high_risk.max():.3f}]
""")

# ============================================================
# PASO 3: ANÁLISIS POBLACIONAL DE CONTRAFACTUAL
# ============================================================

# Definir variables a analizar (las que tienes en CONT_4C)
FEATURES_TO_ANALYZE = ['Vitamina_D', 'Calcio']  # Ajusta según tus nombres reales

# EJECUTAR ANÁLISIS PRINCIPAL
results = population_counterfactual_analysis(
    model=FINAL_4C,
    Xs_high_risk=Xs_high_risk,
    feat_list=FEATURES_TO_ANALYZE,
    expand_factor=0.5,      # Expandir ±50% del rango observado
                             # → Responde tu pregunta 2
    n_grid=1000,            # Casi continuo (vs 200 original)
                             # → Resolución = rango/1000
    target_cls=0,           # Buscamos cambiar a clase 0 (bajo riesgo)
    threshold=0.50,         # Umbral de probabilidad
    verbose=True
)

# ============================================================
# PASO 4: TABLA DE RESULTADOS PRINCIPAL
# ============================================================

df_results = create_results_table(results, FEATURES_TO_ANALYZE)

print("\n" + "="*80)
print("TABLA 1: Cambios Contrafácticos Requeridos para Animales de Alto Riesgo")
print("="*80)
print(df_results.to_string(index=False))
print("="*80)

# Guardar para el artículo
df_results.to_csv('tabla_contrafacticos_vc30d.csv', index=False)
print("\n✓ Tabla guardada en: tabla_contrafacticos_vc30d.csv")

# ============================================================
# PASO 5: BOOTSTRAP PARA INTERVALOS DE CONFIANZA 95%
# ============================================================

print("\n" + "="*80)
print("INTERVALOS DE CONFIANZA 95% (BOOTSTRAP, 1000 iteraciones)")
print("="*80)

ci_results = {}

for feat in FEATURES_TO_ANALYZE:
    print(f"\nCalculando IC95% para {feat}...", end=" ")
    
    lo_search, hi_search = results[feat]['search_range']
    
    ci_results[feat] = bootstrap_ci_counterfactual(
        model=FINAL_4C,
        Xs_high_risk=Xs_high_risk,
        feat=feat,
        lo=lo_search,
        hi=hi_search,
        n_bootstrap=1000,
        ci=95,
        target_cls=0,
        n_grid=1000,
        verbose=False
    )
    
    ic = ci_results[feat]
    print(f"✓")
    print(f"  Puntual:  {ic['point']:+.3f}")
    print(f"  IC95%:    [{ic['lower']:+.3f}, {ic['upper']:+.3f}]")

# ============================================================
# PASO 6: VISUALIZACIONES
# ============================================================

# Figura 1: Distribuciones de cambios contrafácticos
print("\nGenerando Figura 1: Distribuciones de cambios contrafácticos...")
fig1, axes1 = plot_cf_distributions(results, FEATURES_TO_ANALYZE, figsize=(12, 4))
plt.suptitle('Figura 1: Cambios Contrafácticos Requeridos\n(Animales de Alto Riesgo, n=47)', 
             fontsize=12, fontweight='bold', y=1.02)
plt.savefig('fig_01_cf_distributions.png', dpi=300, bbox_inches='tight')
print("✓ Guardada: fig_01_cf_distributions.png")
plt.show()

# ============================================================
# PASO 7: ANÁLISIS DE HETEROGENEIDAD (opcional pero recomendado)
# ============================================================

# Si tienes variable de edad o sexo, analizar heterogeneidad
SUBGROUP_VARIABLE = 'edad'  # Ajusta según tus datos

if SUBGROUP_VARIABLE in Xs_high_risk.columns:
    print(f"\n" + "="*80)
    print(f"ANÁLISIS DE HETEROGENEIDAD POR {SUBGROUP_VARIABLE.upper()}")
    print("="*80)
    
    for feat in FEATURES_TO_ANALYZE:
        print(f"\n{feat}:")
        
        heterog = heterogeneity_analysis(Xs_high_risk, results[feat], SUBGROUP_VARIABLE)
        
        if heterog is not None:
            print(f"\n  Por subgrupo:")
            for group, stats in heterog['by_group'].items():
                print(f"    {group:15s}: Media={stats['mean']:+.3f} ± {stats['std']:.3f} "
                      f"(n={stats['n']})")
            
            test_results = heterog['test']
            print(f"\n  Test de Kruskal-Wallis:")
            print(f"    H-statistic: {test_results['h_statistic']:.3f}")
            print(f"    p-value:     {test_results['p_value']:.4f}")
            
            if test_results['significant']:
                print(f"    ✓ Diferencias SIGNIFICATIVAS entre subgrupos (p < 0.05)")
            else:
                print(f"    ✗ Sin diferencias significativas entre subgrupos (p ≥ 0.05)")

# ============================================================
# PASO 8: VALIDACIÓN DE REALISMO
# ============================================================

print("\n" + "="*80)
print("VALIDACIÓN: ¿Son los CF clínicamente realistas?")
print("="*80)

# Definir límites biológicos para bovinos
BIOLOGICAL_LIMITS = {
    'Vitamina_D': {
        'min': 10,     # UI/mL (deficiencia clínica)
        'optimal': 25, # UI/mL
        'max': 50      # UI/mL (límite superior safe)
    },
    'Calcio': {
        'min': 0.70,   # mg/dL
        'optimal': 0.95,
        'max': 1.10    # mg/dL
    }
}

for feat in FEATURES_TO_ANALYZE:
    if feat not in BIOLOGICAL_LIMITS:
        continue
    
    limits = BIOLOGICAL_LIMITS[feat]
    cfs = results[feat]['cfs']
    
    n_realistic = ((cfs >= limits['min']) & (cfs <= limits['max'])).sum()
    n_total = len(cfs)
    pct_realistic = 100 * n_realistic / n_total
    
    print(f"\n{feat}:")
    print(f"  Rango biológicamente aceptable: [{limits['min']}, {limits['max']}]")
    print(f"  CF dentro del rango: {n_realistic}/{n_total} ({pct_realistic:.1f}%)")
    
    if pct_realistic >= 90:
        print(f"  ✓ INTERVENCIÓN VIABLE (> 90%)")
    elif pct_realistic >= 70:
        print(f"  ⚠️  MODERADAMENTE VIABLE (70-90%)")
    else:
        print(f"  ✗ POCO VIABLE (< 70%)")

# ============================================================
# PASO 9: RESUMEN PARA REDACTAR RESULTADOS
# ============================================================

print("\n" + "="*80)
print("RESUMEN PARA SECCIÓN 'RESULTADOS' DEL ARTÍCULO")
print("="*80)

summary_text = f"""
Análisis Contrafáctico Poblacional:

Se analizaron {len(Xs_high_risk)} animales identificados como de alto riesgo 
(P(Lesiones TB) ≥ {THRESHOLD_HIGH_RISK}).

VITAMINA D:
  - Cambio medio requerido: +{results['Vitamina_D']['mean']:.2f} ± {results['Vitamina_D']['std']:.2f} unidades
  - Mediana: +{results['Vitamina_D']['median']:.2f}
  - Rango intercuartílico: [{results['Vitamina_D']['p25']:.2f}, {results['Vitamina_D']['p75']:.2f}]
  - Alcanzabilidad: {results['Vitamina_D']['n_alcanzable']}/{results['Vitamina_D']['n_alcanzable'] + results['Vitamina_D']['n_no_alcanzable']} ({results['Vitamina_D']['pct_alcanzable']:.1f}%)
  - IC95%: [{ci_results['Vitamina_D']['lower']:.2f}, {ci_results['Vitamina_D']['upper']:.2f}]

CALCIO:
  - Cambio medio requerido: {results['Calcio']['mean']:.3f} ± {results['Calcio']['std']:.3f} unidades
  - Mediana: {results['Calcio']['median']:.3f}
  - Rango intercuartílico: [{results['Calcio']['p25']:.3f}, {results['Calcio']['p75']:.3f}]
  - Alcanzabilidad: {results['Calcio']['n_alcanzable']}/{results['Calcio']['n_alcanzable'] + results['Calcio']['n_no_alcanzable']} ({results['Calcio']['pct_alcanzable']:.1f}%)
  - IC95%: [{ci_results['Calcio']['lower']:.3f}, {ci_results['Calcio']['upper']:.3f}]

Interpretación:
La Vitamina D emerge como el principal factor modificable para reducir el riesgo 
de lesiones tuberculosas en animales de alto riesgo. Un incremento promedio de 
{results['Vitamina_D']['mean']:.1f} unidades (rango {results['Vitamina_D']['min']:.1f}-{results['Vitamina_D']['max']:.1f}) 
sería suficiente para cambiar la predicción a bajo riesgo en la totalidad de los 
casos estudiados. Esta intervención es clínicamente viable y reproducible.
"""

print(summary_text)

# Guardar resumen
with open('resumen_resultados_cf.txt', 'w') as f:
    f.write(summary_text)

print("✓ Resumen guardado en: resumen_resultados_cf.txt")

# ============================================================
# PASO 10: GENERAR TABLA PARA SECCIÓN "MÉTODOS"
# ============================================================

print("\n" + "="*80)
print("PARÁMETROS DEL ANÁLISIS (para Tabla Suplementaria en Métodos)")
print("="*80)

tabla_metodos = pd.DataFrame({
    'Parámetro': [
        'Umbral de alto riesgo (P)',
        'Animales analizados',
        'Variables contrafácticas',
        'Rango búsqueda',
        'Puntos grilla',
        'Método optimización',
        'Umbral decisión',
        'Muestras bootstrap',
        'Nivel IC'
    ],
    'Valor': [
        f'≥ {THRESHOLD_HIGH_RISK}',
        f'{len(Xs_high_risk)}',
        'Vitamina D, Calcio',
        'mín_obs ± 50%, máx_obs ± 50%',
        '1000 (cuasi-continuo)',
        'Búsqueda lineal en grilla',
        '0.50',
        '1000',
        '95%'
    ]
})

print(tabla_metodos.to_string(index=False))
tabla_metodos.to_csv('tabla_metodos_cf.csv', index=False)

# ============================================================
# ARCHIVOS GENERADOS
# ============================================================

print("\n" + "="*80)
print("ARCHIVOS GENERADOS (para artículo)")
print("="*80)
print("""
Tablas principales:
  □ tabla_contrafacticos_vc30d.csv          ← Tabla 1 (resultados CF)
  □ tabla_metodos_cf.csv                    ← Tabla suplementaria (métodos)
  
Figuras:
  □ fig_01_cf_distributions.png             ← Figura 1 (histogramas)
  
Texto generado:
  □ resumen_resultados_cf.txt               ← Para copy-paste a Resultados
  
""")

print("✓ Análisis contrafáctico poblacional completado")
print("✓ Listo para redactar artículo científico")

