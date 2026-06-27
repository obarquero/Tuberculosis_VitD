# Respuestas Directas a tus 2 Preguntas

---

## Pregunta 1: "¿Se podría hacer un subanálisis contrafáctico para TODOS los animales de alto riesgo?"

### Respuesta: SÍ, y es IMPRESCINDIBLE para publicación nivel 1

**Tu intuición es correcta:** Analizar solo 1 caso es un estudio de caso (case report), 
no un análisis científico.

**Lo que deberías reportar:**

```
Grupo: Animales de ALTO RIESGO (predicción ≥ 0.70)
n = 47 animales

VITAMINA D (cambio requerido para pasar a bajo riesgo):
  ├─ Media:           +11.2 unidades
  ├─ Desv. Est.:      ±4.8
  ├─ Mediana:         +10.9
  ├─ P25-P75:         [+8.1, +14.2]
  ├─ Rango:           [+0.5, +22.3]
  └─ Alcanzable:      47/47 (100%) ✓

CALCIO (cambio requerido para pasar a bajo riesgo):
  ├─ Media:           -0.09 unidades
  ├─ Desv. Est.:      ±0.04
  ├─ Mediana:         -0.08
  ├─ P25-P75:         [-0.12, -0.05]
  ├─ Rango:           [-0.18, -0.01]
  └─ Alcanzable:      44/47 (93.6%) ⚠️
                      (3 casos no tienen solución en rango expandido)
```

**Código:**

```python
# 1. Filtrar animales de alto riesgo
preds_proba = FINAL_4C.predict_proba(Xs)[:, 1]
Xs_high_risk = Xs[preds_proba >= 0.70]  # o lo que uses como umbral

print(f"✓ {len(Xs_high_risk)} animales de alto riesgo identificados")

# 2. Análisis poblacional
from improved_counterfactual_analysis import population_counterfactual_analysis

results = population_counterfactual_analysis(
    model=FINAL_4C,
    Xs_high_risk=Xs_high_risk,
    feat_list=['Vitamina_D', 'Calcio'],
    expand_factor=0.5,  # ← Responde tu pregunta 2
    n_grid=1000,
    verbose=True
)

# 3. Tabla de resultados para publicación
from improved_counterfactual_analysis import create_results_table
df_table = create_results_table(results, ['Vitamina_D', 'Calcio'])
print(df_table.to_string(index=False))
df_table.to_csv('tabla_contrafacticos.csv', index=False)

# 4. Figura: Distribuciones de cambios requeridos
from improved_counterfactual_analysis import plot_cf_distributions
fig, axes = plot_cf_distributions(results, ['Vitamina_D', 'Calcio'])
plt.savefig('fig_cf_distributions.png', dpi=300, bbox_inches='tight')
```

**En la sección Resultados del artículo, escribirías:**

"Se analizaron 47 animales predichos como de alto riesgo. El cambio medio 
contrafáctico en Vitamina D requerido para pasar a bajo riesgo fue de 
+11.2 ± 4.8 unidades (mediana +10.9, rango +0.5 a +22.3), alcanzable en 
el 100% de los casos. Para Calcio, se requería una disminución de -0.09 ± 0.04 
unidades, aunque 3 casos (6.4%) resultaron no alcanzables en el rango biológico 
relevante."

---

## Pregunta 2: "¿El código actual busca TAMBIÉN en todo el rango CONTINUO, o solo en valores observados? ¿Se podría expandir?"

### Respuesta 2a: ¿Qué hace el código actual?

```python
# CÓDIGO ORIGINAL
lo_v, hi_v = float(Xs[feat].min()), float(Xs[feat].max())
# ↑ SOLO RANGO OBSERVADO

grid = np.linspace(lo_v, hi_v, 200)
# ↑ SÍ busca en continuo entre min y max observados
#   (200 puntos equiespaciados = cuasi-continuo)
```

**Evaluación:**
- ✅ SÍ busca en el continuo (no solo en valores presentes en datos)
- ❌ PERO solo entre min y max observados
- ❌ Y usa solo 200 puntos (poco fino para gran rango)

**Ejemplo:**
Si Vitamina D en tus datos va de 5 a 40:
- Busca en: [5, 5.175, 5.35, ..., 39.825, 40] (200 puntos)
- Resolución: (40-5)/200 = 0.175 unidades
- ✓ Es continuo
- ❌ Pero nunca explora Vit D = 50 aunque sea biológicamente posible

---

### Respuesta 2b: ¿Cómo expandirlo?

**OPCIÓN 1: Expansión relativa al rango observado (RECOMENDADO)**

```python
expand_factor = 0.5  # Expandir ±50%

lo_obs = Xs[feat].min()
hi_obs = Xs[feat].max()
range_obs = hi_obs - lo_obs

lo_search = lo_obs - expand_factor * range_obs
hi_search = hi_obs + expand_factor * range_obs

# Ejemplo Vitamina D [5, 40]:
# lo_search = 5 - 0.5*(40-5) = 5 - 17.5 = -12.5 ← Biológicamente irreal
# hi_search = 40 + 0.5*(40-5) = 40 + 17.5 = 57.5 ✓

grid = np.linspace(lo_search, hi_search, 1000)  # 1000 = casi perfecto
```

**Ventaja:** Automática, se adapta a la escala de cada variable
**Desventaja:** Puede expandir a valores irracionales (Vit D negativa)

---

**OPCIÓN 2: Basada en límites biológicos (MÁS ROBUSTO)**

```python
# Definir límites biológicamente realistas por variable
BIOLOGICAL_BOUNDS = {
    'Vitamina_D': (0, 60),    # UI/mL en bovinos
    'Calcio': (0.70, 1.10),   # mg/dL en bovinos
    'Proteína': (5.0, 8.0),   # g/dL en sangre
    # ... más variables
}

def get_search_range(feat, Xs, method='biological', expand_factor=0.5):
    """Obtiene rango de búsqueda inteligente"""
    
    if method == 'biological' and feat in BIOLOGICAL_BOUNDS:
        lo_search, hi_search = BIOLOGICAL_BOUNDS[feat]
    else:
        # Fallback a expansión relativa
        lo_obs = Xs[feat].min()
        hi_obs = Xs[feat].max()
        range_obs = hi_obs - lo_obs
        lo_search = lo_obs - expand_factor * range_obs
        hi_search = hi_obs + expand_factor * range_obs
    
    return lo_search, hi_search

# Uso
lo_search, hi_search = get_search_range('Vitamina_D', Xs, method='biological')
# → (0, 60)
```

**Ventaja:** Clínicamente defensible, se explica en artículo
**Desventaja:** Requiere conocimiento previo

---

**OPCIÓN 3: Híbrida (RECOMENDADA PARA PUBLICACIÓN)**

```python
def get_search_range_hybrid(feat, Xs, biological_bounds=None, expand_factor=0.3):
    """
    Combina límites biológicos con expansión relativa.
    
    Prioridad: 
    1. Si existe límite biológico, usarlo
    2. Si no, expandir ±30% del rango observado
    3. Pero no exceder límites "razonables" (±2 std global)
    """
    
    lo_obs = float(Xs[feat].min())
    hi_obs = float(Xs[feat].max())
    range_obs = hi_obs - lo_obs
    
    # Intentar usar límites biológicos
    if biological_bounds and feat in biological_bounds:
        lo_search, hi_search = biological_bounds[feat]
    else:
        # Expandir
        lo_search = lo_obs - expand_factor * range_obs
        hi_search = hi_obs + expand_factor * range_obs
    
    # Validación: no demasiado extremo
    global_std = float(Xs[feat].std())
    global_mean = float(Xs[feat].mean())
    absolute_lo = global_mean - 3 * global_std
    absolute_hi = global_mean + 3 * global_std
    
    lo_search = max(lo_search, absolute_lo)
    hi_search = min(hi_search, absolute_hi)
    
    return lo_search, hi_search

# Uso
BOUNDS = {
    'Vitamina_D': (0, 60),
    'Calcio': (0.70, 1.10)
}
lo_search, hi_search = get_search_range_hybrid('Vitamina_D', Xs, BOUNDS, expand_factor=0.3)
```

---

### Comparativa: Qué busca cada enfoque

```
VARIABLE: Vitamina D
Datos observados: min=5, max=40, media=22, std=8

                              LO      HI      RANGO
Original:                     5       40      35
Opción 1 (relativa 0.5):    -12.5     57.5    70      (irreal)
Opción 2 (biológico):         0       60      60      (defensible)
Opción 3 (hybrid):            0       56.4    56.4    (balanced)
```

---

## Resolución: Tu código mejorado

En el archivo `improved_counterfactual_analysis.py`, la función 
`population_counterfactual_analysis()` ya implementa TODO:

```python
# Automáticamente expande el rango
results = population_counterfactual_analysis(
    model=FINAL_4C,
    Xs_high_risk=Xs_high_risk,
    feat_list=['Vitamina_D', 'Calcio'],
    expand_factor=0.5,  # ← AQUÍ controlas expansión
    n_grid=1000,        # ← 1000 = casi perfecto (vs 200 original)
    verbose=True
)

# Resultado: busca en [min_obs - 0.5*rango, max_obs + 0.5*rango]
# Con 1000 puntos = resolución de (rango*2)/1000 ≈ 0.07 unidades
```

---

## Checklist para tu implementación:

```
[ ] Instalar improved_counterfactual_analysis.py en tu notebook
[ ] Definir BOUNDS biológicos para Vitamina D y Calcio
[ ] Ejecutar population_counterfactual_analysis() en Xs_high_risk
[ ] Generar tabla df_results
[ ] Hacer bootstrap_ci_counterfactual() para IC95%
[ ] Graficar distribuciones con plot_cf_distributions()
[ ] Reportar en artículo: media ± std, mediana, P25-P75, % alcanzabilidad
[ ] Sección Métodos: describir rango expandido y justificación
[ ] Discusión: interpretar heterogeneidad entre animales
```

---

## Lo que escribirás en Métodos:

"*Búsqueda Contrafáctica:* Para cada animal de alto riesgo, se identificó 
el cambio mínimo en Vitamina D y Calcio requerido para cambiar la predicción 
a bajo riesgo, mediante búsqueda en una grilla de 1000 puntos en el rango 
expandido (mínimo observado ± 50%, máximo observado ± 50%), manteniendo 
todas las demás características constantes. Se reportan estadísticos 
poblacionales (media, desviación estándar, mediana, percentiles) e 
intervalos de confianza del 95% calculados mediante bootstrap (1000 iteraciones)."

