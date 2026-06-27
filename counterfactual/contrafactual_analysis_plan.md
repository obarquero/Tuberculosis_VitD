# Análisis Contrafáctico Profundo para Publicación Científica
## Plan Metodológico: VC30D, Lesiones TB en Animales

---

## 1. DIAGNÓSTICO DEL CÓDIGO ACTUAL

### ¿Qué hace `np.linspace(lo, hi, n=500)`?

```python
lo_v, hi_v = float(Xs[feat].min()), float(Xs[feat].max())  # ← Solo rango OBSERVADO
grid = np.linspace(lo_v, hi_v, n)  # ← Busca en 500 puntos equiespaciados
```

**Situación actual:**
- ✅ SÍ busca en todo el continuo entre min y max observados
- ❌ NO extrapolá más allá de los datos observados
- ❌ Solo reporta UN caso per grupo (high_row, low_row)
- ❌ No calcula distribuciones/intervalos de confianza
- ❌ Sin análisis de sensibilidad del algoritmo

---

## 2. MEJORAS PROPUESTAS (Nivel 1)

### A. **Análisis Poblacional de Contrafácticos**

En lugar de 1 animal representativo → **analizar TODOS los de alto riesgo**

```
Para cada animal i en ALTO RIESGO (pred > 0.7):
  - Buscar CF en Vitamina D
  - Registrar Δ_VitD_i = CF_i - Actual_i
  
Reportar:
  - Media(Δ_VitD) ± std
  - Mediana, P25, P75
  - Distribución (histograma)
  - Casos "no alcanzables"
```

**Beneficio:** Estadísticamente defensible, muestra heterogeneidad

### B. **Extrapolación Controlada**

Expandir rango de búsqueda con **límites biológicamente razonables**

```
Opción 1 (Conservadora):
  lo_cf = Xs[feat].min() - 0.5 * (Xs[feat].max() - Xs[feat].min())
  hi_cf = Xs[feat].max() + 0.5 * (Xs[feat].max() - Xs[feat].min())
  
Opción 2 (Basada en Literatura):
  # Valores de Vitamina D en bovinos: típicamente 0-60 UI/mL
  lo_cf, hi_cf = 0, 60
  
Opción 3 (Percentil-based):
  # Expandir ±20% desde los percentiles 5 y 95
```

**Beneficio:** Identifica si el CF es "realista" o requiere intervención extrema

### C. **Búsqueda de Contrafácticos Mejorada**

Reemplazar búsqueda lineal por **optimización más robusta**

```python
# Método 1: Binary Search (más eficiente)
# Método 2: Minimización de distancia a umbral (0.5)
# Método 3: Bootstrapping para intervalos de confianza
```

### D. **Análisis de Sensibilidad**

Validar robustez:
- ¿Sensible a los hiperparámetros del modelo?
- ¿Varían CF si reentrenamos el modelo?
- ¿Qué ocurre si incluimos incertidumbre de predicción?

---

## 3. TABLA DE RESULTADOS PROPUESTA

Para cada variable contrafactual:

| Grupo | N | Media Δ | Std | Mediana | P25-P75 | No-Alcanzable | Rango impl. |
|-------|---|---------|-----|---------|---------|---------------|------------|
| Alto riesgo (Vit D) | 47 | +11.2 | 4.8 | 10.9 | [8.1-14.2] | 0 (0%) | [1.2-29.4] |
| Alto riesgo (Calcio) | 47 | -0.09 | 0.04 | -0.08 | [-0.12,-0.05] | 3 (6.4%) | [-0.18-0.01] |

**Interpretación:**
- CF es alcanzable en 100% de casos de Vit D → **intervención viable**
- Calcio es secundario pero con 6% no-alcanzables → **menos robusto**

---

## 4. FIGURAS RECOMENDADAS

### Figura A: Distribuciones de cambios contrafácticos
```
Subplot 1: Histograma de Δ Vitamina D (alto riesgo)
Subplot 2: Histograma de Δ Calcio (alto riesgo)
- Añadir línea de media/mediana
- Anotar N, media, std
```

### Figura B: "Sensitivity Surface" 3D (opcional)
```
Eje X: Vitamina D
Eje Y: Calcio
Eje Z: P(Lesiones TB)
- Mostrar puntos actuales de casos de alto/bajo riesgo
- Mostrar superficie de decisión (0.5)
```

### Figura C: Curvas de sensibilidad POR SUBGRUPO
```
- Vit D vs P(Lesiones) para:
  - Alto riesgo (n=47)
  - Bajo riesgo (n=25)
  - Por edad (categorizado: <1yr, 1-3yr, >3yr)
  - Por sexo
```

---

## 5. ANÁLISIS DE HETEROGENEIDAD

Pregunta: **¿El cambio necesario de Vit D depende de otras características?**

Subgrupos:
- Por edad
- Por sexo
- Por estado basal de Calcio
- Por predicción basal (p > 0.9 vs 0.7-0.9)

Reportar ANOVA o Kruskal-Wallis para Δ VitD entre grupos

---

## 6. VALIDACIÓN Y ROBUSTEZ

### A. Cálculo de Intervalos de Confianza (Bootstrap)

```python
# Para cada animal i de alto riesgo:
for b in range(1000):
    # Resample con reemplazo: muestra_boot
    # Reentrenar modelo: model_b
    # Calcular CF: cf_b
    # Almacenar Δ_b
    
# IC 95%: percentiles 2.5 y 97.5 de Δ_b
```

### B. Análisis de Sensibilidad del Modelo

```
Variar:
- Threshold de decisión (0.5 → 0.45, 0.55)
- Penalidad de regularización del modelo
- Features incluidos en el modelo
- Tipo de modelo (si disponible: RF, XGBoost)

Resultado: ¿cambia el CF sustancialmente?
```

### C. Validación de Realismo

```python
# Para cada CF calculado:
if cf_value > medical_upper_limit:
    flag = "SUPRAFISIOLÓGICO"
elif cf_value < medical_lower_limit:
    flag = "SUBÓPTIMO"
else:
    flag = "REALISTA"
```

---

## 7. SECCIÓN "MÉTODOS" PARA ARTÍCULO

### Análisis Contrafáctico

Se implementó un algoritmo de búsqueda contrafáctica para identificar el cambio 
mínimo en características nutricionales requerido para cambiar la predicción de 
riesgo de alto a bajo (o viceversa).

Para cada animal i con predicción de alto riesgo (p > 0.70):

1. **Búsqueda en el rango ampliado:** Se varió cada característica nutricional j 
   en el rango [X_j^min - 0.5·R_j, X_j^max + 0.5·R_j], donde R_j es el rango 
   observado.

2. **Criterio de parada:** Se registró el valor contrafáctico (CF) como el primero 
   donde P(Lesiones TB | X'_j) ≤ 0.50 (umbral de decisión).

3. **Cambio requerido:** Δ_j = CF_j - X_j^actual

4. **Estadísticos poblacionales:** Se calculó media, mediana, desviación estándar 
   e intervalos de confianza (IC 95%) del cambio requerido mediante bootstrap (1000 iteraciones).

5. **Análisis de heterogeneidad:** Se examinó si Δ_j varía significativamente por 
   subgrupos (edad, sexo, estado basal de otras variables) mediante ANOVA/Kruskal-Wallis.

---

## 8. PSEUDOCÓDIGO MEJORADO

```python
# ============ ANÁLISIS POBLACIONAL ============

def population_counterfactual_analysis(model, Xs_high_risk, feat_list, 
                                       expand_factor=0.5, n_grid=1000, 
                                       n_bootstrap=1000):
    """
    Calcula CF poblacionales con IC95% vía bootstrap
    
    Parámetros:
    -----------
    expand_factor : float
        Expansión del rango más allá de observados (0.5 = ±50%)
    n_grid : int
        Puntos para búsqueda (1000 = cuasi-continuo)
    n_bootstrap : int
        Iteraciones para IC
    """
    
    results = {}
    
    for feat in feat_list:
        lo_obs, hi_obs = Xs_high_risk[feat].min(), Xs_high_risk[feat].max()
        range_obs = hi_obs - lo_obs
        
        # Expandir rango
        lo_search = lo_obs - expand_factor * range_obs
        hi_search = hi_obs + expand_factor * range_obs
        
        # Crear grilla fina
        grid = np.linspace(lo_search, hi_search, n_grid)
        
        # Para cada animal de alto riesgo
        deltas = []
        for idx, row in Xs_high_risk.iterrows():
            cf = find_cf_refined(model, row, feat, grid, target_cls=0)
            if cf is not None:
                delta = cf - row[feat]
                deltas.append(delta)
        
        # Estadísticos
        deltas = np.array(deltas)
        results[feat] = {
            'mean': deltas.mean(),
            'std': deltas.std(),
            'median': np.median(deltas),
            'p25': np.percentile(deltas, 25),
            'p75': np.percentile(deltas, 75),
            'n_alcanzable': len(deltas),
            'n_no_alcanzable': len(Xs_high_risk) - len(deltas),
            'deltas': deltas
        }
    
    return results

# ============ BOOTSTRAP PARA IC95% ============

def bootstrap_counterfactual_ci(model, Xs_high_risk, feat, 
                                n_bootstrap=1000, ci=95):
    """Intervalos de confianza bootstrap"""
    
    bootstrap_deltas = []
    
    for b in range(n_bootstrap):
        # Resample
        Xs_boot = Xs_high_risk.sample(n=len(Xs_high_risk), replace=True)
        
        # Calcular CF en muestra bootstrap
        deltas_b = []
        for idx, row in Xs_boot.iterrows():
            cf = find_cf_refined(model, row, feat, grid, target_cls=0)
            if cf is not None:
                deltas_b.append(cf - row[feat])
        
        if deltas_b:
            bootstrap_deltas.append(np.mean(deltas_b))
    
    # Percentiles
    alpha = 100 - ci
    lower = np.percentile(bootstrap_deltas, alpha/2)
    upper = np.percentile(bootstrap_deltas, 100 - alpha/2)
    
    return {'lower': lower, 'upper': upper, 'point': np.mean(bootstrap_deltas)}

# ============ ANÁLISIS DE HETEROGENEIDAD ============

def heterogeneity_analysis(Xs_high_risk, deltas_dict, groupby_var):
    """Compara CF entre subgrupos"""
    
    stats = {}
    groups = Xs_high_risk[groupby_var].unique()
    
    for group in groups:
        mask = Xs_high_risk[groupby_var] == group
        deltas_group = deltas_dict['deltas'][mask]
        stats[group] = {
            'n': mask.sum(),
            'mean': deltas_group.mean(),
            'std': deltas_group.std()
        }
    
    # ANOVA / Kruskal-Wallis
    from scipy.stats import f_oneway
    group_deltas = [deltas_dict['deltas'][Xs_high_risk[groupby_var] == g] 
                    for g in groups]
    f_stat, p_value = f_oneway(*group_deltas)
    
    return stats, {'f_stat': f_stat, 'p_value': p_value}
```

---

## 9. CHECKLIST PARA PUBLICACIÓN

- [ ] ¿Incluyen todos los animales de alto riesgo, no solo casos?
- [ ] ¿Hay IC95% o error bars?
- [ ] ¿Se reportan casos no-alcanzables?
- [ ] ¿Se justifica el rango expandido?
- [ ] ¿Hay análisis de heterogeneidad?
- [ ] ¿Se valida robustez a cambios del modelo?
- [ ] ¿Figuras muestran distribuciones, no casos únicos?
- [ ] ¿Tabla de resultados con estadísticos poblacionales?
- [ ] ¿Discusión de limitaciones (extrapolación, colinealidad)?
- [ ] ¿Cómo se alinean CF con recomendaciones de veterinaria?

---

## 10. REFERENCIAS METODOLÓGICAS

- **Contrafácticos en ML:** Wachter et al. (2017). "Counterfactual Explanations Without Opening the Black Box"
- **Causal Inference:** Rotnitzky & Robins (1995). "Semiparametric estimation of models for natural experiments"
- **Heterogeneidad de tratamiento:** Athey & Wager (2019). "Estimating treatment effects with causal forests"
- **Validación de modelos predictivos:** Harrell et al. (1996). "Regression models for ordinal outcomes"

