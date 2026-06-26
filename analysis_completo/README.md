# Pipeline analítico — Tuberculosis bovina

Pipeline reproducible para predecir y, sobre todo, **cuantificar honestamente** el valor
predictivo de biomarcadores del hospedador y coinfecciones sobre cuatro desenlaces de TB
bovina (lesión, patrón, gravedad, IDTC) en una cohorte real de **106 animales / 4 explotaciones**.

## Contenido

| Archivo | Descripción |
|---|---|
| `tb_utils.py` | Módulo compartido (carga, limpieza, preprocesamiento sin fuga, CV agrupada, estadística, E-value, estética). **Requerido por los 3 notebooks.** |
| `01_EDA_Estadistica.ipynb` | EDA, datos faltantes, descriptivos y comparaciones bivariantes con FDR. |
| `02_Modelado_Interpretabilidad_Causalidad.ipynb` | Modelado (logística penalizada vs. XGBoost), CV anidada + leave-one-farm-out, calibración, SHAP/PDP/ALE/permutación/contrafactuales con estabilidad, y causalidad (DAG + DoWhy + E-value). |
| `03_Prediccion_Conforme.ipynb` | Predicción conforme (MAPIE): conjuntos e intervalos, cobertura marginal y condicional, eficiencia, split vs. CV-conformal. |
| `Articulo_TB_bovina.docx` | Artículo científico IMRaD (STROBE-Vet, referencias Vancouver/AMA) con tablas y las 14 figuras. |
| `requirements.txt` | Versiones exactas validadas. |
| `CHECKLIST_VALIDACION.md` | Supuestos críticos, datos faltantes para producción y recomendaciones de revisión por pares. |
| `figures/` | Las 14 figuras en PNG a 300 dpi. |

## Cómo ejecutar

```bash
pip install -r requirements.txt
# Coloca BD.csv junto a los notebooks (o ajusta DATA_PATH en la primera celda)
jupyter lab   # ejecuta 01 -> 02 -> 03 en orden
```

Los notebooks se entregan **ya ejecutados** (con resultados y figuras embebidos). Cada uno
es autosuficiente: carga `BD.csv`, lo limpia con `tb_utils` y genera sus figuras en `figures/`.

## Mensaje científico

En esta cohorte, las covariables evaluadas tienen **valor predictivo validado escaso o nulo**
fuera de la explotación de origen (ROC ≈ azar; E-value ≈ 1; conjuntos/intervalos conformes
amplios). El aporte es **metodológico y de cuantificación de incertidumbre**: el estudio es
**generador de hipótesis**, no una herramienta de predicción clínica individual. Véase el
checklist antes de cualquier uso o envío a revista.
