# Esquema ingenuo — TB bovina (set 3: ignorar la explotación)

Tercer y último conjunto de la serie. Implementa el enfoque **más sencillo y común por
defecto**: **ignorar la explotación por completo** (ni agrupamiento en la CV ni covariable) y
hacer **validación cruzada estándar a nivel de animal** con ajuste de modelos.

## Lugar en la serie (tríptico de validación)

| Set | Esquema | Pregunta | ROC lesión |
|---|---|---|---|
| 1 | Leave-one-farm-out (agrupar por granja) | ¿generaliza a **granja nueva**? | 0,47 |
| 2 | CV-animal + granja como covariable | ¿predice en **granja conocida**? | 0,58 |
| **3** | **CV-animal ignorando la granja** | *pooling* por defecto (ingenuo) | **0,46** |

## Contenido

| Archivo | Descripción |
|---|---|
| `tb_utils.py` | Módulo compartido (idéntico al resto de la serie). |
| `C_Modelado_Sin_Granja.ipynb` | CV repetida estratificada (5×10) a nivel animal, **solo biomarcadores**; los 4 desenlaces; calibración; SHAP + importancia por permutación; **asociación ingenua vs. ajustada** por granja (con E-value); **figura capstone de los 3 esquemas**. |
| `D_Conformal_Sin_Granja.ipynb` | Predicción conforme a nivel animal sin granja; cobertura marginal + **condicional por explotación** (la estructura ignorada reaparece); intervalos para IDTC. |
| `Articulo_TB_bovina_Sin_Granja.docx` | Artículo IMRaD (6 figuras, 3 tablas) que además **sintetiza los tres esquemas**. |
| `figures/` | figC1–figC4, figD1–figD2 (PNG 300 dpi). |

## Resultado central

- Ignorando la granja y con solo biomarcadores: **ROC ≈ 0,46–0,54** (azar), MCC ≤ 0;
  Spearman fuera de muestra ≈ 0 para IDTC y gravedad.
- Asociación ingenua de vitamina D: OR 0,78 [0,52–1,16], **E-value del IC = 1,00** (sin
  robustez). Ajustar por granja la **refuerza** ligeramente (0,61 [0,38–1,00]) → ignorar la
  granja puede **enmascarar** una débil asociación intra-granja.
- Cobertura conforme condicional por granja: 0,81–0,95 (dispersión 0,14) → la estructura
  ignorada reaparece como validez desigual entre explotaciones.

## Lectura conjunta

La conclusión sobre los biomarcadores (**sin valor predictivo validado**) es **robusta a los
tres esquemas**. La única ganancia de rendimiento aparece al incluir la explotación, y refleja
el **riesgo basal de granja**, no la biología del hospedador. El esquema ingenuo no generaliza
a granjas nuevas y oculta esa estructura: útil como contraste, desaconsejable como base de
decisiones.

## Ejecución

```bash
pip install -r ../requirements.txt
# BD.csv junto a los notebooks (o ajusta DATA_PATH)
jupyter lab   # ejecuta C -> D
```
Notebooks entregados ya ejecutados. El EDA es común; ver `01_EDA_Estadistica.ipynb` del set principal.
