# Esquema CV-animal — TB bovina (set complementario)

Segundo conjunto de entregables que implementa **validación cruzada a nivel de animal**
(cada caso) **incluyendo la explotación como covariable** (efectos fijos), en contraste con
el *leave-one-farm-out* (LOFO) del set principal.

## Pregunta que responde este esquema

> Predicción para un **animal nuevo de una de las granjas ya conocidas**, condicionando en la
> explotación — frente al LOFO, que estima la predicción en una **granja nueva**.

Ambos son válidos; responden a escenarios de despliegue distintos.

## Contenido

| Archivo | Descripción |
|---|---|
| `tb_utils.py` | Módulo compartido (idéntico al del set principal). |
| `A_Modelado_CV_Animal.ipynb` | CV repetida estratificada (5×10) a nivel animal; tres familias por desenlace (biomarcadores / +granja / solo-granja); **test de valor incremental** (razón de verosimilitudes + ΔPR-AUC); calibración; SHAP condicional a la granja; LOO-CV. |
| `B_Conformal_CV_Animal.ipynb` | Predicción conforme a nivel animal (validez marginal estándar); eficiencia con/sin granja; cobertura condicional por explotación; intervalos para IDTC. |
| `Articulo_TB_bovina_CV_Animal.docx` | Artículo companion (IMRaD) con 6 figuras y 3 tablas. |
| `figures/` | figA1–figA3, figB1–figB3 (PNG 300 dpi). |

## Resultado central

La señal predecible reside en la **explotación**, no en los biomarcadores:

| Desenlace (métrica) | Biomarc. | +Granja | **Solo Granja** |
|---|---|---|---|
| Lesión (ROC-AUC) | 0,46 | 0,58 | **0,68** |
| Patrón (ROC-AUC) | 0,51 | 0,66 | **0,82** |
| IDTC (Spearman) | 0,14 | 0,15 | **0,27** |
| Gravedad (Spearman) | ≈0 | 0,23 | **0,43** |

Test de razón de verosimilitudes del aporte de los biomarcadores sobre la granja: p = 0,065
(no significativo). Añadir biomarcadores a la granja **no mejora** la predicción fuera de muestra.

## Cómo se relaciona con el set principal

- La "mejora" de este esquema frente al LOFO es **efecto de pertenencia a granja**, aprendible
  solo porque las mismas granjas están en *train* y *test*, y **no transfiere** a granjas nuevas.
- Ambos análisis convergen en la misma conclusión por vías distintas: los biomarcadores del
  hospedador no aportan valor predictivo; lo poco predecible es el **riesgo basal de granja**.

## Ejecución

```bash
pip install -r ../requirements.txt   # mismas versiones que el set principal
# Coloca BD.csv junto a los notebooks (o ajusta DATA_PATH en la primera celda)
jupyter lab   # ejecuta A -> B
```
Los notebooks se entregan ya ejecutados (resultados y figuras embebidos).

> El EDA y la estadística bivariante son comunes a ambos esquemas; ver el Notebook 1 del set
> principal (`01_EDA_Estadistica.ipynb`).
