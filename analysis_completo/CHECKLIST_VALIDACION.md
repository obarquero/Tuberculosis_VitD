# CHECKLIST DE VALIDACIÓN — Pipeline TB bovina

> Documento de cierre del pipeline (3 notebooks + artículo). Reúne los **supuestos
> críticos**, la **información que faltaría para un uso en producción** y las
> **recomendaciones de revisión por pares** antes del envío a revista.
> Dataset real validado: `BD.csv` — **106 animales, 4 explotaciones**.

---

## (a) Supuestos críticos `[ASUMIR: ...]`

1. **`[ASUMIR: Expl = explotación/granja es la unidad de agrupamiento natural]`**
   Toda la validación honesta (leave-one-farm-out, cobertura condicional) descansa en
   que `Expl` captura el manejo, la exposición y la nutrición compartidos. Con **solo 4
   niveles**, la varianza entre-granja domina y la generalización a granjas nuevas es muy
   incierta. *Acción:* confirmar que no hay sub-estructura adicional (lote, parcela, año).

2. **`[ASUMIR: DAG causal de la sección 2.7]`**
   La estructura `Expl → {VitD, Calcio, Lesión, Edad}`, `Edad → {VitD, Lesión}`,
   `VitD/Calcio → Lesión` es una **hipótesis de trabajo**. El conjunto de ajuste por
   puerta trasera, el efecto estimado y el E-value dependen de ella. *Acción:* validar el
   DAG con el equipo clínico/epidemiológico; explorar DAGs alternativos.

3. **`[ASUMIR: IDTC tratada como recuento]`** (regresión de Poisson).
   Si IDTC es una medida continua (p. ej., mm) o un índice acotado, debería re-modelarse
   (gamma, beta o regresión robusta). *Acción:* confirmar la definición y unidad de IDTC.

4. **`[ASUMIR: Score_lesional ordinal con discontinuidad {0,1,2,4,5} (sin 3)]`**
   El 0 coincide con ausencia de lesión, mezclando «presencia» y «gravedad». *Acción:*
   decidir si la gravedad debe modelarse solo en lesionados; verificar la ausencia real
   de la categoría 3 (no un error de codificación).

5. **`[ASUMIR: Patron_lesiones definido solo en animales lesionados]`**
   La ausencia es **estructural** (41 NA ≡ 41 no lesionados), no aleatoria; no se imputa.
   *Acción:* confirmar que ningún lesionado quedó sin clasificar por error.

6. **`[ASUMIR: serología binaria Positivo/Negativo; raza Angus (n=1) colapsada en «Otra»]`**
   Piroplasma es ~86% positivo, sexo 98F/8M: predictores casi degenerados. *Acción:*
   valorar excluir sexo y/o Piroplasma del modelado por baja variabilidad.

7. **`[ASUMIR: umbral de clasificación 0,5 para sensibilidad/especificidad]`**
   El punto operativo clínico debe fijarse según el coste relativo de falsos
   negativos/positivos. *Acción:* definir el umbral con el cliente.

8. **`[ASUMIR: aproximación RR ≈ √OR para el E-value]`** (outcome frecuente, prev. 0,61).
   Es una aproximación estándar pero conservadora. *Acción:* reportarla como tal.

---

## (b) Información/datos faltantes para una ejecución «real» (producción)

- **Validación prospectiva multi-explotación.** Con 4 granjas no es posible estimar de
  forma fiable la generalización. Se necesita un conjunto de **validación externa** con
  varias explotaciones nuevas e idealmente otra región/temporada.
- **Mayor tamaño muestral.** n=106 limita potencia y CV anidada completa. Un objetivo
  razonable para predicción sería ≥ 10–20 eventos por predictor y ≥ 10–15 explotaciones.
- **Metadatos de explotación.** Variables de manejo, dieta, densidad, co-pastoreo, estado
  vacunal y carga de exposición permitirían modelar (o ajustar) el confusor dominante.
- **Definición operativa y trazabilidad** de IDTC, Score y patrón (protocolo de medición,
  observador, fecha) para descartar variabilidad de medida y sesgos.
- **Recalibración por explotación** y **predicción conforme adaptada a grupos**
  (Mondrian/agrupada) antes de cualquier despliegue, dada la no-intercambiabilidad.
- **Plan de monitorización** de deriva (data drift) y de recalibración periódica si el
  modelo llegara a usarse operativamente.
- **Diccionario de datos versionado** y control de versiones de datos/código (hashes).

---

## (c) Recomendaciones para la revisión por pares (antes del envío)

1. **Enmarcar como estudio cautelar / generador de hipótesis**, no predictivo. El mensaje
   central —valor predictivo validado escaso o nulo fuera de explotación— es robusto y
   honesto; evitar cualquier sobreinterpretación causal.
2. **Reportar según STROBE-Vet**: diagrama de flujo de animales, tabla de características,
   tamaños de efecto con IC, manejo de datos faltantes y estrategia de validación.
3. **Justificar explícitamente** la elección de leave-one-farm-out y la brecha frente a la
   CV que ignora la granja (resultado metodológico clave, Figura 5).
4. **Acompañar la interpretabilidad de sus límites**: dejar claro que SHAP/PDP/ALE
   describen el modelo en muestra; incluir la importancia por permutación LOFO y la
   estabilidad SHAP como contrapeso.
5. **Análisis de sensibilidad causal**: presentar el E-value y las refutaciones; añadir,
   si es posible, DAGs alternativos y un análisis de confusores no medidos.
6. **Predicción conforme**: explicar la cobertura marginal vs. condicional y citar la
   no-intercambiabilidad por granja; preferir CV-conformal y mostrar la inestabilidad de
   split-conformal.
7. **Disponibilidad de datos y código**: depositar los notebooks y el módulo de utilidades
   en un repositorio con DOI; completar las declaraciones `[ASUMIR: ...]` (ética,
   financiación, conflictos, CRediT).
8. **Revisión estadística independiente** recomendada, dada la combinación de métodos
   (validación agrupada, causalidad observacional, conformal) en muestra pequeña.
9. **Lenguaje calibrado**: sustituir cualquier afirmación de «predicción» por «asociación
   no validada» donde corresponda; verificar la coherencia numérica entre los tres
   notebooks, las tablas y el texto del artículo (ya verificada en esta entrega).

---

## Coherencia numérica verificada (resumen)

| Magnitud | Valor | Fuente |
|---|---|---|
| n / explotaciones | 106 / 4 | NB1 |
| Prevalencia de lesión | 0,613 | NB1 |
| Patrón generalizado (lesionados) | 0,369 (24/65) | NB1 |
| Asociaciones bivariantes (FDR) | ninguna significativa | NB1 |
| Lesión — ROC LOFO (LogReg / XGB) | 0,46 / 0,59 | NB2 |
| Patrón — ROC LOFO | 0,35 | NB2 |
| IDTC — Spearman LOFO | 0,05 | NB2 |
| Estabilidad SHAP (Spearman medio) | 0,83 | NB2 |
| Vitamina D — OR/SD ajustado | 0,59 (IC95% 0,36–0,98) | NB2 |
| E-value (punto / límite IC) | 1,92 / 1,12 | NB2 |
| Conforme lesión 80% (cobertura / tamaño) | 0,92 / 1,73 de 2 | NB3 |
| Conforme IDTC 80% (cobertura / anchura) | 0,97 / 18 de 32 | NB3 |
| Cobertura condicional por granja | 0,81–0,95 | NB3 |

*Fin del checklist.*
