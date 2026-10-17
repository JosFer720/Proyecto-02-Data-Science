# Fernando: Evaluación y comparación Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Evaluar los cuatro modelos congelados en la semana de prueba, medir generalización a ofertas no vistas y producir tablas, figuras, notebook, comparación y conclusiones reproducibles.

**Architecture:** `src/evaluacion.py` concentra métricas, tiempos, curvas, matrices y validación agrupada por oferta como funciones puras. El notebook 08 orquesta esas funciones, carga únicamente modelos `07_*`, escribe los artefactos contractuales y genera las figuras. El informe consume exclusivamente cifras producidas por esos artefactos.

**Tech Stack:** Python 3.12, pandas 2.2.3, numpy 2.1.2, scikit-learn 1.6.1, scipy 1.15.1, joblib 1.4.2, matplotlib, pyarrow, pytest, Jupyter.

**Spec:** `C:/Users/anna2/Downloads/plan_proyecto2_resultados_y_visualizaciones.md`, secciones 2, 3.4, 5 y 6.

## Global Constraints

- La semana de prueba es 2013-04-24 a 2013-04-30 y no interviene en ajuste ni selección.
- La evaluación principal usa exactamente los cuatro modelos congelados `07_*` y `cargar_modelos()`.
- Solo la validación por ofertas reentrena modelos temporales; usa al menos cinco divisiones con ofertas disjuntas.
- AUC es la métrica primaria; también se reportan precisión, sensibilidad, F1, tiempos y matriz de confusión.
- Los umbrales de clasificación vienen del manifiesto y no se optimizan con la semana de prueba.
- Figuras con paleta Okabe-Ito, línea de azar y etiquetas/unidades directas.
- No modificar artefactos propiedad de Joel, Nery o Kevin.

## Review Focus

- Una partición con una sola clase no produce AUC engañoso: se valida o se informa explícitamente.
- Los tiempos separan entrenamiento registrado de predicción medido, sin reentrenar modelos congelados.
- La matriz de confusión mantiene el orden TN, FP, FN, TP y las métricas usan `zero_division=0`.
- Las divisiones agrupadas verifican intersección vacía de ofertas entre entrenamiento y prueba.
- El entorno incompatible con el manifiesto falla antes de producir resultados, en vez de cargar joblib silenciosamente.

---

### Task 1: Biblioteca de evaluación

**Files:**
- Create: `src/evaluacion.py`
- Create: `tests/test_evaluacion.py`

**Interfaces:**
- Consumes: objetos con `predict_proba`, etiquetas binarias y metadatos del manifiesto.
- Produces: `metricas_binarias`, `curva_roc`, `evaluar_modelo_congelado`, `validar_por_ofertas` y tablas normalizadas.

- [ ] Escribir pruebas para métricas, umbral, tiempos, ROC y ofertas disjuntas.
- [ ] Ejecutar pruebas y confirmar fallo por módulo ausente.
- [ ] Implementar API mínima con tipos y validaciones.
- [ ] Ejecutar `pytest tests/test_evaluacion.py -q` y confirmar éxito.

### Task 2: Notebook y artefactos contractuales

**Files:**
- Create: `notebooks/08_evaluacion_y_comparacion.ipynb`
- Create: `results/tables/08_metricas_prueba.csv`
- Create: `results/tables/08_predicciones_prueba.parquet`
- Create: `results/tables/08_validacion_por_ofertas.csv`

**Interfaces:**
- Consumes: biblioteca de Task 1, dataset etiquetado, modelos y manifiesto `07_*`.
- Produces: métricas y predicciones para la app y el informe.

- [ ] Crear notebook con validaciones de entorno y separación temporal.
- [ ] Evaluar los modelos congelados una vez en la semana de prueba.
- [ ] Ejecutar cinco divisiones agrupadas por oferta con hiperparámetros del manifiesto.
- [ ] Guardar tablas y comprobar esquemas, recuentos y ausencia de ofertas compartidas.

### Task 3: Figuras de comparación

**Files:**
- Create: `results/figures/08_curvas_roc.png`
- Create: `results/figures/08_auc_modelos.png`
- Create: `results/figures/08_tiempo_vs_auc.png`
- Create: `results/figures/08_auc_ofertas_no_vistas.png`
- Create: `results/figures/08_matrices_confusion.png`

**Interfaces:**
- Consumes: las tres tablas de Task 2.
- Produces: cinco figuras estáticas legibles y consistentes.

- [ ] Generar las cinco figuras desde los artefactos guardados.
- [ ] Verificar que existen, no están vacías y tienen dimensiones adecuadas.

### Task 4: Informe de Fernando

**Files:**
- Create: `informe/resultados/07_comparacion_de_algoritmos.md`
- Create: `informe/resultados/09_conclusiones.md`

**Interfaces:**
- Consumes: cifras finales de Tasks 2 y 3 y objetivos del plan fuente.
- Produces: comparación, elección justificada y conclusiones que cierran objetivos 1 y 3 sin inventar el objetivo 5.

- [ ] Redactar comparación de efectividad, costo y errores con cifras exactas.
- [ ] Redactar conclusiones por objetivo y marcar dependencias ajenas aún pendientes.

### Task 5: Verificación y revisión

- [ ] Ejecutar el notebook desde cero en el entorno exacto del manifiesto.
- [ ] Ejecutar `pytest -q` y compilar módulos Python.
- [ ] Revisar tablas contra cálculos independientes y notebook sin errores.
- [ ] Solicitar revisión final de código y corregir hallazgos importantes.
