# Comparación de los algoritmos

## Protocolo de evaluación

Los cuatro pipelines ajustados se evaluaron una sola vez sobre la semana de prueba, del 24 al 30 de abril de 2013, formada por 35,499 clientes. Esta semana no intervino en la búsqueda de hiperparámetros ni en la selección de umbrales. Cada modelo se cargó desde el archivo congelado registrado en `results/modelos/manifiesto.json`; su huella SHA-256 y la versión de scikit-learn se comprobaron antes de deserializarlo. El AUC se usó como medida principal porque evalúa la capacidad de ordenar compradores y no compradores sin depender de un umbral. La precisión, sensibilidad y F1 sí se calcularon con el umbral elegido previamente en validación para cada modelo.

| Modelo | AUC | Precisión | Sensibilidad | F1 | Entrenamiento (s) | Predicción (s) |
|---|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | **0.6133** | 0.4046 | **0.6969** | **0.5120** | **0.73** | **0.10** |
| Perceptrón multicapa | 0.6122 | 0.4393 | 0.5078 | 0.4711 | 51.46 | 0.17 |
| Regresión logística | 0.6026 | **0.4461** | 0.3886 | 0.4154 | 8.89 | 0.13 |
| Random Forest | 0.6019 | 0.4230 | 0.5238 | 0.4680 | 55.25 | 0.51 |

Gradient Boosting obtuvo el mayor AUC, 0.6133, aunque la diferencia frente al perceptrón multicapa fue de apenas 0.0011. También logró la mayor sensibilidad, 0.6969, y el mayor F1, 0.5120. Esto significa que recuperó cerca de siete de cada diez recompradores con su umbral de validación, a costa de una precisión de 0.4046. La regresión logística alcanzó la precisión más alta, 0.4461, pero su sensibilidad de 0.3886 dejó sin detectar a la mayoría de los recompradores.

Ningún modelo final superó el AUC 0.645 del modelo logístico previo: las diferencias fueron −0.0317 para Gradient Boosting, −0.0328 para el perceptrón, −0.0424 para la regresión logística ajustada y −0.0431 para Random Forest. Esta caída no invalida el ajuste, porque el 0.645 provenía de una partición anterior; muestra que el desempeño cambia al mover la evaluación a una semana futura que permaneció intacta durante el desarrollo. Las curvas ROC de la Figura `08_curvas_roc.png` permiten ver que los cuatro modelos discriminan mejor que el azar, pero con una separación moderada entre clases.

## Costo de cómputo y elección

La comparación de tiempos favorece con claridad a Gradient Boosting. Su entrenamiento registrado tomó 0.73 segundos y la predicción de los 35,499 clientes tomó 0.10 segundos. Random Forest requirió 55.25 segundos para entrenarse y 0.51 segundos para predecir sin aportar una mejora de AUC. El perceptrón necesitó 51.46 segundos de entrenamiento para un AUC casi idéntico al de Gradient Boosting. Por tanto, Gradient Boosting ofrece el mejor equilibrio entre discriminación, sensibilidad y costo, y se selecciona como modelo final.

Los tiempos dependen del equipo y no deben interpretarse como una propiedad absoluta del algoritmo. Su utilidad aquí es comparar las cuatro implementaciones ejecutadas con el mismo conjunto de datos y el mismo entorno. La Figura `08_tiempo_vs_auc.png` muestra ambas magnitudes sin combinarlas en un indicador artificial.

## Errores de clasificación

La tabla principal usa el umbral de validación propio de cada modelo. Para comparar la forma de los errores bajo una misma regla, la Figura `08_matrices_confusion.png` fija un umbral común de 0.50. Con ese corte, Random Forest casi nunca predice recompra: produce solamente 32 predicciones positivas entre 35,499 clientes. Gradient Boosting ofrece el reparto más útil, con 7,212 verdaderos positivos y 5,536 falsos negativos, aunque también genera 9,595 falsos positivos. Esta comparación confirma que el umbral debe elegirse de acuerdo con el costo comercial de enviar un cupón innecesario frente al costo de omitir a un cliente que sí volvería.

## Ofertas no vistas

En la misma semana temporal se separaron las ofertas según su presencia en el entrenamiento original. Hubo 1,334 clientes con siete ofertas vistas y 34,165 clientes con cinco ofertas no vistas, es decir, 96.3% de los clientes de prueba recibieron una oferta nueva. En ofertas vistas, los AUC de regresión logística, Random Forest, Gradient Boosting y perceptrón fueron 0.6578, 0.6900, 0.6810 y 0.6833; en ofertas no vistas fueron 0.6066, 0.6010, 0.6101 y 0.6095. La degradación respectiva fue 0.0512, 0.0890, 0.0709 y 0.0738 puntos de AUC.

La robustez a ofertas nuevas también se midió mediante cinco divisiones `GroupShuffleSplit`. En cada división se reservaron cinco ofertas completas; la intersección de ofertas entre entrenamiento y prueba fue vacía. Los modelos se reentrenaron temporalmente en cada división y se descartaron después de evaluarlos.

| Modelo | AUC medio | Desviación estándar | Mínimo | Máximo |
|---|---:|---:|---:|---:|
| Gradient Boosting | **0.6197** | 0.0659 | 0.5460 | 0.6934 |
| Random Forest | 0.6107 | **0.0349** | 0.5803 | 0.6654 |
| Regresión logística | 0.6090 | 0.0518 | 0.5454 | 0.6820 |
| Perceptrón multicapa | 0.5824 | 0.0590 | 0.5321 | 0.6800 |

Gradient Boosting conservó el mejor AUC promedio, 0.6197, pero su desviación estándar de 0.0659 fue la mayor. Random Forest fue el más estable, con desviación de 0.0349, aunque su AUC medio quedó 0.0090 por debajo. Los rangos amplios de la Figura `08_auc_ofertas_no_vistas.png` muestran que la identidad de la oferta influye de forma sustancial en el desempeño. Por ello, el AUC 0.6197 se interpreta como evidencia de generalización moderada, no como garantía de igual desempeño para cualquier campaña futura.

## Reproducibilidad

Las cifras provienen de `results/tables/08_metricas_prueba.csv`, `08_predicciones_prueba.parquet` y `08_validacion_por_ofertas.csv`, generadas por `notebooks/08_evaluacion_y_comparacion.ipynb`. Las cinco figuras `08_*` incluyen fuente, período, unidades y fecha de ejecución. Las predicciones individuales se guardaron para que la aplicación pueda reconstruir curvas ROC y matrices de confusión sin volver a entrenar ni modificar los modelos.
