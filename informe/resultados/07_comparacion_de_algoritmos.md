# Comparación de los algoritmos

## Protocolo de evaluación

Los cuatro modelos versión 2 se reentrenaron con los datos del 1 de marzo al 23 de abril de 2013, conservando los hiperparámetros y umbrales elegidos previamente en validación. Después se evaluaron una sola vez en la semana de prueba, del 24 al 30 de abril, formada por 35,499 clientes. La prueba no intervino en el ajuste. El AUC fue la medida principal; precisión, sensibilidad y F1 se calcularon con el umbral de validación propio de cada modelo.

| Modelo | AUC | Precisión | Sensibilidad | F1 | Entrenamiento (s) | Predicción (s) |
|---|---:|---:|---:|---:|---:|---:|
| Regresión logística | **0.6470** | 0.3936 | **0.8848** | **0.5448** | 10.02 | 0.10 |
| Perceptrón multicapa | 0.6461 | **0.4154** | 0.7476 | 0.5341 | 10.69 | 0.12 |
| Random Forest | 0.6151 | 0.3764 | 0.8169 | 0.5153 | 53.74 | 0.34 |
| Gradient Boosting | 0.6121 | 0.3846 | 0.7622 | 0.5112 | **0.70** | **0.07** |

La regresión logística obtuvo el mayor AUC y F1, aunque su ventaja de AUC frente al perceptrón fue de solo 0.0009. El perceptrón logró la mayor precisión y produjo menos clasificaciones positivas. Por ello, ambos modelos muestran una capacidad de discriminación prácticamente equivalente; la elección entre ellos depende del costo de contactar clientes adicionales.

![Curvas ROC de los modelos](../../results/figures/08_curvas_roc.png)

![AUC de los modelos](../../results/figures/08_auc_modelos.png)

El modelo logístico de la fase exploratoria obtuvo un AUC de 0.6446 sobre la misma semana de prueba, también con entrenamiento hasta el 23 de abril. Frente a esa referencia comparable, la regresión logística versión 2 mejoró 0.0024 y el perceptrón 0.0015; Random Forest y Gradient Boosting quedaron 0.0295 y 0.0325 por debajo, respectivamente.

## Versión 1 frente a versión 2

La versión 1 de cada modelo se entrenó solo hasta el 16 de abril; la versión 2 agregó la semana de validación y conservó los mismos hiperparámetros. La tabla compara el AUC de ambas en la semana de prueba. Las cifras de la versión 1 corresponden a la primera ejecución de este cuaderno.

| Modelo | AUC versión 1 | AUC versión 2 | Cambio |
|---|---:|---:|---:|
| Regresión logística | 0.6026 | **0.6470** | +0.0444 |
| Perceptrón multicapa | 0.6122 | 0.6461 | +0.0339 |
| Random Forest | 0.6019 | 0.6151 | +0.0132 |
| Gradient Boosting | **0.6133** | 0.6121 | −0.0012 |

Agregar la semana más reciente mejoró a tres de los cuatro modelos y cambió el orden: con la versión 1, Gradient Boosting tenía el mayor AUC. La semana de validación se parece más a la de prueba, con 33.60% de recompra frente a 35.91%. Además, con ella los modelos pasaron de conocer 7 a conocer 11 de las 12 ofertas que aparecen en la prueba, así que parte de la mejora se explica porque ya habían visto esas ofertas. La sección de ofertas no vistas analiza ese efecto.

## Costo de cómputo y elección

Gradient Boosting fue el más rápido, con 0.70 segundos de entrenamiento y 0.07 segundos de predicción, pero obtuvo el menor AUC. La regresión logística y el perceptrón necesitaron cerca de 10 segundos para entrenarse y superaron 0.646 de AUC. Se toma la regresión logística como modelo de referencia por alcanzar el mayor AUC, sensibilidad y F1, sin una carga de cómputo relevante. El perceptrón es una alternativa razonable si se prefiere reducir la cantidad de clientes clasificados como recompra.

![Tiempo de entrenamiento frente a AUC](../../results/figures/08_tiempo_vs_auc.png)

## Errores de clasificación

Las matrices usan el umbral seleccionado en validación para cada modelo, igual que la tabla principal. La regresión logística clasificó como recompra al 80.7% de los clientes; el perceptrón, al 64.6%; Random Forest, al 77.9%; y Gradient Boosting, al 71.2%. Estas proporciones explican la sensibilidad alta y la precisión moderada: los umbrales favorecen detectar recompradores, pero también generan falsos positivos. Antes del despliegue deben recalibrarse según el costo del descuento y el valor esperado de una recompra, sin usar la semana de prueba para optimizarlos.

![Matrices de confusión con umbrales de validación](../../results/figures/08_matrices_confusion.png)

## Ofertas no vistas

Una oferta cuenta como vista si aparece en los datos con que se entrenaron los modelos versión 2, del 1 de marzo al 23 de abril. Con ese criterio, 33,086 clientes de la semana de prueba, el 93.2%, recibieron una de las 11 ofertas ya vistas, y solo 2,413, el 6.8%, recibieron la única oferta nueva. Por lo tanto, la semana de prueba mide sobre todo el desempeño con ofertas conocidas.

| Modelo | AUC con ofertas vistas | AUC con la oferta no vista | Diferencia |
|---|---:|---:|---:|
| Regresión logística | 0.6552 | 0.5710 | −0.0841 |
| Perceptrón multicapa | 0.6507 | 0.5871 | −0.0636 |
| Random Forest | 0.6206 | 0.5664 | −0.0542 |
| Gradient Boosting | 0.6185 | 0.5757 | −0.0428 |

Los cuatro modelos pierden AUC con la oferta nueva. Los que más ganaron al incorporar la semana de validación, la regresión logística y el perceptrón, son también los que más pierden. Como el grupo no visto es una sola oferta, esta comparación es solo indicativa.

La situación del conjunto de prueba de Kaggle es distinta: allí el 96.3% de los clientes recibió ofertas ausentes del entrenamiento. Para medir ese caso se usaron cinco divisiones `GroupShuffleSplit` que reservaron ofertas completas. En cada división los modelos se reentrenan con los hiperparámetros de su ficha, por lo que este resultado no depende de la versión 1 o 2.

| Modelo | AUC medio | Desviación estándar | Mínimo | Máximo |
|---|---:|---:|---:|---:|
| Gradient Boosting | **0.6197** | 0.0659 | 0.5460 | 0.6934 |
| Random Forest | 0.6107 | **0.0349** | 0.5803 | 0.6654 |
| Regresión logística | 0.6090 | 0.0518 | 0.5454 | 0.6820 |
| Perceptrón multicapa | 0.5824 | 0.0590 | 0.5321 | 0.6800 |

![AUC al reservar ofertas no vistas](../../results/figures/08_auc_ofertas_no_vistas.png)

Con ofertas nuevas, Gradient Boosting obtuvo el mayor AUC medio, 0.6197, y la regresión logística el tercero, 0.6090. La diferencia, 0.0107, es menor que la desviación estándar de ambos modelos, así que con ofertas nuevas ningún algoritmo se distingue con claridad. La regresión logística se mantiene como modelo de referencia porque obtuvo el mayor AUC en la semana de prueba, su desempeño con ofertas nuevas queda dentro del rango de los demás y es el más sencillo de interpretar. Aun así, su rendimiento debe vigilarse en cada campaña que traiga ofertas nuevas.

## Reproducibilidad

Las cifras provienen de `results/tables/08_metricas_prueba.csv`, `08_predicciones_prueba.parquet` y `08_validacion_por_ofertas.csv`, generadas por el notebook 08 ejecutado con `cargar_modelos(version=2)`. Las ofertas vistas se definen respecto del entrenamiento y la validación, que son los datos de los modelos versión 2. Las predicciones individuales permiten reconstruir las curvas ROC y matrices sin reentrenar los modelos.
