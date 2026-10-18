# Metodología

## Pasos que siguió el grupo

El trabajo se organizó en etapas, una después de otra. Cada etapa tiene un cuaderno de Jupyter o un módulo de código en el repositorio, y cada una genera los archivos que usa la siguiente.

1. Carga y exploración inicial. Se cargaron los archivos de la competencia y se describió la variable respuesta. Hay 160,057 clientes con respuesta conocida, de los cuales el 27.14% volvió a comprar.
2. Preprocesamiento de las compras. Las 349,655,789 líneas de compra se leyeron por partes desde el archivo comprimido y se resumieron en una fila por cliente, con 37 variables de comportamiento anteriores a la oferta.
3. Análisis exploratorio. Se estudiaron las variables numéricas y categóricas, su relación con la recompra, los valores faltantes y los valores extremos.
4. Revisión bibliográfica y selección de algoritmos. Se revisaron estudios de recompra, abandono de clientes y respuesta a cupones para elegir los algoritmos.
5. Separación de los datos. Los clientes se separaron según la fecha de su oferta en entrenamiento, validación y prueba.
6. Ajuste de hiperparámetros. Cada algoritmo se entrenó con el bloque de entrenamiento y sus hiperparámetros se eligieron con el AUC de la semana de validación.
7. Modelos finales. Con la configuración elegida, cada algoritmo se volvió a entrenar con entrenamiento y validación juntos, y se guardó como un modelo congelado que ya no se vuelve a entrenar.
8. Evaluación. Los modelos congelados se evaluaron una sola vez en la semana de prueba. Su desempeño con ofertas nuevas se midió con divisiones agrupadas por oferta.
9. Aplicación. Se construyó una aplicación que prepara los datos de un cliente nuevo y lo clasifica con los modelos congelados.

## Selección de los conjuntos de entrenamiento y prueba

El conjunto de prueba de la competencia, con 151,484 clientes, no trae la variable respuesta, así que no sirve para medir los modelos. Por eso la evaluación se hizo con los 160,057 clientes con respuesta conocida, separándolos según la fecha de su oferta como muestra la siguiente tabla.

| Conjunto | Fechas de la oferta | Clientes | Porcentaje | Tasa de recompra | Uso |
|---|---|---:|---:|---:|---|
| Entrenamiento | 1 de marzo al 16 de abril de 2013 | 106,062 | 66% | 23.08% | Entrenar los modelos durante el ajuste |
| Validación | 17 al 23 de abril | 18,496 | 12% | 33.60% | Elegir los hiperparámetros y el umbral |
| Prueba | 24 al 30 de abril | 35,499 | 22% | 35.91% | Medir los modelos finales una sola vez |

La separación queda en 66% para entrenamiento, 12% para validación y 22% para prueba. Si se juntan entrenamiento y validación, la proporción es de 78% contra 22%, cerca de la separación 80/20 que se usa con frecuencia. El corte se planteó como un 80/20 por fecha, con el último 20% de los clientes para la prueba, pero se ajustó a días completos para que ningún día quedara repartido entre dos conjuntos. Por eso la prueba quedó en la última semana de abril, del 24 al 30, con el 22% de los clientes. Después, la semana anterior a la prueba, del 17 al 23 de abril, se separó del entrenamiento para usarla como validación.

Esta proporción deja suficientes datos en cada parte. El entrenamiento tiene más de 100,000 clientes para que los modelos aprendan, y la prueba tiene 35,499 clientes, una cantidad grande para que el AUC no cambie mucho por casualidad. La validación, con 18,496 clientes, alcanza para comparar las configuraciones de hiperparámetros sin tocar la semana de prueba.

La separación respeta el orden del tiempo por tres razones. La primera es que así se usa un modelo en la práctica, porque se entrena con el pasado y se aplica a clientes futuros. La segunda es que así separó los datos la competencia, cuyo conjunto de prueba tiene las ofertas emitidas a partir del 1 de mayo de 2013. La tercera es que el comportamiento cambia entre semanas. La tasa de recompra sube de 23.08% en el entrenamiento a 35.91% en la prueba, y una separación al azar mezclaría esas semanas y daría una medida demasiado optimista.

Después de elegir los hiperparámetros, los modelos finales se volvieron a entrenar con entrenamiento y validación juntos, que suman 124,558 clientes del 1 de marzo al 23 de abril, con los mismos hiperparámetros y umbrales. La semana de validación es la más parecida a la de prueba, y un modelo que se usa en la práctica debe aprovechar los datos más recientes que tenga. La semana de prueba nunca se usó para ajustar ni para elegir umbrales.

Esta decisión se tomó después de conocer el resultado de la primera versión de los modelos, entrenada solo hasta el 16 de abril. Por transparencia se reportan las dos versiones.

### Validación con ofertas nuevas

La semana de prueba tiene sobre todo ofertas que los modelos finales ya conocían. En cambio, la mayoría de los clientes del conjunto de prueba de la competencia recibió ofertas que no aparecen en el entrenamiento. Para medir ese caso se hicieron cinco divisiones de los 124,558 clientes de entrenamiento y validación, separando ofertas completas. En cada división se apartaron cinco ofertas para probar, se entrenó un modelo temporal con las demás y se revisó que ninguna oferta quedara en los dos lados. Esos modelos temporales se descartaron después de evaluarlos.

## Selección de los algoritmos

Los algoritmos se eligieron a partir de la revisión bibliográfica y de las características de los datos, que se resumen en la siguiente tabla.

| Característica de los datos | Qué significa para la elección |
|---|---|
| Datos en tablas, con una fila por cliente | Favorece a los ensambles de árboles |
| 160,057 clientes | Descarta métodos cuyo costo crece mucho con el número de filas |
| 37 variables de comportamiento y 5 identificadores de grupo con pocas categorías | Basta con codificar los grupos, sin modelos pensados para categorías con miles de valores |
| Faltantes con significado en los días desde la última compra de la categoría | Conviene un algoritmo que use el faltante como información |
| Valores extremos en el gasto y la frecuencia | Los árboles los soportan bien, mientras que los modelos lineales y las redes necesitan transformarlos |
| 27.14% de clientes recurrentes | Conviene ponderar las clases y medir con el AUC |
| Uso en una aplicación | El modelo debe dar probabilidades y predecir rápido |

Se eligieron cuatro algoritmos de aprendizaje automático, uno por cada familia que los estudios revisados señalan como importante.

| Algoritmo | Papel en el proyecto | Razón |
|---|---|---|
| Regresión logística | Referencia | Es el modelo más usado en la industria, es fácil de interpretar y ya se había usado en la fase exploratoria |
| Random Forest | Ensamble por bagging | Lessmann et al. (2015) lo recomiendan como punto de comparación obligatorio |
| Gradient Boosting, con HistGradientBoostingClassifier | Ensamble por boosting | Fue el mejor en los estudios más parecidos (Martínez et al., 2020; Ahmad et al., 2019) y maneja por sí solo los valores faltantes |
| Perceptrón multicapa | Red neuronal | Permite comprobar si una red neuronal supera a los árboles en estos datos |

Para Gradient Boosting se eligió la versión de scikit-learn y no XGBoost ni LightGBM por tres razones. Está inspirada en LightGBM, así que trabaja con histogramas de la misma forma. Ya estaba instalada en el entorno del proyecto. Y se usa igual que los demás modelos, lo que facilita guardarlos y cargarlos en la aplicación.

También se descartaron otros algoritmos. Las máquinas de soporte vectorial con kernel tardan mucho más en entrenar a medida que crece el número de clientes y no dan probabilidades de forma directa. Naive Bayes supone que las variables son independientes entre sí, y aquí no lo son, porque las compras de los últimos 30 días también cuentan dentro de las de 60, 90 y 180 días. Las redes profundas para datos en tablas no superan a los ensambles de árboles y necesitan mucho más ajuste (Shwartz-Ziv y Armon, 2022; Grinsztajn et al., 2022). Las máquinas de factorización están pensadas para categorías con muchísimos valores distintos, y aquí hay pocas. Por último, no se usó sobremuestreo con SMOTE, porque Verbeke et al. (2012) no encontraron mejoras significativas con esa técnica, y se prefirió ponderar las clases.
