# Metodología

## Pasos que siguió el grupo

El trabajo se organizó en etapas encadenadas. Cada etapa tiene un cuaderno de Jupyter o un módulo de código en el repositorio, y cada una produce los archivos que usa la siguiente.

1. **Carga y exploración inicial.** Se cargaron los archivos de la competencia y se describió la variable respuesta. Hay 160,057 clientes con respuesta conocida, de los cuales 27.14% volvió a comprar.
2. **Preprocesamiento de las transacciones.** Las 349,655,789 líneas de compra se leyeron por bloques desde el archivo comprimido y se resumieron en una fila por cliente, con 37 variables de comportamiento previas a la oferta.
3. **Análisis exploratorio.** Se estudiaron las variables numéricas y categóricas, sus relaciones con la recompra, los valores faltantes y los atípicos.
4. **Revisión bibliográfica y selección de algoritmos.** Se revisaron estudios de recompra, abandono de clientes y respuesta a cupones para elegir los algoritmos.
5. **Partición de los datos.** Los clientes se separaron por fecha de oferta en entrenamiento, validación y prueba.
6. **Ajuste de hiperparámetros.** Cada algoritmo se entrenó con el bloque de entrenamiento y sus hiperparámetros se eligieron con el AUC de la semana de validación.
7. **Modelos finales.** Con la configuración elegida, cada algoritmo se reentrenó con entrenamiento y validación juntos y se guardó como un modelo congelado, que ya no se vuelve a entrenar.
8. **Evaluación.** Los modelos congelados se evaluaron una sola vez en la semana de prueba. Su desempeño con ofertas nuevas se midió con divisiones agrupadas por oferta.
9. **Aplicación.** Se construyó una aplicación que preprocesa los datos de un cliente nuevo y lo clasifica con los modelos congelados.

## Selección de los conjuntos de entrenamiento y prueba

El conjunto de prueba de la competencia, con 151,484 clientes, no incluye la variable respuesta, así que no sirve para medir los modelos. Por eso la evaluación se hizo dentro de los 160,057 clientes con respuesta conocida, separándolos por la fecha de su oferta:

| Conjunto | Fechas de la oferta | Clientes | Tasa de recompra | Uso |
|---|---|---:|---:|---|
| Entrenamiento | 1 de marzo al 16 de abril de 2013 | 106,062 | 23.08% | Entrenar los modelos durante el ajuste |
| Validación | 17 al 23 de abril | 18,496 | 33.60% | Elegir los hiperparámetros y el umbral |
| Prueba | 24 al 30 de abril | 35,499 | 35.91% | Medir los modelos finales una sola vez |

La partición respeta el orden del tiempo por tres razones:

- **Así se usa un modelo en la práctica.** Se entrena con el pasado y se aplica a clientes futuros.
- **Así separó los datos la competencia.** Su conjunto de prueba contiene las ofertas emitidas a partir del 1 de mayo de 2013.
- **El comportamiento cambia entre semanas.** La tasa de recompra sube de 23.08% en el entrenamiento a 35.91% en la prueba. Una partición al azar mezclaría esas semanas y daría una medida demasiado optimista.

Después de elegir los hiperparámetros, los modelos finales se reentrenaron con entrenamiento y validación juntos, 124,558 clientes del 1 de marzo al 23 de abril, conservando los hiperparámetros y los umbrales elegidos. La semana de validación es la más parecida a la de prueba, y un modelo de producción debe aprovechar los datos más recientes disponibles. La semana de prueba nunca se usó para ajustar ni para elegir umbrales.

Esta decisión se tomó después de conocer el resultado de la primera versión de los modelos, entrenada solo hasta el 16 de abril. Por transparencia se reportan ambas versiones.

### Validación con ofertas nuevas

La semana de prueba contiene sobre todo ofertas que los modelos finales ya conocían. En cambio, la mayoría de los clientes del conjunto de prueba de la competencia recibió ofertas que no aparecen en el entrenamiento. Para medir ese caso, se hicieron cinco divisiones de los 124,558 clientes de entrenamiento y validación que reservan ofertas completas. En cada división se apartaron cinco ofertas para probar, se entrenó un modelo temporal con las demás y se verificó que ninguna oferta estuviera en ambos lados. Esos modelos temporales se descartaron después de evaluarlos.

## Selección de los algoritmos

Los algoritmos se eligieron a partir de la revisión bibliográfica y de las características de los datos:

| Característica de los datos | Consecuencia para la elección |
|---|---|
| Datos tabulares, una fila por cliente | Favorece a los ensambles de árboles |
| 160,057 clientes | Descarta métodos cuyo costo crece mucho con el número de filas |
| 37 variables de comportamiento y 5 identificadores de grupo con pocas categorías | Basta con codificar los grupos; no se necesitan modelos para categorías con miles de niveles |
| Faltantes con significado en la recencia de la categoría | Conviene un algoritmo que use el faltante como información |
| Colas largas y valores extremos | Los árboles los toleran; los modelos lineales y las redes necesitan transformarlos |
| 27.14% de clientes recurrentes | Ponderar las clases y medir con el AUC |
| Uso en una aplicación | El modelo debe dar probabilidades y predecir rápido |

Se seleccionaron cuatro algoritmos de aprendizaje automático, uno por cada familia que la literatura identifica como relevante:

| Algoritmo | Papel en el proyecto | Razón |
|---|---|---|
| Regresión logística | Referencia | Es el estándar de la industria, se interpreta con facilidad y ya se había usado en la fase exploratoria |
| Random Forest | Ensamble por bagging | Lessmann et al. (2015) lo recomiendan como punto de comparación obligatorio |
| Gradient Boosting, con `HistGradientBoostingClassifier` | Ensamble por boosting | Fue el mejor en los estudios más parecidos (Martínez et al., 2020; Ahmad et al., 2019) y maneja los faltantes de forma nativa |
| Perceptrón multicapa | Red neuronal | Permite comprobar si una red neuronal supera a los árboles en estos datos |

Para gradient boosting se eligió la implementación de scikit-learn y no XGBoost ni LightGBM por tres razones:

- Está inspirada en LightGBM, así que conserva su forma de trabajar con histogramas.
- Ya estaba disponible en el entorno del proyecto.
- Comparte la interfaz de los demás modelos, lo que simplifica guardarlos y cargarlos en la aplicación.

Se descartaron otros algoritmos por las siguientes razones:

- **Máquinas de soporte vectorial con kernel:** su costo de entrenamiento crece rápidamente con el número de clientes y no entregan probabilidades de forma directa.
- **Naive Bayes:** supone variables independientes, y las ventanas de compra de 30, 60, 90 y 180 días están anidadas.
- **Redes profundas para datos tabulares:** no superan a los ensambles de árboles y requieren mucho más ajuste (Shwartz-Ziv y Armon, 2022; Grinsztajn et al., 2022).
- **Máquinas de factorización:** están pensadas para categorías con muchísimos niveles, y aquí hay pocas.
- **Sobremuestreo con SMOTE:** Verbeke et al. (2012) no encontraron mejoras significativas con sobremuestreo, así que se prefirió ponderar las clases.

## Selección de las herramientas

El proyecto se desarrolló en Python con scikit-learn, pandas y numpy, y el código se versionó en GitHub. Los recursos de cómputo, los lenguajes, las bibliotecas y sus versiones se detallan en la sección de herramientas utilizadas. Las tecnologías de la aplicación se justifican en la sección que la describe.
