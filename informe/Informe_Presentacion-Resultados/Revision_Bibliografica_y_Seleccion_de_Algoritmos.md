<div align="center">

UNIVERSIDAD DEL VALLE DE GUATEMALA<br>
Data Science<br>
Facultad de Ingeniería<br>
Sección 40<br>
Erick Marroquín

<br>

**Revisión bibliográfica y selección de algoritmos**

<br>

Joel Jaquez - 23369<br>
Fernando Ruiz - 23065<br>
Kevin Villagran - 23584<br>
Nery Molina - 23218

<br>

GUATEMALA, 16 de octubre 2026

</div>

---

# Revisión bibliográfica y selección de algoritmos

## Predicción de compradores recurrentes

Departamento de Ciencias de la Computación. CC3084 Data Science. Proyecto 2, Resultados.

---

## 1. Introducción

Este documento cubre las dos primeras actividades de la guía de resultados. La primera es revisar qué algoritmos de aprendizaje automático se han usado para resolver problemas similares al nuestro. La segunda es seleccionar, con base en esa revisión, los algoritmos que el grupo usará para construir y comparar sus modelos.

El problema ya quedó definido en la fase exploratoria. A cada cliente se le ofreció un cupón sobre un producto y se busca estimar la probabilidad de que vuelva a comprar ese producto después de redimirlo. Es una clasificación binaria sobre datos tabulares: 160,057 clientes de entrenamiento, 37 variables de comportamiento construidas a partir de 349,655,789 transacciones y cinco identificadores anónimos de la oferta. El 27.14% de los clientes recompró. La métrica oficial del reto es el área bajo la curva ROC, AUC.

Hasta ahora el proyecto tiene un solo algoritmo, la regresión logística, en dos versiones. La línea base obtuvo un AUC de 0.561 en la semana de validación y la versión con variables avanzadas obtuvo 0.645. La revisión busca responder qué otros algoritmos tiene sentido probar y por qué.

## 2. Criterios de búsqueda

Se buscaron artículos en revistas indexadas y en actas de conferencias arbitradas de primer nivel. Las fuentes consultadas fueron ScienceDirect de Elsevier, SAGE Journals, Springer, las actas de ACM SIGKDD y NeurIPS, el Journal of Machine Learning Research y Nature. Cada referencia se verificó contra el registro de Crossref para confirmar autores, año, volumen y páginas.

Se usaron tres familias de problemas parecidos al nuestro:

- **Predicción de recompra y de compra futura.** Es el problema más cercano: estimar si un cliente volverá a comprar.
- **Predicción de abandono de clientes.** Es el problema inverso, estimar si un cliente dejará de comprar, y es el que tiene más literatura comparativa de algoritmos.
- **Respuesta a cupones y promociones.** Comparte el detonante del problema, que es una oferta de descuento.

A esto se sumaron estudios que comparan algoritmos sobre datos tabulares en general y las referencias originales de cada algoritmo.

## 3. Algoritmos usados en problemas similares

### 3.1 Predicción de recompra y de compra futura

Liu et al. (2016) ganaron la primera etapa de la competencia de compradores recurrentes que Alibaba organizó en IJCAI 2015 con datos del evento "Double 11" de Tmall, entre 753 equipos. El problema es casi idéntico al nuestro: muchos compradores llegan atraídos por una promoción y solo algunos se convierten en clientes leales. Su conclusión principal es que la construcción de variables fue el factor más importante del resultado. Crearon perfiles de usuarios, comercios, marcas, categorías, productos y de sus interacciones.

Zhang et al. (2022) trabajaron con el mismo conjunto de Tmall. Entrenaron máquinas de factorización, regresión logística, XGBoost y LightGBM, corrigieron el desbalance de clases con SMOTE y combinaron los modelos con un ensamble de dos capas por apilamiento. La mejora total que reportan en AUC fue de 0.01161.

Martínez et al. (2020) construyeron un marco para predecir si un cliente comprará en el mes siguiente en un entorno no contractual, que es el mismo tipo de relación que tiene un supermercado con sus clientes. Las variables se basaron en el momento y el valor de las compras anteriores. Entre los métodos que compararon, el gradient boosting de árboles fue el mejor, con un AUC de 0.95 sobre más de 10,000 clientes y 200,000 compras.

Larivière y Van den Poel (2005) predijeron la siguiente compra, el abandono parcial y la evolución de la rentabilidad de 100,000 clientes de una empresa europea de servicios financieros. Los bosques aleatorios superaron a la regresión logística y a la regresión lineal en las muestras de estimación y de validación. El comportamiento pasado del cliente fue lo que más pesó para predecir la recompra.

### 3.2 Predicción de abandono de clientes

Buckinx y Van den Poel (2005) es el estudio más parecido a nuestro contexto. Predijeron el abandono parcial de los clientes más leales de un minorista de productos de consumo masivo en un entorno no contractual. Compararon regresión logística, redes neuronales con determinación automática de relevancia y bosques aleatorios, medidos con exactitud y AUC. Las mejores variables predictoras fueron las de comportamiento pasado: recencia, frecuencia y valor monetario.

Lemmens y Croux (2006) aplicaron bagging y boosting de árboles de clasificación a la base de clientes de una empresa estadounidense de telefonía móvil. Ambas técnicas mejoraron de forma significativa la predicción del abandono. También recomiendan un muestreo balanceado cuando el evento es poco frecuente, acompañado de una corrección del sesgo que introduce.

Neslin et al. (2006) analizaron un torneo abierto en el que académicos y profesionales estimaron modelos de abandono sobre los mismos datos. Encontraron que el método sí importa, porque las diferencias de exactitud entre propuestas cambiaban la rentabilidad de una campaña en cientos de miles de dólares. También encontraron que los modelos conservan su desempeño cuando se aplican a datos recolectados tres meses después de los de entrenamiento.

Verbeke et al. (2012) compararon una amplia gama de clasificadores sobre once conjuntos de datos de operadores de telecomunicaciones. Un grupo grande de clasificadores obtuvo desempeños comparables, pocas variables bastaron para predecir con exactitud y el sobremuestreo en general no mejoró los resultados de forma significativa. Proponen además una medida de evaluación basada en la ganancia de la campaña de retención.

Vafeiadis et al. (2015) compararon redes neuronales, árboles de decisión, máquinas de soporte vectorial, Naive Bayes y regresión logística en un conjunto público de abandono de telefonía, con y sin boosting. Aplicar boosting mejoró el desempeño de los clasificadores.

De Caigny et al. (2018) propusieron el modelo logit leaf, que segmenta a los clientes con un árbol de decisión y ajusta una regresión logística en cada segmento. Lo compararon con árboles de decisión, regresión logística, bosques aleatorios y árboles con modelos logísticos. Superó a sus dos componentes por separado y fue al menos tan bueno como los bosques aleatorios, conservando la facilidad de interpretación.

Ahmad et al. (2019) predijeron el abandono en un operador de telefonía con nueve meses de datos de todos sus clientes. Compararon árbol de decisión, bosque aleatorio, gradient boosting y XGBoost, y XGBoost dio los mejores resultados con un AUC de 93.3%. Las nuevas variables de redes sociales subieron el AUC de 84% a 93.3%, otro caso en que las variables pesaron más que el algoritmo.

Burez y Van den Poel (2009) estudiaron el desbalance de clases en la predicción de abandono. Compararon métodos de submuestreo y dos técnicas de modelado, gradient boosting y bosques aleatorios ponderados, contra modelos estándar. Recomiendan medir con AUC en lugar de exactitud, porque el AUC no depende de un umbral de clasificación.

### 3.3 Respuesta a cupones

Ren et al. (2021) propusieron un modelo de dos etapas para decidir a qué consumidores enviar cupones electrónicos. Primero segmentaron a los consumidores por su comportamiento en línea y separaron a los poco activos. Después ajustaron un bosque de aislamiento para el segmento poco activo y una regresión logística para los segmentos activos. La segmentación redujo el desbalance de los datos y mejoró la predicción frente a los métodos tradicionales.

### 3.4 Comparaciones generales sobre datos tabulares

Lessmann et al. (2015) compararon 41 clasificadores sobre varios conjuntos de datos de riesgo crediticio, otro problema de clasificación binaria sobre datos tabulares de clientes. Los ensambles homogéneos tendieron a superar a los clasificadores individuales, y entre ellos el bosque aleatorio dio las estimaciones más exactas. El mejor resultado global fue un ensamble heterogéneo que selecciona y combina modelos de distintos algoritmos. Las redes neuronales fueron significativamente mejores que la regresión logística. Recomiendan usar el bosque aleatorio como punto de comparación obligatorio para cualquier algoritmo nuevo y evitar comparar únicamente contra la regresión logística.

Shwartz-Ziv y Armon (2022) compararon modelos recientes de aprendizaje profundo diseñados para datos tabulares contra XGBoost. XGBoost los superó en todos los conjuntos evaluados, incluidos los que usaron los autores de esos modelos, y requirió mucho menos ajuste de hiperparámetros.

Grinsztajn et al. (2022) llegaron a la misma conclusión con un banco de pruebas de 45 conjuntos de datos. Los modelos basados en árboles, como XGBoost y los bosques aleatorios, siguen siendo superiores en datos tabulares de tamaño mediano, incluso sin contar su ventaja en velocidad.

### 3.5 Síntesis

| Estudio | Problema | Algoritmos comparados | Resultado relevante |
|---|---|---|---|
| Liu et al., 2016 | Recompra tras promoción, Tmall | Construcción de variables y entrenamiento de modelos | Las variables fueron el factor decisivo |
| Zhang et al., 2022 | Recompra tras promoción, Tmall | Factorización, logística, XGBoost, LightGBM, apilamiento | El apilamiento subió el AUC en 0.01161 |
| Martínez et al., 2020 | Compra del mes siguiente | Varios métodos de aprendizaje automático | Gradient boosting fue el mejor |
| Larivière y Van den Poel, 2005 | Siguiente compra y abandono | Bosque aleatorio contra regresiones | El bosque aleatorio ajustó mejor |
| Buckinx y Van den Poel, 2005 | Abandono parcial en minorista | Logística, red neuronal, bosque aleatorio | Recencia, frecuencia y valor fueron las mejores variables |
| Lemmens y Croux, 2006 | Abandono en telefonía | Bagging y boosting de árboles | Ambos mejoraron la predicción |
| Verbeke et al., 2012 | Abandono en telefonía | Amplia gama de clasificadores | Muchos clasificadores rinden parecido |
| Ahmad et al., 2019 | Abandono en telefonía | Árbol, bosque, gradient boosting, XGBoost | XGBoost fue el mejor |
| De Caigny et al., 2018 | Abandono | Árbol, logística, bosque, modelo híbrido | El híbrido igualó al bosque aleatorio |
| Ren et al., 2021 | Uso de cupones electrónicos | Segmentación, bosque de aislamiento, logística | Segmentar redujo el desbalance |
| Lessmann et al., 2015 | Riesgo crediticio | 41 clasificadores | El bosque aleatorio es el punto de comparación recomendado |
| Shwartz-Ziv y Armon, 2022 | Datos tabulares en general | XGBoost contra redes profundas | XGBoost fue superior y más fácil de ajustar |

De la revisión se desprenden seis patrones:

1. **La regresión logística aparece en casi todos los estudios** como referencia. Es el estándar de la industria y es interpretable, pero rara vez es el mejor modelo.
2. **Los ensambles de árboles, bosques aleatorios y boosting, ganan o empatan con el mejor** en la mayoría de las comparaciones sobre datos tabulares de clientes.
3. **Las redes neuronales son competitivas en algunos estudios**, pero en datos tabulares no superan de forma consistente a los ensambles de árboles y requieren más ajuste.
4. **Las variables pesan más que el algoritmo.** Liu et al. y Ahmad et al. lo muestran directamente, y nuestro propio proyecto también: con el mismo algoritmo, las variables avanzadas subieron el AUC de 0.561 a 0.645, y haber comprado antes el producto exacto sube la tasa de recompra de 24.2% a 42.8%.
5. **El AUC es la métrica estándar** cuando las clases están desbalanceadas, y la validación debe respetar el tiempo, porque los modelos se aplican a clientes de periodos posteriores.
6. **Generar datos sintéticos no ha mostrado ventajas claras frente al desbalance.** Verbeke et al. no encontraron mejoras significativas con sobremuestreo, y la mejora de Zhang et al. combinando SMOTE con apilamiento fue pequeña. Ponderar las clases dentro del modelo es la alternativa más simple.

## 4. Arquitecturas y variantes de los algoritmos candidatos

### 4.1 Regresión logística

La regresión logística modela la probabilidad de la clase positiva como una función logística de una combinación lineal de las variables (Cox, 1958). Sus variantes principales se distinguen por la regularización: L2 encoge todos los coeficientes, L1 lleva algunos a cero y funciona como selección de variables. Supone que cada variable tiene un efecto lineal y aditivo sobre el logaritmo de la razón de probabilidades, así que las colas largas y los valores extremos deben transformarse antes de entrenar.

### 4.2 Bosque aleatorio

El bosque aleatorio entrena muchos árboles de decisión, cada uno sobre una muestra con reemplazo de los datos y considerando en cada división un subconjunto aleatorio de variables. La predicción final promedia los árboles (Breiman, 2001). Captura relaciones no lineales e interacciones sin especificarlas, es poco sensible a valores extremos y no necesita escalar las variables. Sus hiperparámetros principales son el número de árboles, la profundidad, el tamaño mínimo de las hojas y la fracción de variables por división.

### 4.3 Gradient boosting

El gradient boosting construye árboles en secuencia, cada uno corrigiendo los errores de los anteriores siguiendo el gradiente de una función de pérdida (Friedman, 2001). Existen varias implementaciones con arquitecturas distintas:

- **GBM clásico** de Friedman (2001): árboles completos en secuencia. Es preciso pero lento con muchos datos.
- **XGBoost** de Chen y Guestrin (2016): agrega regularización a la función objetivo, un algoritmo que aprovecha datos dispersos y aproximaciones por cuantiles para escalar a conjuntos grandes.
- **LightGBM** de Ke et al. (2017): agrupa los valores de cada variable en histogramas y usa muestreo por gradiente y agrupación de variables exclusivas. Es mucho más rápido que XGBoost con precisión similar.
- **HistGradientBoostingClassifier** de scikit-learn (Pedregosa et al., 2011): implementación inspirada en LightGBM. Según su documentación, es mucho más rápida que el GBM clásico cuando hay 10,000 observaciones o más. Además maneja los valores faltantes de forma nativa, porque en cada división aprende hacia qué rama enviarlos.

### 4.4 Perceptrón multicapa

El perceptrón multicapa es una red neuronal de capas totalmente conectadas que se entrena con retropropagación del error (Rumelhart et al., 1986). Su arquitectura se define por el número de capas ocultas, las neuronas por capa, la función de activación y la regularización. Puede aprender relaciones no lineales, pero es sensible a la escala de las variables, a los valores extremos y a la elección de hiperparámetros. Por eso necesita escalamiento y detención temprana del entrenamiento.

## 5. Selección de algoritmos

### 5.1 Características del problema que condicionan la elección

| Característica | Valor en nuestros datos | Consecuencia para el algoritmo |
|---|---|---|
| Tipo de datos | Tabulares, una fila por cliente | Favorece ensambles de árboles |
| Tamaño | 160,057 clientes con etiqueta | Descarta métodos cuyo costo crece mucho con el número de filas |
| Variables | 37 de comportamiento y 5 identificadores anónimos | Pocas categorías: 130 cadenas, 34 mercados, 13 categorías |
| Valores faltantes | 45.38% en la recencia de la categoría, y significan que el cliente nunca la compró | Conviene un algoritmo que use el faltante como información |
| Colas largas y extremos | 161 clientes con volúmenes de compra extremos | Los árboles los toleran; logística y red neuronal necesitan transformación |
| Desbalance | 27.14% de positivos | Ponderar las clases y medir con AUC |
| Cambio en el tiempo | La recompra pasa de 24.64% a 35.91% en la última semana | Validación temporal, nunca al azar |
| Uso en la aplicación | La app debe clasificar clientes nuevos al instante | El modelo debe dar probabilidades y predecir rápido |

### 5.2 Algoritmos seleccionados

Se seleccionan cuatro algoritmos que cubren las familias que la literatura identifica como relevantes: un modelo lineal, un ensamble por bagging, un ensamble por boosting y una red neuronal. Los cuatro son algoritmos de aprendizaje automático, conocido en inglés como machine learning, y ninguno es de aprendizaje profundo.

| Algoritmo | Papel en el proyecto | Respaldo en la literatura | Ventaja para este problema |
|---|---|---|---|
| Regresión logística | Referencia e interpretación | Cox, 1958; Buckinx y Van den Poel, 2005; De Caigny et al., 2018 | Ya existe en el proyecto y sus coeficientes explican el efecto de cada variable |
| Bosque aleatorio | Punto de comparación exigente | Breiman, 2001; Larivière y Van den Poel, 2005; Lessmann et al., 2015 | Captura interacciones y tolera los valores extremos sin transformarlos |
| Gradient boosting con HistGradientBoostingClassifier | Candidato principal | Friedman, 2001; Lemmens y Croux, 2006; Martínez et al., 2020; Ahmad et al., 2019; Shwartz-Ziv y Armon, 2022 | Fue el mejor en los estudios más parecidos; maneja los faltantes de forma nativa y entrena rápido |
| Perceptrón multicapa | Representante de las redes neuronales | Rumelhart et al., 1986; Lessmann et al., 2015; Buckinx y Van den Poel, 2005 | Permite comprobar si una red neuronal supera a los árboles en estos datos |

Para gradient boosting se elige la implementación de scikit-learn y no XGBoost ni LightGBM por tres razones. Es una implementación inspirada en LightGBM, así que conserva su arquitectura. Ya está instalada en el entorno del proyecto, mientras que XGBoost y LightGBM no lo están. Y usa la misma interfaz que los demás modelos, lo que simplifica guardarlos y cargarlos en la aplicación. Si el grupo logra instalar LightGBM, puede agregarse como quinto modelo para comparar implementaciones.

La expectativa para el perceptrón multicapa es moderada. Shwartz-Ziv y Armon (2022) y Grinsztajn et al. (2022) muestran que en datos tabulares las redes rara vez superan a los ensambles de árboles. Se incluye porque Lessmann et al. (2015) encontraron que superan a la regresión logística, y porque comparar las cuatro familias permite justificar la elección final con evidencia propia y no solo con la literatura.

### 5.3 Algoritmos descartados

| Algoritmo | Motivo |
|---|---|
| Máquinas de soporte vectorial con kernel | Su costo de entrenamiento crece rápidamente con el número de observaciones y no entregan probabilidades de forma directa. Verbeke et al. (2012) encontraron desempeños comparables entre muchos clasificadores, así que el costo no se compensa |
| Naive Bayes | Supone variables independientes, y las nuestras están muy relacionadas: las compras de 30, 60, 90 y 180 días son ventanas anidadas |
| Redes profundas para datos tabulares | Shwartz-Ziv y Armon (2022) y Grinsztajn et al. (2022) muestran que no superan a los ensambles de árboles y requieren mucho más ajuste |
| Máquinas de factorización | Están pensadas para interacciones entre categorías con muchísimos niveles, como los pares usuario y comercio de Tmall. Aquí hay pocas: 130 cadenas y 13 categorías |
| SMOTE y sobremuestreo | Verbeke et al. (2012) no encontraron mejoras significativas; se prefiere ponderar las clases dentro de cada modelo |

Los ensambles que combinan varios algoritmos no se descartan, se dejan para una etapa posterior. Lessmann et al. (2015) obtuvieron su mejor resultado con un ensamble heterogéneo y Zhang et al. (2022) con apilamiento, pero la ganancia de Zhang et al. fue de solo 0.01161 en AUC y complican la aplicación. Conviene evaluarlos solo cuando los cuatro modelos individuales estén terminados.

### 5.4 Ajuste de hiperparámetros y evaluación

La revisión también define cómo comparar los modelos:

- **División temporal en tres conjuntos.** Entrenamiento con las ofertas del 1 de marzo al 16 de abril de 2013, validación del 17 al 23 de abril para elegir hiperparámetros y prueba del 24 al 30 de abril para reportar resultados. Esto sigue la conclusión de Neslin et al. (2006): un modelo se aplica a periodos posteriores y debe evaluarse así.
- **Búsqueda aleatoria de hiperparámetros.** Bergstra y Bengio (2012) mostraron que la búsqueda aleatoria encuentra configuraciones tan buenas o mejores que la búsqueda en rejilla con el mismo presupuesto de cómputo.
- **AUC como métrica principal** (Fawcett, 2006; Burez y Van den Poel, 2009), complementada con precisión, sensibilidad, F1, matriz de confusión y tiempos de entrenamiento y predicción.

Rangos sugeridos para la búsqueda:

| Algoritmo | Hiperparámetros a ajustar |
|---|---|
| Regresión logística | Fuerza de regularización C entre 0.001 y 10; penalización L1 o L2; pesos balanceados por clase |
| Bosque aleatorio | Entre 300 y 800 árboles; profundidad máxima; mínimo de observaciones por hoja entre 1 y 50; fracción de variables por división; pesos balanceados por clase |
| HistGradientBoosting | Tasa de aprendizaje entre 0.02 y 0.2; número máximo de hojas entre 15 y 127; mínimo de observaciones por hoja entre 20 y 200; regularización L2; detención temprana |
| Perceptrón multicapa | Capas ocultas de 64, de 128 y 64, o de 64 y 32 neuronas; regularización alfa entre 0.00001 y 0.01; tasa de aprendizaje inicial; detención temprana |

Los resultados de otros estudios no son directamente comparables con los nuestros. Los AUC de 0.93 y 0.95 que reportan Ahmad et al. (2019) y Martínez et al. (2020) corresponden a otros datos y otras definiciones del evento. La referencia válida para este proyecto es la tabla final de la competencia de Kaggle, donde el primer lugar obtuvo un AUC cercano a 0.63 (Kaggle, 2014), además de nuestra propia línea base.

## 6. Aporte al informe

- **Marco teórico.** Las secciones 3 y 4 aportan la teoría de cada algoritmo y los antecedentes de su uso en problemas de recompra, abandono y cupones.
- **Metodología.** La sección 5 aporta la justificación de los algoritmos elegidos, la división en entrenamiento, validación y prueba, y el método de ajuste de hiperparámetros.
- **Conclusiones.** El patrón de que las variables pesan más que el algoritmo sirve para discutir los resultados cuando se comparen los cuatro modelos.

---

## 7. Referencias

Ahmad, A. K., Jafar, A., & Aljoumaa, K. (2019). Customer churn prediction in telecom using machine learning in big data platform. *Journal of Big Data, 6*(1), Artículo 28. https://journalofbigdata.springeropen.com/articles/10.1186/s40537-019-0191-6

Bergstra, J., & Bengio, Y. (2012). Random search for hyper-parameter optimization. *Journal of Machine Learning Research, 13*(10), 281–305. https://jmlr.org/papers/v13/bergstra12a.html

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://link.springer.com/article/10.1023/A:1010933404324

Buckinx, W., & Van den Poel, D. (2005). Customer base analysis: Partial defection of behaviourally loyal clients in a non-contractual FMCG retail setting. *European Journal of Operational Research, 164*(1), 252–268. https://www.sciencedirect.com/science/article/pii/S0377221703009184

Burez, J., & Van den Poel, D. (2009). Handling class imbalance in customer churn prediction. *Expert Systems with Applications, 36*(3), 4626–4636. https://www.sciencedirect.com/science/article/pii/S0957417408002121

Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. En *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 785–794). ACM. https://www.kdd.org/kdd2016/papers/files/rfp0697-chenAemb.pdf

Cox, D. R. (1958). The regression analysis of binary sequences. *Journal of the Royal Statistical Society: Series B (Methodological), 20*(2), 215–232. https://academic.oup.com/jrsssb/article/20/2/215/7027376

De Caigny, A., Coussement, K., & De Bock, K. W. (2018). A new hybrid classification algorithm for customer churn prediction based on logistic regression and decision trees. *European Journal of Operational Research, 269*(2), 760–772. https://www.sciencedirect.com/science/article/pii/S0377221718301243

Fawcett, T. (2006). An introduction to ROC analysis. *Pattern Recognition Letters, 27*(8), 861–874. https://www.sciencedirect.com/science/article/pii/S016786550500303X

Friedman, J. H. (2001). Greedy function approximation: A gradient boosting machine. *The Annals of Statistics, 29*(5), 1189–1232. https://projecteuclid.org/journals/annals-of-statistics/volume-29/issue-5/Greedy-function-approximation-A-gradient-boosting-machine/10.1214/aos/1013203451.full

Grinsztajn, L., Oyallon, E., & Varoquaux, G. (2022). Why do tree-based models still outperform deep learning on typical tabular data? En *Advances in Neural Information Processing Systems 35* (pp. 507–520). Curran Associates. https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html

Kaggle. (2014). *Acquire valued shoppers challenge. Predict which shoppers will become repeat buyers* [Conjunto de datos y descripción de la competencia]. https://www.kaggle.com/competitions/acquire-valued-shoppers-challenge/data

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.-Y. (2017). LightGBM: A highly efficient gradient boosting decision tree. En *Advances in Neural Information Processing Systems 30*. Curran Associates. https://proceedings.neurips.cc/paper/2017/hash/6449f44a102fde848669bdd9eb6b76fa-Abstract.html

Larivière, B., & Van den Poel, D. (2005). Predicting customer retention and profitability by using random forests and regression forests techniques. *Expert Systems with Applications, 29*(2), 472–484. https://www.sciencedirect.com/science/article/pii/S0957417405000965

Lemmens, A., & Croux, C. (2006). Bagging and boosting classification trees to predict churn. *Journal of Marketing Research, 43*(2), 276–286. https://journals.sagepub.com/doi/10.1509/jmkr.43.2.276

Lessmann, S., Baesens, B., Seow, H.-V., & Thomas, L. C. (2015). Benchmarking state-of-the-art classification algorithms for credit scoring: An update of research. *European Journal of Operational Research, 247*(1), 124–136. https://www.sciencedirect.com/science/article/pii/S0377221715004208

Liu, G., Nguyen, T. T., Zhao, G., Zha, W., Yang, J., Cao, J., Wu, M., Zhao, P., & Chen, W. (2016). Repeat buyer prediction for e-commerce. En *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 155–164). ACM. https://www.kdd.org/kdd2016/papers/files/adf0160-liuA.pdf

Martínez, A., Schmuck, C., Pereverzyev, S., Pirker, C., & Haltmeier, M. (2020). A machine learning framework for customer purchase prediction in the non-contractual setting. *European Journal of Operational Research, 281*(3), 588–596. https://www.sciencedirect.com/science/article/pii/S0377221718303370

Neslin, S. A., Gupta, S., Kamakura, W., Lu, J., & Mason, C. H. (2006). Defection detection: Measuring and understanding the predictive accuracy of customer churn models. *Journal of Marketing Research, 43*(2), 204–211. https://journals.sagepub.com/doi/10.1509/jmkr.43.2.204

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P., Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*(85), 2825–2830. https://jmlr.org/papers/v12/pedregosa11a.html

Ren, X., Cao, J., Xu, X., & Gong, Y. (2021). A two-stage model for forecasting consumers' intention to purchase with e-coupons. *Journal of Retailing and Consumer Services, 59*, Artículo 102289. https://www.sciencedirect.com/science/article/pii/S0969698920312972

Rumelhart, D. E., Hinton, G. E., & Williams, R. J. (1986). Learning representations by back-propagating errors. *Nature, 323*(6088), 533–536. https://www.nature.com/articles/323533a0

Shwartz-Ziv, R., & Armon, A. (2022). Tabular data: Deep learning is not all you need. *Information Fusion, 81*, 84–90. https://www.sciencedirect.com/science/article/pii/S1566253521002360

Vafeiadis, T., Diamantaras, K. I., Sarigiannidis, G., & Chatzisavvas, K. C. (2015). A comparison of machine learning techniques for customer churn prediction. *Simulation Modelling Practice and Theory, 55*, 1–9. https://www.sciencedirect.com/science/article/pii/S1569190X15000386

Verbeke, W., Dejaeger, K., Martens, D., Hur, J., & Baesens, B. (2012). New insights into churn prediction in the telecommunication sector: A profit driven data mining approach. *European Journal of Operational Research, 218*(1), 211–229. https://www.sciencedirect.com/science/article/pii/S0377221711008599

Zhang, M., Lu, J., Ma, N., Cheng, T. C. E., & Hua, G. (2022). A feature engineering and ensemble learning based approach for repeated buyers prediction. *International Journal of Computers Communications & Control, 17*(6), Artículo 4988. https://univagora.ro/jour/index.php/ijccc/article/view/4988
