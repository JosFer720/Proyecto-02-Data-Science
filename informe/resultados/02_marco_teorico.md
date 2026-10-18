# Marco teórico

La guía del proyecto propone la estructura del marco teórico según el tipo de reto. Este proyecto pertenece a la categoría de negocios, así que se adapta la estructura propuesta para los sistemas de recomendación en sus tres partes, que son la definición teórica del problema, las actividades de preprocesamiento general y los algoritmos de aprendizaje automático usados para resolverlo. La evaluación de los algoritmos se incluye al final de la tercera parte.

## 1. Definición teórica de la predicción de compradores recurrentes

### 1.1 El valor del cliente que regresa

Una empresa que conserva a sus clientes depende menos de atraer clientes nuevos, algo que suele costar más. Christy et al. (2021) señalan que retener a los clientes es más importante que encontrar clientes nuevos, y que agruparlos según su comportamiento permite dirigir estrategias de mercadeo a cada grupo.

La recompra es la forma más directa de retención en el comercio minorista. Zhang et al. (2022) reportan que en la plataforma Tmall solo el 6.1% de los compradores vuelve a comprar, y citan estudios según los cuales un aumento de cinco por ciento en la tasa de recompra puede subir las utilidades entre 25% y 95%. Predecir bien también tiene consecuencias económicas directas. En un torneo de modelos de abandono de clientes, Neslin et al. (2006) encontraron que las diferencias de exactitud entre propuestas cambiaban la ganancia de una campaña en cientos de miles de dólares.

### 1.2 Promociones y cazadores de ofertas

Las promociones atraen muchos compradores nuevos, pero una parte de ellos solo aprovecha el descuento y no vuelve. Liu et al. (2016) describen a estos clientes como cazadores de ofertas de una sola vez, y señalan que identificar a quienes pueden convertirse en compradores leales permite dirigirles las promociones, bajar su costo y aumentar el retorno de la inversión. El problema de este proyecto es el mismo, porque a cada cliente se le ofreció un cupón y se busca anticipar si volverá a comprar el producto después de usarlo.

Los cupones también se estudian como un problema de selección de clientes. Ren et al. (2021) proponen decidir a qué consumidores enviar cupones electrónicos a partir de su comportamiento anterior, primero separándolos en grupos y después ajustando un modelo distinto para cada grupo.

### 1.3 Relación no contractual y comportamiento de compra

En un supermercado la relación con el cliente es no contractual, porque el cliente no avisa cuando deja de comprar, simplemente deja de hacerlo. Por eso la empresa tiene que estimar su lealtad a partir de su historial. Buckinx y Van den Poel (2005) estudiaron este caso en una tienda de productos de consumo masivo y encontraron que las variables de comportamiento pasado fueron las mejores para predecir el abandono parcial de los clientes más leales. Martínez et al. (2020) predijeron si un cliente comprará en el mes siguiente a partir del momento y el valor de sus compras anteriores.

### 1.4 Modelo RFM

El modelo RFM resume el comportamiento de un cliente en tres medidas calculadas con sus compras (Christy et al., 2021). La recencia es el tiempo que pasó desde su última compra, la frecuencia es el número de compras o visitas en un periodo y el monto es lo que gastó en total.

En el estudio de Buckinx y Van den Poel (2005), estas tres medidas fueron las mejores para predecir. Pero describen al cliente en general y no su relación con un producto específico. En un problema de recompra conviene agregar el historial del cliente con la categoría, la marca y el producto de la oferta, de forma parecida a los perfiles de usuarios, comercios, marcas, categorías y productos que construyeron Liu et al. (2016).

### 1.5 Planteamiento como clasificación binaria

El problema se plantea como una clasificación binaria. Cada cliente tiene una variable respuesta con dos valores, recurrente si volvió a comprar el producto de la oferta al menos una vez, o no recurrente si no volvió. A partir del historial anterior a la oferta, un modelo de aprendizaje automático estima la probabilidad de que el cliente sea recurrente. Con esa probabilidad, la empresa puede ordenar a sus clientes y dar prioridad a los de mayor probabilidad en una campaña. La competencia original mide los modelos con el área bajo la curva ROC (Kaggle, 2014).

Aunque el problema tiene relación con el mercadeo, se diferencia de un sistema de recomendación. Un sistema de recomendación ordena muchos productos para cada cliente según sus gustos. En este proyecto cada cliente recibió una sola oferta y lo que se estima es si volverá a comprar ese producto. Las probabilidades del modelo sí se podrían usar después para decidir a quién enviar cupones.

## 2. Actividades de preprocesamiento general

Los datos de una tienda suelen llegar como registros de compras, con una fila por cada línea de compra. Antes de entrenar un modelo hay que convertirlos en una tabla con una fila por cliente.

### 2.1 Unión y resumen de las compras por cliente

La primera actividad es unir las compras con la información de cada cliente y de cada oferta, y resumirlas por cliente. Liu et al. (2016) concluyen que la construcción de variables fue lo más importante de su solución, y Ahmad et al. (2019) muestran que agregar nuevas variables puede subir el AUC de un modelo de 84% a 93.3%. Por eso el preprocesamiento es una de las partes más importantes del modelado.

Al resumir las compras hay dos cuidados importantes. El primero es usar solo el historial anterior a la oferta. Si se incluyen compras posteriores, el modelo aprendería información que no existía al momento de decidir, algo que se conoce como fuga de información. El segundo es respetar las ventanas de tiempo. Las variables calculadas en periodos recientes, como los 30, 60, 90 o 180 días antes de la oferta, muestran cambios de comportamiento que el total histórico esconde.

### 2.2 Calidad de los datos

Las compras reales traen registros que no son compras normales, como montos negativos por devoluciones o descuentos, cantidades en cero y unidades de medida vacías. La decisión para cada caso depende de lo que significa. Por ejemplo, un descuento forma parte del gasto neto del cliente y no debe eliminarse.

Los valores faltantes también pueden tener significado. Si un cliente nunca compró la categoría de la oferta, los días desde su última compra de esa categoría no existen, y esa ausencia dice algo sobre el cliente. En esos casos se conserva el faltante y se agrega una variable que lo marca, en lugar de eliminar la fila.

Los valores extremos son comunes en el gasto y en la frecuencia de compra, porque unos pocos clientes compran muchísimo más que el resto. En lugar de eliminarlos, se puede reducir su peso con una transformación logarítmica, que acerca los valores grandes al resto y conserva el orden de los clientes.

### 2.3 Preparación según el algoritmo

Cada familia de algoritmos necesita una preparación distinta. La regresión logística y las redes neuronales son sensibles a la escala de las variables, por lo que se estandarizan. Los identificadores de cadena, mercado, categoría, compañía y marca son grupos sin orden, así que se convierten en columnas binarias o en códigos que el modelo trata como grupos. Los modelos lineales y las redes necesitan que se rellenen los valores faltantes, mientras que algunas versiones de gradient boosting los manejan por sí solas (Pedregosa et al., 2011). Los árboles de decisión y sus ensambles no dependen de la escala, así que pueden usar las variables sin transformar.

### 2.4 Desbalance de clases

En los problemas de retención y recompra, la clase de interés suele ser la minoritaria. Con pocos casos positivos, un modelo que siempre prediga la clase mayoritaria alcanza una exactitud alta sin aportar nada útil.

Hay tres formas comunes de manejar el desbalance. La primera es ponderar las clases dentro del modelo, para que los errores en la clase minoritaria pesen más. La segunda es reducir la clase mayoritaria con un submuestreo. Lemmens y Croux (2006) recomiendan un muestreo balanceado cuando el evento es poco frecuente, junto con una corrección del sesgo que produce. La tercera es aumentar la clase minoritaria con un sobremuestreo, por ejemplo con el método SMOTE, que genera ejemplos artificiales de esa clase (Chawla et al., 2002).

La evidencia sobre el sobremuestreo no es clara. Verbeke et al. (2012) no encontraron mejoras significativas al aplicarlo en once conjuntos de datos de abandono de clientes. Burez y Van den Poel (2009) compararon métodos de muestreo con técnicas como los bosques aleatorios ponderados, y recomiendan medir con el AUC, que no depende de un umbral de clasificación.

## 3. Algoritmos de aprendizaje automático

### 3.1 Antecedentes en problemas de negocio parecidos

Los estudios de predicción de recompra, abandono de clientes y respuesta a cupones han usado muchos algoritmos distintos, como se resume en la siguiente tabla.

| Estudio | Problema | Algoritmos | Resultado principal |
|---|---|---|---|
| Liu et al., 2016 | Recompra después de una promoción | Construcción de variables y entrenamiento de modelos | Las variables fueron lo decisivo |
| Zhang et al., 2022 | Recompra después de una promoción | Factorización, regresión logística, XGBoost, LightGBM y apilamiento | El apilamiento subió el AUC en 0.01161 |
| Martínez et al., 2020 | Compra del mes siguiente | Varios métodos de aprendizaje automático | Gradient boosting fue el mejor |
| Larivière y Van den Poel, 2005 | Siguiente compra y abandono | Bosque aleatorio contra regresiones | El bosque aleatorio ajustó mejor |
| Buckinx y Van den Poel, 2005 | Abandono parcial en una tienda | Regresión logística, red neuronal y bosque aleatorio | Recencia, frecuencia y monto fueron las mejores variables |
| Lemmens y Croux, 2006 | Abandono en telefonía | Bagging y boosting de árboles | Ambos mejoraron la predicción |
| Verbeke et al., 2012 | Abandono en telefonía | Muchos clasificadores | Muchos clasificadores rindieron parecido |
| De Caigny et al., 2018 | Abandono | Árbol, regresión logística, bosque aleatorio y un modelo híbrido | El híbrido igualó al bosque aleatorio |
| Ahmad et al., 2019 | Abandono en telefonía | Árbol, bosque aleatorio, gradient boosting y XGBoost | XGBoost fue el mejor |
| Ren et al., 2021 | Uso de cupones electrónicos | Grupos de clientes, bosque de aislamiento y regresión logística | Separar en grupos redujo el desbalance |
| Lessmann et al., 2015 | Riesgo de crédito | 41 clasificadores | El bosque aleatorio es el punto de comparación recomendado |

En estos estudios se repiten tres cosas. La regresión logística aparece como referencia en casi todos. Los ensambles de árboles, ya sea por bagging o por boosting, obtienen los mejores resultados o empatan con ellos en la mayoría de las comparaciones con datos de clientes en tablas. Las redes neuronales compiten bien en algunos estudios, y Lessmann et al. (2015) encontraron que superan a la regresión logística. Pero en datos de tablas no superan de forma constante a los ensambles de árboles (Shwartz-Ziv y Armon, 2022; Grinsztajn et al., 2022).

### 3.2 Regresión logística

La regresión logística calcula la probabilidad de la clase positiva con una función logística aplicada a una combinación lineal de las variables (Cox, 1958). Sus coeficientes indican cómo cambia la probabilidad con cada variable, lo que la hace fácil de interpretar. Se puede regularizar para evitar el sobreajuste. La penalización L2 encoge todos los coeficientes y la L1 lleva algunos a cero. Supone que el efecto de cada variable es lineal en la escala logarítmica de las probabilidades, por lo que necesita variables transformadas y escaladas.

### 3.3 Bosque aleatorio

El bosque aleatorio entrena muchos árboles de decisión, cada uno con una muestra al azar de los datos y usando en cada división un grupo al azar de variables. La predicción final es el promedio de los árboles (Breiman, 2001). Puede encontrar relaciones no lineales e interacciones entre variables sin que haya que indicarlas, le afectan poco los valores extremos y no necesita escalar las variables. Sus hiperparámetros principales son el número de árboles, la profundidad máxima, el tamaño mínimo de las hojas y la fracción de variables en cada división.

### 3.4 Gradient boosting

El gradient boosting construye árboles uno después de otro, y cada árbol corrige los errores de los anteriores siguiendo el gradiente de una función de pérdida (Friedman, 2001). Existen varias versiones. XGBoost agrega regularización y técnicas para trabajar con conjuntos de datos grandes y dispersos (Chen y Guestrin, 2016). LightGBM agrupa los valores de cada variable en histogramas, lo que hace el entrenamiento mucho más rápido con una precisión parecida (Ke et al., 2017). HistGradientBoostingClassifier, de scikit-learn, se inspira en LightGBM y maneja por sí solo los valores faltantes y las variables categóricas (Pedregosa et al., 2011). Sus hiperparámetros principales son la tasa de aprendizaje, el número máximo de hojas por árbol, el tamaño mínimo de las hojas y la regularización.

### 3.5 Perceptrón multicapa

El perceptrón multicapa es una red neuronal de capas totalmente conectadas que se entrena con retropropagación del error (Rumelhart et al., 1986). Su arquitectura se define por el número de capas ocultas, el número de neuronas en cada capa y la regularización. Puede aprender relaciones no lineales, pero es sensible a la escala de las variables y a la elección de hiperparámetros, por lo que necesita escalamiento y detener el entrenamiento a tiempo. Con pocas capas ocultas no se considera aprendizaje profundo.

### 3.6 Curva ROC y AUC

La curva ROC muestra la tasa de verdaderos positivos frente a la tasa de falsos positivos para todos los umbrales posibles. El área bajo esa curva, llamada AUC, mide qué tan bien el modelo pone a los casos positivos por encima de los negativos. Un valor de 0.5 equivale al azar y un valor de 1 a una separación perfecta (Fawcett, 2006). Como no depende de un umbral, es la medida adecuada cuando las clases están desbalanceadas (Burez y Van den Poel, 2009).

### 3.7 Métricas con umbral

Al fijar un umbral, cada cliente se clasifica como recurrente o no, y se arma la matriz de confusión con los verdaderos positivos, los falsos positivos, los falsos negativos y los verdaderos negativos. De ella salen tres medidas. La precisión es la proporción de clientes clasificados como recurrentes que en verdad lo son. La sensibilidad es la proporción de clientes recurrentes que el modelo detecta. El F1 es la media armónica de la precisión y la sensibilidad, o sea un promedio que solo es alto cuando las dos lo son.

### 3.8 Validación temporal

Un modelo de clientes se usa para predecir lo que pasará en periodos futuros, así que debe evaluarse con datos posteriores a los de entrenamiento. Neslin et al. (2006) mostraron que los modelos de abandono mantienen su desempeño cuando se aplican a datos recolectados tres meses después de los de entrenamiento, lo que respalda esta forma de evaluar.

### 3.9 Ajuste de hiperparámetros

Los hiperparámetros se eligen con un conjunto de validación separado del de prueba. Bergstra y Bengio (2012) mostraron que la búsqueda aleatoria encuentra configuraciones tan buenas o mejores que la búsqueda en rejilla con el mismo tiempo de cómputo.
