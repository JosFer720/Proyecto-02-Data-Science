# Marco teórico

La guía del proyecto propone la estructura del marco teórico según el tipo de reto. Este proyecto pertenece a la categoría de negocios, así que se adapta la estructura propuesta para los sistemas de recomendación en sus tres partes: la definición teórica del problema, las actividades de preprocesamiento general y los algoritmos de aprendizaje automático usados para resolverlo. La evaluación de los algoritmos se incluye al final de la tercera parte.

## 1. Definición teórica: predicción de compradores recurrentes

### 1.1 El valor del cliente que regresa

Una empresa que conserva a sus clientes depende menos de atraer clientes nuevos, que suele ser más costoso. Christy et al. (2021) señalan que retener a los clientes es más importante que encontrar clientes nuevos y que segmentarlos según su comportamiento permite dirigir estrategias de mercadeo a cada grupo.

La recompra es la forma más directa de retención en el comercio minorista. Zhang et al. (2022) reportan que en la plataforma Tmall solo el 6.1% de los compradores vuelve a comprar, y citan estudios según los cuales un aumento de cinco por ciento en la tasa de recompra puede elevar las utilidades entre 25% y 95%. La predicción también tiene consecuencias económicas directas: en un torneo de modelos de abandono, Neslin et al. (2006) encontraron que las diferencias de exactitud entre propuestas cambiaban la rentabilidad de una campaña en cientos de miles de dólares.

### 1.2 Promociones y cazadores de ofertas

Las promociones atraen muchos compradores nuevos, pero una parte de ellos solo aprovecha el descuento y no vuelve. Liu et al. (2016) describen a estos clientes como cazadores de ofertas de una sola vez y señalan que identificar a quienes pueden convertirse en compradores leales permite dirigirles las promociones, reducir su costo y aumentar el retorno de la inversión. El problema de este proyecto es el mismo: a cada cliente se le ofreció un cupón y se busca anticipar si volverá a comprar el producto después de usarlo.

Los cupones también se estudian como un problema de selección de clientes. Ren et al. (2021) proponen decidir a qué consumidores enviar cupones electrónicos a partir de su comportamiento anterior, primero segmentándolos y después ajustando un modelo distinto para cada segmento.

### 1.3 Relación no contractual y comportamiento de compra

En un supermercado la relación con el cliente es no contractual: el cliente no avisa cuando deja de comprar, solo deja de hacerlo. Por eso la empresa debe inferir su lealtad a partir de su historial. Buckinx y Van den Poel (2005) estudiaron este entorno en un minorista de productos de consumo masivo y encontraron que las variables de comportamiento pasado fueron las mejores para predecir el abandono parcial de los clientes más leales. Martínez et al. (2020) predijeron si un cliente comprará en el mes siguiente a partir del momento y el valor de sus compras anteriores.

### 1.4 Modelo RFM

El modelo RFM resume el comportamiento de un cliente en tres medidas calculadas a partir de sus transacciones (Christy et al., 2021):

- **Recencia:** tiempo transcurrido desde la última compra.
- **Frecuencia:** número de compras o visitas en un periodo.
- **Monto:** gasto acumulado del cliente.

En el estudio de Buckinx y Van den Poel (2005), estas tres medidas fueron las mejores variables predictoras. Sin embargo, describen al cliente en general y no su relación con un producto específico. En un problema de recompra conviene agregar el historial del cliente con la categoría, la marca y el producto ofrecidos, de forma similar a los perfiles de usuarios, comercios, marcas, categorías y productos que construyeron Liu et al. (2016).

### 1.5 Formulación como clasificación binaria

El problema se formula como una clasificación binaria. Cada cliente tiene una variable respuesta con dos valores: recurrente, si volvió a comprar el producto ofertado al menos una vez, o no recurrente. A partir del historial previo a la oferta, un modelo de aprendizaje automático estima la probabilidad de que el cliente sea recurrente. Con esa probabilidad, la empresa puede ordenar a sus clientes y priorizar a los de mayor probabilidad en una campaña. La competencia original mide los modelos con el área bajo la curva ROC (Kaggle, 2014).

Aunque el problema se relaciona con el mercadeo, no es un sistema de recomendación. Un sistema de recomendación ordena muchos productos para cada cliente según sus preferencias. Aquí cada cliente recibió una sola oferta y lo que se estima es si volverá a comprar ese producto. Las probabilidades del modelo sí podrían usarse después para decidir a quién enviar cupones.

## 2. Actividades de preprocesamiento general

Los datos de un negocio minorista suelen llegar como bitácoras de transacciones, con una fila por línea de compra. Antes de entrenar un modelo hay que transformarlas en una tabla con una fila por cliente.

### 2.1 Integración y agregación por cliente

La primera actividad es unir las transacciones con la información de cada cliente y de cada oferta, y resumirlas por cliente. Liu et al. (2016) concluyen que la construcción de variables fue el factor más importante de su solución, y Ahmad et al. (2019) muestran que agregar nuevas variables puede subir el AUC de un modelo de 84% a 93.3%. Por eso el preprocesamiento no es un paso mecánico, sino una parte central del modelado.

Dos cuidados son indispensables al agregar:

- **Usar solo el historial previo a la oferta.** Si se incluyen compras posteriores, el modelo aprendería información que no existía al momento de decidir, lo que se conoce como fuga de información.
- **Respetar las ventanas de tiempo.** Las variables calculadas en ventanas recientes, como 30, 60, 90 o 180 días antes de la oferta, capturan cambios de comportamiento que el total histórico oculta.

### 2.2 Calidad de los datos

Las transacciones reales contienen registros que no son compras normales: montos negativos por devoluciones o descuentos, cantidades en cero y unidades de medida vacías. La decisión para cada caso depende de su significado; por ejemplo, un descuento forma parte del gasto neto del cliente y no debe eliminarse.

Los **valores faltantes** también pueden tener significado. Si un cliente nunca compró la categoría ofertada, la recencia de esa categoría no existe, y esa ausencia es información sobre el cliente. En esos casos se conserva el faltante y se agrega una variable que lo marca, en lugar de eliminar la fila.

Los **valores atípicos** son comunes en el gasto y la frecuencia de compra, que tienen colas largas. En lugar de eliminarlos, se puede reducir su influencia con una transformación logarítmica, que comprime los valores grandes y conserva el orden de los clientes.

### 2.3 Transformación según el algoritmo

Cada familia de algoritmos necesita una preparación distinta:

- **Escalado.** La regresión logística y las redes neuronales son sensibles a la escala de las variables, por lo que se estandarizan.
- **Codificación de categorías.** Los identificadores de cadena, mercado, categoría, compañía y marca son grupos sin orden, así que se convierten en columnas binarias o en códigos que el modelo trata como grupos.
- **Faltantes.** Los modelos lineales y las redes requieren imputarlos. Algunas implementaciones de gradient boosting los manejan de forma nativa (Pedregosa et al., 2011).

Los árboles de decisión y sus ensambles no dependen de la escala, así que pueden usar las variables sin transformar.

### 2.4 Desbalance de clases

En los problemas de retención y recompra la clase de interés suele ser minoritaria. Con pocos casos positivos, un modelo que siempre prediga la clase mayoritaria alcanza una exactitud alta sin aportar información. Hay tres formas habituales de manejarlo:

- **Ponderar las clases** dentro del modelo, para que los errores en la clase minoritaria pesen más.
- **Submuestrear la clase mayoritaria.** Lemmens y Croux (2006) recomiendan un muestreo balanceado cuando el evento es poco frecuente, acompañado de una corrección del sesgo que introduce.
- **Sobremuestrear la clase minoritaria.** El método SMOTE genera ejemplos sintéticos de la clase minoritaria (Chawla et al., 2002).

La evidencia sobre el sobremuestreo no es concluyente: Verbeke et al. (2012) no encontraron mejoras significativas al aplicarlo en once conjuntos de datos de abandono de clientes. Burez y Van den Poel (2009) compararon métodos de muestreo con técnicas como los bosques aleatorios ponderados y recomiendan medir con el AUC, que no depende de un umbral de clasificación.

## 3. Algoritmos de aprendizaje automático

### 3.1 Antecedentes en problemas de negocio similares

La literatura de predicción de recompra, abandono de clientes y respuesta a cupones ha usado una gama amplia de algoritmos:

| Estudio | Problema | Algoritmos | Resultado principal |
|---|---|---|---|
| Liu et al., 2016 | Recompra tras una promoción | Construcción de variables y entrenamiento de modelos | Las variables fueron el factor decisivo |
| Zhang et al., 2022 | Recompra tras una promoción | Factorización, regresión logística, XGBoost, LightGBM y apilamiento | El apilamiento subió el AUC en 0.01161 |
| Martínez et al., 2020 | Compra del mes siguiente | Varios métodos de aprendizaje automático | Gradient boosting fue el mejor |
| Larivière y Van den Poel, 2005 | Siguiente compra y abandono | Bosque aleatorio contra regresiones | El bosque aleatorio ajustó mejor |
| Buckinx y Van den Poel, 2005 | Abandono parcial en un minorista | Regresión logística, red neuronal y bosque aleatorio | Recencia, frecuencia y monto fueron las mejores variables |
| Lemmens y Croux, 2006 | Abandono en telefonía | Bagging y boosting de árboles | Ambos mejoraron la predicción |
| Verbeke et al., 2012 | Abandono en telefonía | Amplia gama de clasificadores | Muchos clasificadores rindieron parecido |
| De Caigny et al., 2018 | Abandono | Árbol, regresión logística, bosque aleatorio y un modelo híbrido | El híbrido igualó al bosque aleatorio |
| Ahmad et al., 2019 | Abandono en telefonía | Árbol, bosque aleatorio, gradient boosting y XGBoost | XGBoost fue el mejor |
| Ren et al., 2021 | Uso de cupones electrónicos | Segmentación, bosque de aislamiento y regresión logística | Segmentar redujo el desbalance |
| Lessmann et al., 2015 | Riesgo crediticio | 41 clasificadores | El bosque aleatorio es el punto de comparación recomendado |

Tres patrones se repiten. La regresión logística aparece como referencia en casi todos los estudios. Los ensambles de árboles, por bagging o por boosting, obtienen los mejores resultados o empatan con ellos en la mayoría de las comparaciones con datos tabulares de clientes. Las redes neuronales son competitivas en algunos estudios: Lessmann et al. (2015) encontraron que superan a la regresión logística, pero en datos tabulares no superan de forma consistente a los ensambles de árboles (Shwartz-Ziv y Armon, 2022; Grinsztajn et al., 2022).

### 3.2 Regresión logística

La regresión logística modela la probabilidad de la clase positiva como una función logística de una combinación lineal de las variables (Cox, 1958). Sus coeficientes indican cómo cambia la probabilidad con cada variable, lo que la hace fácil de interpretar. Puede regularizarse para evitar el sobreajuste: la penalización L2 encoge todos los coeficientes y la L1 lleva algunos a cero. Supone que cada variable tiene un efecto lineal y aditivo sobre el logaritmo de la razón de probabilidades, por lo que necesita variables transformadas y escaladas.

### 3.3 Bosque aleatorio

El bosque aleatorio entrena muchos árboles de decisión, cada uno con una muestra con reemplazo de los datos y considerando en cada división un subconjunto aleatorio de variables. La predicción final promedia los árboles (Breiman, 2001). Captura relaciones no lineales e interacciones entre variables sin especificarlas, es poco sensible a los valores extremos y no necesita escalar las variables. Sus hiperparámetros principales son el número de árboles, la profundidad máxima, el tamaño mínimo de las hojas y la fracción de variables por división.

### 3.4 Gradient boosting

El gradient boosting construye árboles en secuencia, cada uno corrigiendo los errores de los anteriores en la dirección del gradiente de una función de pérdida (Friedman, 2001). Existen varias implementaciones:

- **XGBoost** agrega regularización a la función objetivo y técnicas para escalar a conjuntos de datos grandes y dispersos (Chen y Guestrin, 2016).
- **LightGBM** agrupa los valores de cada variable en histogramas, lo que acelera mucho el entrenamiento con una precisión similar (Ke et al., 2017).
- **HistGradientBoostingClassifier** de scikit-learn está inspirado en LightGBM y maneja de forma nativa los valores faltantes y las variables categóricas (Pedregosa et al., 2011).

Sus hiperparámetros principales son la tasa de aprendizaje, el número máximo de hojas por árbol, el tamaño mínimo de las hojas y la regularización.

### 3.5 Perceptrón multicapa

El perceptrón multicapa es una red neuronal de capas totalmente conectadas que se entrena con retropropagación del error (Rumelhart et al., 1986). Su arquitectura se define por el número de capas ocultas, el número de neuronas por capa y la regularización. Puede aprender relaciones no lineales, pero es sensible a la escala de las variables y a la elección de hiperparámetros, por lo que requiere escalamiento y detención temprana del entrenamiento. Con pocas capas ocultas no se considera aprendizaje profundo.

### 3.6 Evaluación de los algoritmos

**Curva ROC y AUC.** La curva ROC muestra la tasa de verdaderos positivos frente a la tasa de falsos positivos para todos los umbrales posibles. El área bajo esa curva, AUC, mide la capacidad del modelo para ordenar a los casos positivos por encima de los negativos: 0.5 equivale al azar y 1 a una separación perfecta (Fawcett, 2006). Como no depende de un umbral, es la medida adecuada con clases desbalanceadas (Burez y Van den Poel, 2009).

**Métricas con umbral.** Al fijar un umbral, cada cliente se clasifica como recurrente o no, y se construye la matriz de confusión con los verdaderos positivos, falsos positivos, falsos negativos y verdaderos negativos. De ella salen tres medidas:

- **Precisión:** proporción de clientes clasificados como recurrentes que realmente lo son.
- **Sensibilidad:** proporción de clientes recurrentes que el modelo detecta.
- **F1:** media armónica de precisión y sensibilidad.

**Validación temporal.** Un modelo de clientes se usa para predecir periodos futuros, así que debe evaluarse con datos posteriores a los de entrenamiento. Neslin et al. (2006) mostraron que los modelos de abandono conservan su desempeño cuando se aplican a datos recolectados tres meses después de los de entrenamiento, lo que respalda esta forma de evaluar.

**Ajuste de hiperparámetros.** Los hiperparámetros se eligen con un conjunto de validación separado del de prueba. Bergstra y Bengio (2012) mostraron que la búsqueda aleatoria encuentra configuraciones tan buenas o mejores que la búsqueda en rejilla con el mismo presupuesto de cómputo.
