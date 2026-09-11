<div align="center">

UNIVERSIDAD DEL VALLE DE GUATEMALA<br>
Data Science<br>
Facultad de Ingeniería<br>
Sección 40<br>
Erick Marroquín

<br>

**Investigación del tema**

<br>

Joel Jaquez - 23369<br>
Fernando Ruiz - 23065<br>
Kevin Villagran - 23584<br>
Nery Molina - 23218

<br>

GUATEMALA, 13 de septiembre 2026

</div>

---

# Investigación previa al análisis exploratorio

## Predicción de compradores recurrentes

Departamento de Ciencias de la Computación. CC3084 Data Science. Proyecto 2, Análisis Exploratorio.

---

## 1. Introducción

El presente documento reúne la investigación previa al análisis exploratorio del reto de predicción de compradores recurrentes. Su finalidad es doble. Por un lado, describir el fenómeno de negocio que da origen al problema, de manera que el análisis posterior tenga un marco de referencia y no se limite a describir columnas. Por el otro, dejar establecido de forma ordenada qué se debe buscar dentro de los datos, con qué tipo de gráfico y con qué pregunta en mente.

La fuente de datos seleccionada es la competencia Acquire Valued Shoppers Challenge, publicada en Kaggle en 2014 con registros reales de cadenas de supermercados de Estados Unidos (Kaggle, 2014).

Nota sobre la fuente. El reto que aparece en la guía del curso con el nombre "Predicción de compradores recurrentes. Cuestionar la línea base" corresponde a una competencia distinta, organizada por Alibaba con datos de la plataforma Tmall. El conjunto de Kaggle aquí utilizado plantea la misma pregunta de fondo, o sea distinguir al comprador de una sola vez del comprador que regresa, pero con otra fuente y otras variables. Esta sustitución queda documentada de forma explícita para que la revisión del trabajo se haga sobre la fuente correcta.

---

## 2. Antecedentes de la situación problemática

### 2.1 El valor económico del cliente que regresa

La investigación en administración de servicios ha mostrado que el cliente que permanece genera un valor muy superior al que sugiere una sola transacción. En un estudio ampliamente citado, Reichheld y Sasser (1990) documentaron que una reducción de apenas cinco por ciento en la tasa de clientes que abandonan una empresa se traduce en aumentos de utilidad que van de veinticinco a ochenta y cinco por ciento, según la industria analizada. La razón es que el cliente antiguo compra con mayor frecuencia, requiere menos gasto de captación y suele recomendar la marca.

De esta observación se desprende una consecuencia directa para la operación comercial. Un peso invertido en conservar a un cliente existente rinde más que el mismo peso invertido en atraer a uno nuevo, siempre que se pueda identificar a quién conviene conservar.

### 2.2 El alcance limitado de las promociones sobre la lealtad

Las promociones de precio cumplen bien la función de generar volumen inmediato. Su efecto sobre la lealtad de largo plazo es mucho más débil y en ocasiones contrario al deseado. La literatura de mercadeo ha señalado que el descuento repetido acostumbra al consumidor a comprar únicamente cuando el precio está reducido, baja su precio de referencia y traslada su preferencia del producto hacia la oferta misma. El resultado es un cliente sensible al precio que no vuelve cuando el precio regresa a su nivel normal.

En términos operativos esto produce una situación desfavorable. La campaña de cupones alcanza a una gran cantidad de personas, la mayoría de ellas responde una sola vez, y el costo del descuento, del envío y de la administración de la campaña se reparte entre un número reducido de clientes que en efecto se quedan.

### 2.3 La segmentación como respuesta al problema

La respuesta que la industria ha adoptado consiste en dejar de tratar a todos los clientes por igual. En lugar de repartir el mismo cupón a toda la base, se estima para cada persona la probabilidad de que su respuesta al incentivo se sostenga en el tiempo, y el incentivo se concentra en quienes muestran esa probabilidad alta. El problema pasa así del terreno del mercadeo masivo al terreno de la clasificación estadística.

Esta es exactamente la tarea que plantea el reto. Se cuenta con clientes que recibieron y utilizaron un cupón sobre un producto determinado, y se busca predecir cuáles de ellos volvieron a comprar ese producto después, ya sin descuento.

---

## 3. Marco conceptual

Los siguientes conceptos provienen del área de administración de la relación con el cliente y de la minería de datos aplicada al comercio. Sirven para orientar la construcción de variables y la lectura de los resultados.

### 3.1 Modelo RFM

El modelo RFM caracteriza a cada cliente con tres medidas tomadas de su historial de transacciones. La recencia mide cuánto tiempo ha pasado desde su última compra. La frecuencia mide cuántas compras realizó en un periodo. El valor monetario mide cuánto dinero gastó. Con esas tres medidas se ordenan y agrupan los clientes, y a cada grupo se le asigna una estrategia comercial distinta (Christy et al., 2021).

Su utilidad para este trabajo es doble. Primero, define un mínimo de variables que cualquier análisis de comportamiento de compra debería incluir. Segundo, el conjunto de datos disponible permite calcular las tres medidas completas, porque registra montos de dinero además de fechas y cantidades.

### 3.2 Predicción de abandono

La predicción de abandono, conocida como churn prediction, estima la probabilidad de que un cliente deje de comprar. Es el planteamiento inverso al de este reto y comparte con él tanto las variables utilizadas como los problemas metodológicos, en particular el desbalance entre clases y la elección de la medida de desempeño (Burez & Van den Poel, 2009).

### 3.3 Valor del cliente en el tiempo

El valor del cliente en el tiempo estima el total de utilidad que una persona aportará durante la relación comercial completa. Se emplea para decidir cuánto dinero es razonable invertir en retener a cada cliente. En este reto la variable repeattrips, que cuenta el número de recompras, funciona como una versión reducida de esa idea.

### 3.4 Construcción de variables a partir de bitácoras de comportamiento

Los datos de este tipo de problema no llegan en forma de tabla lista para modelar. Llegan como una bitácora de transacciones individuales que hay que resumir por cliente. Un estudio sobre predicción de compradores recurrentes en comercio electrónico encontró que ninguna variable tomada por separado distingue bien al comprador que regresa, y que el desempeño aceptable se logra al combinar muchos perfiles distintos, del cliente, del comercio, de la marca, de la categoría y de las interacciones entre ellos (Liu et al., 2016).

Este hallazgo tiene una implicación práctica para el análisis exploratorio. Encontrar correlaciones bajas entre las variables individuales y la variable objetivo es el resultado esperado en este tipo de problema y no una señal de que el análisis esté mal hecho.

---

## 4. Descripción de la fuente de datos

### 4.1 Origen y definición de la tarea

Los datos corresponden a una campaña de cupones de descuento aplicada en cadenas de supermercados. A cada cliente del conjunto se le entregó un cupón sobre un producto específico y el cliente lo utilizó. La tarea consiste en predecir la probabilidad de que ese cliente volviera a comprar el mismo producto después de la promoción. La competencia se evalúa mediante el AUC, que corresponde al área bajo la curva ROC (Kaggle, 2014).

Cada registro del conjunto de entrenamiento representa un par formado por un cliente y una oferta, y la respuesta conocida indica si hubo recompra.

### 4.2 Estructura de los archivos

El conjunto está formado por cinco archivos. Cuatro tienen los datos y el quinto, sampleSubmission.csv, solo muestra el formato en que se envían las predicciones a la competencia.

offers.csv contiene el catálogo de cupones. Tiene 37 filas y 6 columnas. La columna offer identifica el cupón, category indica la categoría del producto, quantity indica cuántas unidades se deben comprar para que el descuento aplique, company identifica a la empresa fabricante, offervalue registra el monto del descuento en dólares y brand identifica la marca. Las 37 ofertas cubren 20 categorías, 18 empresas y 19 marcas. Los descuentos van de 0.75 a 5 dólares. La cantidad exigida es casi siempre una unidad y solo una oferta requiere dos.

trainHistory.csv contiene los clientes con respuesta conocida. Tiene 160,057 filas y 7 columnas. La columna id identifica al cliente, chain indica la cadena de tiendas, offer indica el cupón recibido, market indica la región geográfica, repeattrips cuenta las recompras posteriores del producto, repeater es la variable objetivo con valor "t" para quien volvió a comprar y "f" para quien no lo hizo, y offerdate registra la fecha de entrega del cupón. Cada cliente aparece una sola vez. El archivo abarca 130 cadenas, 34 regiones y 24 de las 37 ofertas, con fechas de cupón entre el 1 de marzo y el 30 de abril de 2013.

testHistory.csv contiene los clientes cuya respuesta se debe predecir. Tiene 151,484 filas y las mismas columnas anteriores sin repeattrips ni repeater. Abarca 29 ofertas, con fechas de cupón entre el 1 de mayo y el 31 de julio de 2013.

transactions.csv contiene el historial de compras previo de todos los clientes. Tiene cerca de 350 millones de filas y 11 columnas, y ocupa alrededor de 22 GB una vez descomprimido. Cubre por lo menos un año de compras anteriores al cupón, con registros que inician en marzo de 2012, y acumula un gasto total de unos 1,570 millones de dólares según el cálculo hecho sobre el archivo completo. Las columnas son id para el cliente, chain para la cadena, dept para el departamento de la tienda, category para la categoría del producto, company para la empresa fabricante, brand para la marca, date para la fecha de compra, productsize para el tamaño del producto, productmeasure para la unidad en que se expresa ese tamaño, purchasequantity para las unidades adquiridas y purchaseamount para el monto pagado.

### 4.3 Consideraciones de procesamiento

El volumen del archivo de transacciones impide cargarlo completo en memoria dentro de un entorno como Google Colab. Existe un procedimiento de reducción documentado por los participantes de la competencia que consiste en conservar únicamente las transacciones cuya categoría o cuya empresa aparezca en offers.csv, ya que el resto de los registros no guarda relación con los productos ofrecidos (Triskelion, 2014). El código de ese procedimiento sigue disponible en GitHub (MLWave, 2014). Aplicada sobre el archivo completo, la reducción lo lleva de cerca de 350 millones a unos 27.8 millones de filas, menos del 8% del original, y de 22 GB a menos de 2 GB. La lectura se realiza por bloques, con el parámetro chunksize de pandas.

Esta reducción constituye en sí misma una operación de limpieza y como tal debe quedar descrita y justificada en el informe, junto con la conversión de tipos de dato, ya que las fechas llegan como texto y los identificadores como enteros de tamaño mayor al necesario.

---

## 5. Orientación para el análisis exploratorio

Esta sección plantea, para cada componente del análisis, la pregunta que se busca responder, la técnica adecuada y el resultado que cabe esperar según lo revisado en la literatura y en los reportes de la competencia.

### 5.1 Inventario y tipificación de las variables

La pregunta inicial es con cuántas observaciones y cuántas variables se cuenta en cada archivo, y de qué tipo es cada variable.

El punto de cuidado está en que casi todas las columnas se almacenan como números enteros sin que eso las convierta en variables cuantitativas. Los campos id, chain, offer, market, category, company, brand y dept son identificadores, o sea variables categóricas representadas con números. Calcularles media o desviación estándar produce resultados sin significado.

Las variables cuantitativas presentes en los datos crudos son offervalue, quantity, repeattrips, productsize, purchasequantity y purchaseamount. El resto de las variables cuantitativas del estudio surgirá al resumir la bitácora de transacciones por cliente.

### 5.2 Variable objetivo

La pregunta es cómo se distribuye la variable repeater y qué tan desbalanceada está.

El resultado esperado es una proporción cercana al 27 por ciento de clientes recurrentes, equivalente a unos 43,438 casos de los 160,057. Se trata de un desbalance moderado que permite trabajar sin recurrir de inmediato a técnicas de remuestreo.

Conviene analizar también repeattrips, que es la versión de conteo de la misma información. Entre los clientes que volvieron, la mediana se ubica en una sola recompra y la media alrededor de 2.4, lo que indica una distribución muy concentrada en valores bajos con una cola derecha larga. En esa cola aparecen registros con cientos o miles de recompras del mismo producto, magnitud que no corresponde al consumo de una persona y que debe examinarse como posible anomalía de registro.

### 5.3 Variables categóricas

Las preguntas son cómo se reparten los clientes entre ofertas, cadenas y regiones, y si la tasa de recompra cambia entre esos grupos. La técnica adecuada son las tablas de frecuencia, las tablas de proporciones y los gráficos de barra, incluyendo barras apiladas por valor de repeater.

Puntos concretos a revisar.

- Distribución de clientes por oferta. El reparto es muy desigual, con ofertas que concentran decenas de miles de clientes y otras con apenas un par de cientos. Esto afecta la confiabilidad de cualquier tasa calculada dentro de las ofertas pequeñas.
- Distribución por cadena y por región, donde también se espera concentración.
- Tasa de recompra dentro de cada oferta, de cada categoría y de cada marca. Este cruce es de los más informativos del análisis, ya que existen categorías de consumo rápido en las que volver a comprar es lo habitual y otras de consumo esporádico en las que no.
- Traslape de ofertas entre entrenamiento y prueba. Solo 16 de las 37 ofertas aparecen en ambos archivos. Este dato es relevante porque limita el uso de la identidad de la oferta como variable predictora y obliga a describir las ofertas por sus atributos, o sea categoría, marca, empresa y valor del descuento.
- Distribución de productmeasure en las transacciones, que permite separar los productos vendidos por peso de los vendidos por unidad y evita comparar tamaños que no son comparables.

### 5.4 Variables cuantitativas y variables derivadas

La pregunta es qué medidas resumen el comportamiento de compra de cada cliente antes de recibir el cupón. La técnica es el resumen estadístico descriptivo acompañado de histogramas y diagramas de caja y bigotes.

Sobre los datos crudos corresponde describir offervalue, purchaseamount, purchasequantity y productsize. Sobre la bitácora agrupada por cliente conviene construir y describir las siguientes medidas.

- Gasto total del cliente en todo el historial disponible.
- Número de visitas, contando fechas distintas con al menos una transacción.
- Gasto promedio por visita.
- Número de compras previas de la categoría del cupón, de la marca del cupón y de la empresa del cupón. Estas medidas son las variables que uno de los participantes de la competencia compartió como punto de partida para los demás (Triskelion, 2014).
- Monto gastado previamente en esa categoría, esa marca y esa empresa.
- Las mismas medidas restringidas a los últimos 30, 60, 90 y 180 días antes de la fecha del cupón, con el fin de capturar la recencia.
- Días transcurridos entre la primera compra registrada del cliente y la fecha del cupón, como medida de antigüedad.
- Días transcurridos entre la última compra de la categoría del cupón y la fecha del cupón.
- Diversidad de consumo, o sea el número de categorías, marcas y empresas distintas que el cliente ha comprado.

El resultado esperado es que casi todas estas distribuciones aparezcan con sesgo positivo marcado, con una minoría de clientes que concentra un gasto muy alto. De ahí que resulte útil aplicar transformaciones como el logaritmo para poder representarlas gráficamente.

### 5.5 Cruces entre variables

La pregunta es cuáles de las medidas anteriores se asocian con la recompra. La técnica consiste en comparar distribuciones entre el grupo que volvió y el que no volvió, mediante diagramas de caja separados por clase, y en calcular la tasa de recompra por tramos de cada variable cuantitativa.

Cruces que conviene incluir.

- Tasa de recompra según si el cliente había comprado antes la marca del cupón. La hipótesis es que el cliente que ya conocía la marca vuelve con mayor frecuencia, y el análisis debe verificar si la diferencia existe y cuál es su magnitud.
- El mismo cruce para la categoría y para la empresa del cupón.
- Tasa de recompra frente al valor del descuento, para examinar si los descuentos mayores atraen clientes más sensibles al precio y producen menos recompra.
- Tasa de recompra frente al gasto histórico total y frente al número de visitas.
- Tasa de recompra por cadena y por región, para detectar efectos geográficos o de formato de tienda.

### 5.6 Valores faltantes

La pregunta es dónde hay ausencia de dato y qué significa esa ausencia.

En este conjunto los faltantes directos son limitados y se concentran en campos como productmeasure. El caso más importante es de otra naturaleza. Al calcular las variables derivadas aparecen clientes sin ninguna compra previa de la categoría, la marca o la empresa del cupón, lo que genera celdas vacías que en realidad representan un valor de cero.

El tratamiento adecuado consiste en asignar cero y agregar además una variable indicadora que registre que el cliente nunca compró ese producto, esa marca o esa categoría antes, ya que la ausencia total de historial es información con capacidad predictiva propia (Triskelion, 2014). Esta decisión debe quedar explicada en el informe, junto con el criterio de por qué no se utiliza imputación por media.

### 5.7 Valores atípicos

La pregunta es cuáles valores extremos corresponden a variación legítima y cuáles a problemas de registro. El conjunto presenta tres casos de naturaleza distinta que exigen tratamientos distintos.

- Clientes con gasto muy alto. Corresponden a la cola natural de la distribución de consumo en el comercio minorista y no a errores. Eliminarlos introduciría un sesgo. El tratamiento recomendado es documentarlos y usar escalas transformadas para su representación gráfica.
- Montos y cantidades negativas en purchaseamount y purchasequantity. La descripción de la competencia los presenta como devoluciones, pero el análisis de los datos muestra que esa explicación solo cubre una parte mínima. De las 8.1 millones de líneas con monto negativo, solo 273,848 tienen además cantidad negativa y corresponden a devoluciones. Las otras 7.8 millones tienen cantidad positiva o cero, se concentran en unas pocas categorías que no aparecen en ninguna oferta, en particular la 9609 con el 64.6% de los casos, y tienen montos redondos sin tamaño de producto, lo que indica cupones o descuentos registrados como una línea aparte. Ambos tipos de línea son información aprovechable. Las devoluciones permiten saber si el cliente había devuelto antes el producto y los descuentos muestran qué tan sensible es a las promociones. Ninguno de los dos debe contarse como una compra del producto.
- Identificadores con volúmenes de transacción imposibles para una persona, del orden de miles de compras por mes. Una explicación posible es que se trate de tarjetas compartidas, tarjetas propias del establecimiento o cuentas donde el sistema registra a los clientes que no presentaron tarjeta. En los datos aparecen 161 clientes de train con más de 750 líneas de compra por mes, un volumen que no corresponde al de un hogar. Este tipo de registro conviene identificarlo, cuantificarlo y decidir de forma explícita si se separa del análisis.

### 5.8 Correlaciones

La pregunta es cómo se relacionan las variables entre sí y con la variable objetivo. La técnica es la matriz de correlación acompañada de un mapa de calor.

Se esperan dos resultados. El primero es redundancia alta entre variables de la misma familia, por ejemplo entre gasto total y número de visitas, o entre las ventanas de 30, 60 y 90 días, que miden lo mismo con distinto corte de tiempo. El segundo es correlación baja entre cualquier variable individual y la variable objetivo, resultado consistente con lo reportado en la literatura sobre este tipo de predicción (Liu et al., 2016).

### 5.9 Componente temporal

La pregunta es cómo se distribuyen las transacciones y los cupones en el tiempo.

Corresponde graficar el número de transacciones y el monto agregado por mes durante todo el periodo cubierto, con dos fines. Verificar que la ventana de historial disponible sea comparable entre clientes, y detectar estacionalidad que pueda confundirse con efecto del cupón.

Un hallazgo estructural que debe quedar registrado es que los cupones de entrenamiento se entregaron entre marzo y abril de 2013 y los de prueba entre mayo y julio de 2013, sin traslape de fechas ni de clientes entre ambos conjuntos.

---

## 6. Consideraciones metodológicas que se desprenden del análisis

Aunque el alcance de esta entrega es exploratorio, el análisis permite anticipar cuatro condiciones que determinan los pasos siguientes y que corresponde señalar en las conclusiones.

Volumen de datos. El archivo de transacciones exige una estrategia de reducción y de lectura por bloques desde el inicio del trabajo, además del control explícito de los tipos de dato.

Separación temporal. La ausencia de traslape de fechas entre entrenamiento y prueba obliga a que cualquier validación posterior respete el orden cronológico. Una partición aleatoria que mezcle los periodos produciría una estimación de desempeño más favorable que la real.

Desbalance de clases y elección de la medida. Con cerca de 27 por ciento de casos positivos, un modelo que prediga ausencia de recompra para todos los clientes alcanzaría un 73 por ciento de exactitud sin aportar información útil. La exactitud queda descartada como medida de desempeño. La medida apropiada es el AUC, que no depende de un punto de corte y no privilegia a la clase mayoritaria (Burez & Van den Poel, 2009). Entre las técnicas disponibles para el desbalance figuran el submuestreo de la clase mayoritaria, el sobremuestreo sintético de la clase minoritaria con métodos como SMOTE (Chawla et al., 2002) y la asignación de pesos distintos a cada clase durante el entrenamiento.

Dificultad esperada del problema. En la tabla final de la competencia el primer lugar obtuvo un AUC cercano a 0.63 y los diez primeros lugares quedaron entre 0.61 y 0.63 (Kaggle, 2014). El enfoque que Triskelion compartió como punto de partida rondaba 0.59 (Triskelion, 2014). Establecer esta referencia desde el inicio evita interpretar como fracaso un resultado que en realidad es competitivo para el problema.

---

## 7. Aporte de esta investigación a los apartados del informe

Situación problemática. Se sustenta en la sección 2. Los elementos centrales son el valor económico del cliente que permanece, el alcance limitado de las promociones de precio sobre la lealtad y el costo de repartir incentivos sin criterio de selección.

Problema científico. Puede formularse como la dificultad de distinguir, a partir del historial de compras previo, cuáles clientes que respondieron a un cupón presentan una probabilidad alta de volver a comprar el producto sin descuento.

Objetivos. El objetivo general puede orientarse a caracterizar el comportamiento de compra de los clientes que respondieron al cupón mediante un análisis exploratorio que identifique los factores asociados a la recompra. Los objetivos específicos, para que resulten medibles y alcanzables, pueden formularse sobre productos verificables, por ejemplo reducir y depurar el archivo de transacciones dejando constancia del criterio aplicado, cuantificar la distribución de la variable objetivo junto con sus valores extremos, y construir y evaluar un conjunto de variables derivadas del historial de compras según su asociación con la recompra.

---

## 8. Referencias

Burez, J., & Van den Poel, D. (2009). Handling class imbalance in customer churn prediction. *Expert Systems with Applications, 36*(3), 4626–4636. https://www.sciencedirect.com/science/article/abs/pii/S0957417408002121

Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer, W. P. (2002). SMOTE. Synthetic minority over-sampling technique. *Journal of Artificial Intelligence Research, 16*, 321–357. https://doi.org/10.1613/jair.953

Christy, A. J., Umamakeswari, A., Priyatharsini, L., & Neyaa, A. (2021). RFM ranking – An effective approach to customer segmentation. *Journal of King Saud University – Computer and Information Sciences, 33*(10), 1251–1257. https://doi.org/10.1016/j.jksuci.2018.09.004

Kaggle. (2014). *Acquire valued shoppers challenge. Predict which shoppers will become repeat buyers* [Conjunto de datos y descripción de la competencia]. https://www.kaggle.com/competitions/acquire-valued-shoppers-challenge/data

Liu, G., Nguyen, T. T., Zhao, G., Zha, W., Yang, J., Cao, J., Wu, M., Zhao, P., & Chen, W. (2016). Repeat buyer prediction for e-commerce. En *Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining* (pp. 155–164). ACM. https://doi.org/10.1145/2939672.2939674

MLWave. (2014). *Code for the Kaggle acquire valued shoppers challenge* [Código fuente]. GitHub. https://github.com/MLWave/kaggle_acquire-valued-shoppers-challenge

Reichheld, F. F., & Sasser, W. E. (1990). Zero defections. Quality comes to services. *Harvard Business Review, 68*(5), 105–111. https://hbr.org/1990/09/zero-defections-quality-comes-to-services

Triskelion. (2014, 15 de abril). *Predicting repeat buyers using purchase history* [Entrada de blog]. MLWave. https://mlwave.com/predicting-repeat-buyers-vowpal-wabbit/

---

