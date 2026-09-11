# 2. Preprocesamiento de las transacciones

Esta sección completa la descripción de los datos con el archivo `transactions.csv`, que contiene el historial de compras de los clientes, y documenta las operaciones de limpieza y preprocesamiento aplicadas antes del análisis exploratorio. El código está en `src/preprocesamiento.py` y el procedimiento completo, con todas las tablas, en `notebooks/02_preprocesamiento_transacciones.ipynb`.

## 2.1 Descripción del archivo de transacciones

`transactions.csv` tiene 349,655,789 observaciones y 11 variables. Registra cada línea de compra de los 311,541 clientes de train y test entre marzo de 2012 y julio de 2013, no solo las compras relacionadas con la oferta. Pesa 2.9 GB comprimido y cerca de 22 GB descomprimido.

| Variable | Tipo | Descripción |
|---|---|---|
| `id` | categórica, identificador | cliente |
| `chain` | categórica, identificador | cadena de tiendas |
| `dept` | categórica, identificador | departamento de la tienda |
| `category` | categórica, identificador | categoría del producto |
| `company` | categórica, identificador | empresa fabricante |
| `brand` | categórica, identificador | marca |
| `date` | fecha | fecha de la compra |
| `productsize` | cuantitativa continua | tamaño del producto |
| `productmeasure` | categórica | unidad en que se expresa el tamaño |
| `purchasequantity` | cuantitativa discreta | unidades compradas |
| `purchaseamount` | cuantitativa continua | monto pagado en dólares |

Solo tres variables son cuantitativas. Las seis columnas de identificadores se almacenan como números enteros, pero son categóricas y vienen anonimizadas, así que calcularles una media o una desviación no tiene significado.

## 2.2 Estrategia de lectura

El archivo no cabe en la memoria de una computadora personal ni de Google Colab, así que se aplicaron dos medidas. La primera fue declarar tipos de dato más pequeños al leerlo, con enteros de 16 y 32 bits para los identificadores que caben en ellos, `float32` para el tamaño y la fecha como tipo fecha en lugar de texto. Con esto un millón de filas pasa de 100.1 MB a 60.1 MB, un 40% menos. Aun así el archivo completo ocuparía unos 21 GB, por lo que la segunda medida fue leerlo por bloques de 2 millones de filas directamente desde el archivo comprimido.

Todo el procesamiento se hace en una sola pasada de 175 bloques que tarda unos cuatro minutos y medio. Para resumir cada cliente completo se aprovechó que las filas de un mismo cliente vienen juntas en el archivo, así que las filas del último cliente de cada bloque se guardan aparte y se agregan al bloque siguiente. La pasada verifica ese supuesto y ningún cliente apareció en dos bloques separados, así que los resúmenes por cliente son exactos.

## 2.3 Validación de las transacciones

`transactions` no tiene la columna `offer`. Cada transacción se relaciona con su cliente por el par `id` y `chain`, y con el producto ofertado por la combinación de `category`, `company` y `brand`. Se conservó una transacción solo si pertenece a un cliente de train o test, si ocurrió en la misma cadena donde el cliente recibió la oferta y si su fecha es anterior a la fecha de la oferta. Esta última condición garantiza que ninguna variable use información posterior al cupón.

| Etapa | Filas | Porcentaje |
|---|---:|---:|
| Filas leídas | 349,655,789 | 100.00% |
| Válidas, con cliente conocido, misma cadena y fecha anterior a la oferta | 349,542,426 | 99.97% |
| Reducidas a la categoría o compañía de alguna oferta | 27,756,831 | 7.94% |

Todas las transacciones son de clientes de train o test, y ninguna tiene fecha igual o posterior a la oferta, porque el historial ya viene cortado antes del cupón. Las únicas filas descartadas son 113,363, el 0.03%, que corresponden al mismo cliente en otra cadena. Los 311,541 clientes conservan historial.

Además se aplicó la reducción que compartió uno de los participantes de la competencia, que consiste en conservar solo las transacciones cuya categoría o compañía aparece en `offers.csv` (Triskelion, 2014). El resultado es de 27.8 millones de filas, menos del 8% del archivo. Estas filas se guardaron en `transacciones_reducidas.parquet` para poder construir variables adicionales en etapas posteriores. Las variables generales de cada cliente sí usan todas las filas válidas.

## 2.4 Calidad de los datos y decisiones de limpieza

| Situación | Líneas | Porcentaje | Decisión |
|---|---:|---:|---|
| Cantidad negativa, devolución | 273,848 | 0.08% | se conserva, entra al gasto neto y se cuenta aparte |
| Monto negativo con cantidad positiva o cero, descuento | 7,847,034 | 2.24% | se conserva, entra al gasto neto y se cuenta aparte |
| Monto igual a cero | 5,422,153 | 1.55% | se conserva y no cuenta como compra |
| Cantidad igual a cero | 518,736 | 0.15% | se conserva y no cuenta como compra |
| Tamaño de producto igual a cero | 11,507,837 | 3.29% | es un faltante disfrazado y no se usa en ninguna variable |
| Unidad de medida vacía | 11,500,472 | 3.29% | son prácticamente las mismas líneas del caso anterior |

Montos negativos. La guía del reto indica que los valores negativos son devoluciones, pero los datos muestran otra cosa. De los montos negativos, solo 273,848 líneas tienen además cantidad negativa. Las otras 7.8 millones tienen cantidad positiva o cero y se concentran en muy pocas categorías. El 64.6% está en la categoría 9609 del departamento 96, con montos redondos y sin tamaño de producto, el 14.5% en la categoría 0 y el 7.8% en la 9781. Ninguna de ellas es una categoría ofertada. Por su forma, estas líneas corresponden a cupones o descuentos registrados como una línea aparte. En conjunto equivalen a menos del 2% del gasto positivo.

Se decidió no eliminar ninguna de estas filas. El gasto total de cada cliente se calculó como la suma neta de los montos, que representa lo que el cliente pagó en realidad, y las devoluciones y los descuentos se contaron en variables propias. Para contar compras, en cambio, solo se consideraron las líneas con cantidad y monto positivos, de modo que un descuento no se confunda con una compra del producto.

Registros imposibles. El monto mínimo de una línea es de -8,593,791 dólares, la cantidad mínima de -32,255 unidades y la máxima de 54,800. Son errores de registro aislados, ya que en una muestra aleatoria del 0.1% del archivo solo 10 líneas superan los 1,000 dólares en valor absoluto. Estos registros explican que la desviación estándar del monto sea 879 dólares cuando la media es 4.49. Se conservaron, porque corregirlos obligaría a inventar un valor, y se tratan como atípicos en el análisis, donde se prefieren la mediana y los percentiles.

Tamaño y unidad de medida. `productsize` no tiene valores nulos, pero vale cero en el 3.3% de las líneas, que son las mismas que no tienen unidad de medida. Ese cero es en realidad un dato faltante. Además, el 81.5% de las líneas está en onzas y el 12.0% en conteo de unidades, por lo que los tamaños solo son comparables dentro de una misma unidad. Por ambas razones el tamaño no se usó para construir variables y no fue necesario imputarlo.

## 2.5 Variables construidas por cliente

A partir de las transacciones válidas se construyó una fila por cliente con 37 variables de comportamiento, organizadas según el modelo RFM de recencia, frecuencia y monto, y según las variables que uno de los participantes de la competencia compartió como punto de partida (Triskelion, 2014). El diccionario completo está en `codebook.md`.

| Grupo | Variables | Qué mide |
|---|---|---|
| Frecuencia | `n_lineas`, `n_compras`, `n_visitas` | cuántas líneas, compras reales y días distintos de compra tiene el cliente |
| Monto | `gasto_total`, `ticket_promedio` | gasto neto y gasto promedio por visita |
| Recencia y antigüedad | `dias_ultima_compra`, `antiguedad_dias` | días desde la última y desde la primera transacción hasta la oferta |
| Diversidad | `n_categorias_distintas`, `n_marcas_distintas`, `n_companias_distintas` | amplitud del consumo |
| Devoluciones y descuentos | `n_devoluciones`, `n_descuentos` | líneas de cada tipo |
| Historial con lo ofertado | `n_compras_*` y `gasto_*` para categoría, compañía, marca y producto exacto | compras previas de lo que se ofertó |
| Ventanas de tiempo | `n_compras_*_30d` a `n_compras_*_180d` | las mismas compras en los 30, 60, 90 y 180 días previos a la oferta |
| Recencia del producto | `dias_ultima_compra_categoria` | días desde la última compra de la categoría ofertada |
| Banderas | `nunca_compro_*` | 1 si no hay compras previas de la categoría, compañía, marca o producto |

Las variables se verificaron recalculando a mano 11 de ellas para cuatro clientes a partir de las transacciones crudas, y todas coincidieron.

## 2.6 Valores faltantes

Los archivos pequeños del reto no tienen valores faltantes. En las variables construidas, un cliente que nunca compró la categoría, la compañía o la marca ofertada tiene en realidad un conteo de cero, por lo que los conteos y montos se completaron con cero. Esta situación es frecuente. El 45.4% de los clientes de train nunca compró la categoría ofertada, el 45.9% la compañía, el 60.6% la marca y el 84.3% el producto exacto.

La única variable con vacíos es `dias_ultima_compra_categoria`, con 72,639 clientes, el 45.38%, que son exactamente los que nunca compraron la categoría. Aquí no se imputó, porque la media inventaría una compra y el cero significaría una compra el día anterior. La ausencia de historial queda registrada en la bandera `nunca_compro_categoria`, que tiene valor predictivo propio, como se muestra en la sección 3.6. Para el modelado se podrá reemplazar el vacío por un valor mayor que el historial máximo o usar un modelo que acepte vacíos.

## 2.7 Resultado

El preprocesamiento produce `data/processed/dataset_modelo.csv`, con 160,057 clientes de train y 49 columnas, que son las 12 de identificación, oferta y variable objetivo y las 37 de comportamiento. La proporción de `repeater` se mantiene en 27.14%, igual que en los datos originales. Con las mismas reglas se generó `dataset_modelo_test.csv` para los 151,484 clientes de test, sin las columnas objetivo, y una muestra aleatoria de 349,163 transacciones crudas para el análisis exploratorio.
