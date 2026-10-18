# Resultados: datos originales y limpieza

## Características del conjunto original

Los datos corresponden a la competencia *Acquire Valued Shoppers Challenge*. La información combina las ofertas recibidas por los clientes, las características de los cupones y el historial de compras previo. Los identificadores de cadena, mercado, departamento, categoría, compañía y marca están anonimizados: sus códigos numéricos representan categorías y no cantidades.

| Archivo | Filas | Columnas | Contenido |
|---|---:|---:|---|
| `trainHistory.csv` | 160,057 | 7 | Clientes y ofertas con respuesta conocida |
| `testHistory.csv` | 151,484 | 5 | Clientes y ofertas sin las columnas de respuesta |
| `offers.csv` | 37 | 6 | Catálogo de ofertas, producto, cantidad requerida y valor del descuento |
| `transactions.csv` | 349,655,789 | 11 | Líneas de compra de los 311,541 clientes, entre marzo de 2012 y julio de 2013 |

El quinto archivo, `sampleSubmission.csv`, es una plantilla de entrega con las columnas `id` y `repeatProbability`; no aporta características de los clientes. En las transacciones, las variables cuantitativas son el tamaño del producto, las unidades compradas y el monto pagado; las demás describen identificadores, fecha y unidad de medida. El archivo ocupa cerca de 22 GB descomprimido.

La respuesta es `repeater`, verdadera cuando `repeattrips > 0`. Entre los clientes con respuesta conocida, 43,438 recompraron (27.14%) y 116,619 no lo hicieron (72.86%). La codificación original de texto `t`/`f` se convirtió a un valor booleano. Los archivos de historial y ofertas no presentan valores faltantes.

## Limpieza y variables resultantes

Las 349,655,789 líneas se leyeron desde el archivo comprimido en 175 bloques de hasta dos millones de filas. Los tipos de datos reducidos disminuyeron el consumo de memoria por millón de filas de 100.1 a 60.1 MB. Se preservó el historial completo de cada cliente cuando sus registros quedaban repartidos entre dos bloques.

La validación exigió un cliente conocido, la misma cadena de la oferta y una fecha anterior al cupón. Se descartaron 113,363 líneas de otras cadenas (0.03% del total) y quedaron 349,542,426 transacciones válidas, con historial para los 311,541 clientes. No se encontraron transacciones de clientes desconocidos ni fechas iguales o posteriores a la oferta. Además, se guardó un subconjunto de 27,756,831 líneas cuya categoría o compañía aparece en el catálogo de ofertas; los indicadores generales del cliente se calcularon con todas las líneas válidas.

| Situación observada | Líneas | Tratamiento aplicado |
|---|---:|---|
| Cantidad negativa | 273,848 | Se conservaron como devoluciones, incluidas en el gasto neto y contadas aparte |
| Monto negativo con cantidad positiva o cero | 7,847,034 | Se conservaron como posibles descuentos, incluidos en el gasto neto y contados aparte |
| Monto igual a cero | 5,422,153 | Se conservaron, sin contarlos como compras |
| Cantidad igual a cero | 518,736 | Se conservaron, sin contarlos como compras |
| Tamaño del producto igual a cero | 11,507,837 | Se interpretó como faltante y se excluyó el tamaño de las variables construidas |
| Unidad de medida vacía | 11,500,472 | Se documentó junto con la ausencia de tamaño; no requirió imputación |

Estas situaciones pueden coincidir en una misma línea, por lo que sus conteos no se suman. La concentración de montos negativos con cantidades no negativas en pocas categorías sugiere descuentos registrados por separado. Por eso el gasto se calculó como suma neta, mientras que una compra requiere simultáneamente cantidad y monto positivos. Los valores extremos se conservaron y quedaron documentados para su análisis exploratorio.

El resultado fue una fila por cliente con 37 variables de comportamiento: frecuencia de compras y visitas, gasto neto y ticket promedio, recencia y antigüedad, diversidad de consumo, devoluciones, descuentos e historial de compra de la categoría, compañía, marca y producto ofertados. También se incluyeron conteos en ventanas de 30, 60, 90 y 180 días y banderas de ausencia de compras previas.

Los conteos y montos sin compras previas se completaron con cero. En cambio, `dias_ultima_compra_categoria` conservó 72,639 faltantes (45.38% de los clientes con respuesta conocida), acompañados de `nunca_compro_categoria`, porque no existe una compra desde la cual calcular esa recencia. Su tratamiento específico para cada algoritmo se presenta en la sección de ajuste.

El conjunto resultante, `dataset_modelo.csv`, contiene 160,057 filas y 49 columnas: 12 de identificación, oferta y respuesta, más las 37 de comportamiento. La recompra se mantiene en 27.14%. También se generó `dataset_modelo_test.csv` para 151,484 clientes, sin las dos columnas de respuesta. La comprobación de 11 variables para cuatro clientes contra las transacciones originales produjo coincidencias en todos los casos.

Fuentes: [carga y descripción inicial](../secciones/01_carga_y_eda_inicial.md), [preprocesamiento de transacciones](../secciones/02_preprocesamiento_transacciones.md) y tablas de respaldo `01_*` y `02_*` en `results/tables/`. El detalle de las variables está en el [diccionario de datos](../../codebook.md).
