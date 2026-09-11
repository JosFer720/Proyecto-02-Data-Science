# Codebook — Acquire Valued Shoppers Challenge

Diccionario de datos de los 5 archivos crudos del reto. Todas las columnas categóricas (`category`, `brand`, `company`, `chain`, `market`, `dept`) vienen anonimizadas como IDs sin significado semántico — el EDA se reporta por frecuencia y por ID, no se puede interpretar qué producto es cada categoría.

## `trainHistory.csv`

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | entero | identificador del cliente |
| `chain` | entero | identificador de la cadena de tiendas donde compró |
| `offer` | entero | identificador de la oferta/cupón recibido |
| `market` | entero | identificador del mercado geográfico |
| `repeattrips` | entero | número de veces que el cliente volvió a comprar el producto ofertado |
| `repeater` | booleano | variable objetivo: `True` si `repeattrips > 0` |
| `offerdate` | fecha | fecha en la que se emitió la oferta |

## `testHistory.csv`

Mismas columnas que `trainHistory.csv` excepto `repeattrips` y `repeater` (es lo que hay que predecir). Contiene ofertas emitidas en o después de 2013-05-01.

## `offers.csv`

| Columna | Tipo | Descripción |
|---|---|---|
| `offer` | entero | identificador de la oferta (clave para unir con train/testHistory) |
| `category` | entero | categoría del producto ofertado |
| `quantity` | entero | unidades que hay que comprar para activar el descuento |
| `company` | entero | compañía fabricante del producto |
| `offervalue` | decimal | valor del descuento en dólares |
| `brand` | entero | marca del producto |

## `transactions.csv`

Historial completo de compras de cada cliente durante al menos un año (no solo lo relacionado a la oferta recibida).

| Columna | Tipo | Descripción |
|---|---|---|
| `id` | entero | identificador del cliente |
| `chain` | entero | cadena de tiendas |
| `dept` | entero | departamento del producto comprado |
| `category` | entero | categoría del producto comprado |
| `company` | entero | compañía fabricante |
| `brand` | entero | marca |
| `date` | fecha | fecha de la transacción |
| `productsize` | decimal | tamaño del producto |
| `productmeasure` | texto | unidad de medida del tamaño (ej. OZ, LB, CT) |
| `purchasequantity` | entero | unidades compradas (negativo = devolución) |
| `purchaseamount` | decimal | monto pagado (negativo = devolución) |

## `sampleSubmission.csv`

Formato exacto de envío esperado por Kaggle: `id`, `repeatProbability`.

## `data/processed/dataset_modelo.csv`

Generado por `notebooks/02_preprocesamiento_transacciones.ipynb` con `src/preprocesamiento.py`. Una fila por cliente de train (160,057) con las columnas de `trainHistory` y de su oferta en `offers`, más las variables de comportamiento. `dataset_modelo_test.csv` tiene las mismas columnas sin `repeattrips` ni `repeater`. Se carga con `preprocesamiento.cargar_dataset_modelo()`.

Todas las variables de historial usan solo transacciones de la misma cadena y con fecha anterior a `offerdate`. Una "compra" es una línea con `purchasequantity > 0` y `purchaseamount > 0`; las devoluciones y las líneas de descuento se cuentan aparte pero sí entran al gasto neto.

| Columna | Tipo | Descripción |
|---|---|---|
| `id`, `chain`, `market`, `offer`, `offerdate` | entero / fecha | igual que en `trainHistory` |
| `category`, `company`, `brand`, `offervalue`, `quantity` | entero / decimal | producto y condiciones de la oferta, desde `offers` |
| `repeattrips`, `repeater` | entero / booleano | variable de conteo y variable objetivo |
| `n_lineas` | entero | líneas de transacción en el historial, incluidas devoluciones y descuentos |
| `n_compras` | entero | líneas de compra real |
| `n_devoluciones` | entero | líneas con cantidad negativa |
| `n_descuentos` | entero | líneas con monto negativo y cantidad mayor o igual a 0 |
| `n_visitas` | entero | días distintos con al menos una transacción |
| `gasto_total` | decimal | suma neta de `purchaseamount` en dólares |
| `ticket_promedio` | decimal | `gasto_total / n_visitas` |
| `dias_ultima_compra` | entero | días entre la última transacción y `offerdate` |
| `antiguedad_dias` | entero | días entre la primera transacción registrada y `offerdate` |
| `n_categorias_distintas`, `n_marcas_distintas`, `n_companias_distintas` | entero | diversidad de consumo |
| `n_compras_{categoria,compania,marca}` | entero | compras previas de la categoría, compañía o marca ofertada |
| `gasto_{categoria,compania,marca}` | decimal | monto neto gastado en ellas |
| `n_compras_{categoria,compania,marca}_{30,60,90,180}d` | entero | compras en los últimos N días antes de `offerdate` |
| `n_compras_producto`, `gasto_producto` | entero / decimal | compras y monto del producto exacto: misma categoría, compañía y marca |
| `dias_ultima_compra_categoria` | entero | días desde la última compra de la categoría ofertada; vacío si nunca la compró |
| `nunca_compro_{categoria,compania,marca,producto}` | 0/1 | 1 si no hay compras previas de esa dimensión |

## Joins entre archivos

- `transactions` ↔ `trainHistory`/`testHistory`: por `(id, chain)`.
- `trainHistory`/`testHistory` ↔ `offers`: por `offer`.
- `transactions` ↔ `offers`: por `(category, brand, company)` — `transactions` no tiene la columna `offer` directamente.
