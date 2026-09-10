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

## Joins entre archivos

- `transactions` ↔ `trainHistory`/`testHistory`: por `(id, chain)`.
- `trainHistory`/`testHistory` ↔ `offers`: por `offer`.
- `transactions` ↔ `offers`: por `(category, brand, company)` — `transactions` no tiene la columna `offer` directamente.
