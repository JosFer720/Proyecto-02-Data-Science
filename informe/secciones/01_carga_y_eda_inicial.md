# 1. Introducción y descripción inicial de los datos

## 1.1 Situación problemática

Cuando un negocio ofrece un cupón de descuento sobre un producto, tiene dos posibles resultados con el mismo cliente: que compre el producto una sola vez aprovechando el descuento (comportamiento de cazador de ofertas) o que se convierta en comprador recurrente de esa marca o categoría incluso después de que termine la promoción. Para el negocio, el segundo grupo es mucho más valioso porque representa un retorno sostenido de la inversión en la promoción, mientras que el primero solo genera una venta puntual que en muchos casos ni siquiera es rentable una vez descontado el valor del cupón. El problema es que, al momento de diseñar y emitir la oferta, el negocio no sabe de antemano a cuál grupo pertenecerá cada cliente. Poder distinguir con anticipación qué clientes tienen mayor probabilidad de volverse recurrentes permite dirigir mejor el presupuesto de marketing, diseñar cupones más efectivos y evaluar objetivamente si una campaña de descuentos realmente construye lealtad o solo subsidia compras que iban a pasar de todas formas.

## 1.2 Problema científico

Dado un cliente que redimió un cupón de descuento sobre un producto específico, y su historial de compras de al menos un año previo a que se le ofreciera el cupón, se plantea un problema de **clasificación binaria**: estimar la probabilidad de que ese cliente vuelva a comprar la misma marca/categoría/compañía del producto ofertado durante o después del periodo promocional (`repeater = True` si `repeattrips > 0`, es decir, si el cliente tuvo al menos una compra de repetición). La variable objetivo es `repeater`, derivada de `repeattrips` en `trainHistory.csv`, y la métrica de evaluación del reto es el área bajo la curva ROC (AUC).

## 1.3 Objetivos

**Objetivo general:**
Construir un modelo de clasificación binaria que estime la probabilidad de que un cliente que redimió un cupón de descuento vuelva a comprar el mismo producto, a partir de su historial de compras previo, y evaluar su desempeño con la métrica AUC.

**Objetivos específicos:**
1. Caracterizar mediante análisis exploratorio de datos la relación entre las variables de comportamiento de compra histórico del cliente (gasto total, frecuencia de compra, recencia) y la probabilidad de recompra, cuantificando dicha relación con coeficientes de correlación y tasas de recompra por segmento.
2. Comparar el desempeño (AUC) de un modelo de línea base (regresión logística con variables mínimas) contra un modelo con variables de comportamiento adicionales derivadas del EDA, cuantificando la mejora obtenida en puntos porcentuales de AUC.

## 1.4 Descripción de los datos (primera mitad)

El reto usa el dataset de Kaggle "Acquire Valued Shoppers Challenge", compuesto por 5 archivos (ver diccionario completo en [`codebook.md`](../../codebook.md)). Los campos categóricos como `category`, `brand`, `company`, `dept`, `chain` y `market` vienen anonimizados como identificadores numéricos sin significado semántico expuesto, así que el análisis exploratorio se reporta por frecuencia y por ID, sin poder interpretar a qué producto o cadena corresponde cada identificador.

Cargando los archivos con `src/carga.py` se obtiene:

| Archivo | Observaciones | Variables | Descripción |
|---|---|---|---|
| `trainHistory.csv` | 160,057 | 7 | historial de ofertas de entrenamiento: `id`, `chain`, `offer`, `market`, `repeattrips`, `repeater`, `offerdate`. Trae la variable objetivo. Corresponde a ofertas emitidas antes del 2013-05-01. |
| `testHistory.csv` | 151,484 | 5 | mismas columnas que `trainHistory.csv` excepto `repeattrips` y `repeater` (es lo que hay que predecir). Ofertas emitidas en o después del 2013-05-01. |
| `offers.csv` | 37 | 6 | catálogo de las 37 ofertas distintas del reto: `offer`, `category`, `quantity`, `company`, `offervalue`, `brand`. |

El train y el test están separados por fecha de emisión de la oferta (`offerdate`), no por un muestreo aleatorio — esto es relevante para el modelado posterior (notebook 05), porque un split aleatorio de validación infla artificialmente el AUC comparado con el corte temporal real del reto.

`trainHistory.csv` trae `repeater` codificado como texto (`'t'`/`'f'`) en el CSV crudo; `src/carga.py` lo convierte a booleano nativo de pandas al cargar.

### Balance de la variable objetivo

De los 160,057 clientes en `trainHistory.csv` que redimieron un cupón, el **27.14%** (43,438 clientes) volvió a comprar el producto ofertado (`repeater = True`), mientras que el **72.86%** (116,619 clientes) no lo hizo. Hay un desbalance de clases moderado (aproximadamente 1:2.7) que debe tenerse en cuenta al evaluar el modelo en el notebook 05/06 — con AUC como métrica esto es menos problemático que con accuracy, pero conviene reportarlo explícitamente.

### Cruce con `offers.csv`

Al cruzar `trainHistory.csv` con `offers.csv` por la columna `offer`, se observa que las 160,057 ofertas emitidas en train corresponden a solo 37 ofertas distintas (cada oferta se repite miles de veces entre distintos clientes). El valor del descuento (`offervalue`) varía entre $0.75 y $5 en las ofertas del dataset, y la cantidad mínima de unidades requerida (`quantity`) es en su mayoría 1, con algunas ofertas que requieren más unidades para activar el descuento. El detalle de la distribución de categorías, marcas, compañías ofertadas y de `offervalue`/`quantity` está en las tablas `results/tables/01_categorias_ofertadas.csv`, `01_marcas_ofertadas.csv`, `01_companias_ofertadas.csv` y `01_resumen_offervalue_quantity.csv`, y en las figuras `results/figures/01_balance_repeater.png` y `01_offervalue_quantity.png`, generadas en `notebooks/01_carga_y_eda_inicial.ipynb`.

El cruce completo se guardó en `data/processed/train_offers.csv` (160,057 filas, una por cliente/oferta de train), que sirve de insumo para el análisis posterior sobre `transactions.csv`.
