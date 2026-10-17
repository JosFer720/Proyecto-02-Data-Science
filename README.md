# Proyecto 2 — Predicción de compradores recurrentes

Curso CC3084 Data Science, UVG. Reto 11 de la guía del Proyecto 2 (categoría Negocios), resuelto con el dataset de Kaggle [Acquire Valued Shoppers Challenge](https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data) (autorizado por el profesor en reemplazo del dataset de Tianchi).

## Problema

A un conjunto de clientes se les ofrece un cupón de descuento sobre un producto puntual. De los que redimen el cupón, la tarea es predecir cuáles volverán a comprar ese mismo producto durante o después del periodo promocional, usando su historial de compras previo. Se plantea como clasificación binaria y se evalúa con AUC.

## Datos

Descargar desde la página del reto en Kaggle y colocar en `data/raw/` (ver `src/download_dataset.py`). El diccionario de datos completo está en [`codebook.md`](codebook.md).

## Estructura del repositorio

```
├── data/
│   ├── raw/            # CSVs originales de Kaggle (no se suben a git)
│   └── processed/      # datasets intermedios generados por los notebooks
├── notebooks/          # 01 a 06, un notebook por etapa del análisis
├── src/                # config.py, download_dataset.py, carga.py, preprocesamiento.py, analisis.py, modelo.py
├── results/
│   ├── figures/
│   ├── modelos/
│   └── tables/
└── informe/
    ├── secciones/      # una sección .md por notebook
    └── presentacion.pptx
```

## Cómo correr

### Aplicación de resultados y visualizaciones

Usar Python 3.12 (los modelos se guardaron con Python 3.12.4). Las versiones
de scikit-learn, numpy, pandas, scipy y joblib se fijan según el manifiesto real.
Desde la raíz del repositorio, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m streamlit run app/Inicio.py
```

Abrir la dirección local que imprime Streamlit. La navegación lateral contiene
Explorar datos, Clasificar y Rendimiento. Explorar permite combinar filtros de
oferta, cadena y categoría; Rendimiento permite elegir modelos, cambiar el umbral
y ocultar toda la evaluación con su interruptor. Cambiar el umbral es una
exploración de errores, no un ajuste del modelo con el conjunto de prueba.

Se necesita `data/processed/dataset_modelo.csv` (notebooks 01 y 02). Rendimiento
lee exclusivamente las tres tablas `08_*` del contrato; Clasificar requiere
los modelos v2, el manifiesto y los ejemplos preparados por Joel. No se entrena
ningún modelo desde la app. Si faltan los datos de exploración o evaluación,
sus páginas muestran instrucciones para generarlos.

Para regenerar las figuras categóricas: `python app/rehacer_figuras_eda.py`.
Los textos de Nery y el estado de integración están en `informe/resultados/`;
el guion y la prueba de uso, en `informe/presentacion_nery.md`.

### Análisis y preparación de datos

```bash
pip install -r requirements.txt
python src/download_dataset.py   # extrae los CSV a data/raw/
jupyter notebook notebooks/01_carga_y_eda_inicial.ipynb
```

