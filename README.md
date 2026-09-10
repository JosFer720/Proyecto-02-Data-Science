# Proyecto 2 — Predicción de compradores recurrentes

Curso CC3092 Deep Learning y Sistemas Inteligentes, UVG. Reto 11 de la guía del Proyecto 2 (categoría Negocios), resuelto con el dataset de Kaggle [Acquire Valued Shoppers Challenge](https://www.kaggle.com/c/acquire-valued-shoppers-challenge/data) (autorizado por el profesor en reemplazo del dataset de Tianchi).

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

```bash
pip install -r requirements.txt
python src/download_dataset.py   # extrae los CSV a data/raw/
jupyter notebook notebooks/01_carga_y_eda_inicial.ipynb
```

