# Verificación del aporte de Nery

Fecha: 17 de octubre de 2026. Rama: `Proyecto2-Resultados-y-Visualizaciones`.

## Comprobaciones realizadas

- Entorno aislado Python 3.12.10; bibliotecas de modelos fijadas según el manifiesto de Kevin: scikit-learn 1.6.1, numpy 2.1.2, pandas 2.2.3, scipy 1.15.1 y joblib 1.4.2. El manifiesto registra Python 3.12.4 para el entrenamiento, dentro de la misma serie 3.12.
- `python -m pip check`: sin dependencias incompatibles.
- `python app/verificar_paginas.py`: las cuatro páginas abren sin excepciones. El filtro por categoría coincide con el conteo calculado desde el CSV, y una combinación de filtros sin clientes muestra un mensaje.
- El cambio de umbral de 0.50 a 0.70 cambia precisión, sensibilidad y F1. La selección vacía de modelos se maneja con un mensaje.
- El interruptor oculta métricas, tablas y controles; también se verificó en Edge que desaparecen las gráficas.
- Las AUC recalculadas con las probabilidades coinciden con las tablas 08 para los cuatro modelos, con tolerancia absoluta de 1e-12. Se verificaron 35,499 filas e identificadores únicos.
- La página Clasificar de Joel abre con los modelos congelados v2 en el entorno fijado. Su tabla de verificación contiene 100 clientes con `coincide=True`; no se modificó ni se atribuye esa verificación a Nery.
- `python app/rehacer_figuras_eda.py`: se regeneraron frecuencias, tasas y matriz de correlación categóricas desde las tablas existentes. Se conservaron los datos originales.
- Edge mediante Playwright: capturas de Inicio, Explorar, Rendimiento, matriz y rendimiento oculto. Se inspeccionaron visualmente las capturas de exploración y ROC.
- Exportación local: `informe/resultados_revision_local.pdf`, A4, márgenes de una pulgada y texto Times New Roman de 12 puntos, justificado. Las tablas usan letra menor para caber. Se generaron `informe/laminas_nery.pdf` y sus fuentes HTML.

## Pendientes reales del cierre

- Realizar la prueba de uso con una persona externa según `informe/presentacion_nery.md`; la automatización no sustituye esta prueba.
- Conectar Google Drive y pegar/integrar los aportes en el documento del equipo. No hubo sesión de documento conectada y el enlace no fue accesible desde la herramienta web. No se editó el documento remoto.
- Falta `01_introduccion_y_objetivos.md`, de Joel. El 18 de octubre de 2026 se agregó `04_datos_y_limpieza.md` a solicitud del equipo, como resumen de las secciones existentes de carga y preprocesamiento. La próxima compilación lo incluirá antes del análisis exploratorio; el PDF y el HTML locales existentes todavía corresponden a la compilación anterior.
- Revisar en equipo la correspondencia de todas las citas APA y referencias y la numeración final de figuras al integrar las secciones. Los números de figuras del EDA se conservan como referencia al informe anterior, no como numeración definitiva del informe completo.
- El PDF local es una compilación para revisión: no constituye la entrega final ni prueba de integración en Drive. El ensayo grupal y la exposición requieren a los integrantes.

## Reproducción

Desde la raíz del proyecto, usar el Python de `.venv` con `requirements.txt`. Ejecutar `app/verificar_paginas.py` para comprobar la interfaz y `app/rehacer_figuras_eda.py` para regenerar las figuras. Para exportar, instalar `playwright==1.55.0` y `markdown==3.9`, arrancar Streamlit en el puerto local 8502 y ejecutar `informe/compilar_resultados.py`; este script utiliza Microsoft Edge.
