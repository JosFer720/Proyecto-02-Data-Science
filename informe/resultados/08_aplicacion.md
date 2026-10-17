# Resultados: aplicación interactiva

La aplicación permite recorrer tres tareas desde Inicio: explorar el conjunto limpio, clasificar clientes y consultar el rendimiento. Se implementó con Streamlit para reutilizar los módulos de Python del proyecto, el preprocesamiento y los modelos congelados; Plotly permite consultar puntos, seleccionar series y ampliar rangos sin generar nuevas imágenes. Los datos de exploración y evaluación se almacenan en caché y se invalidan cuando cambia el archivo de origen. Las páginas de exploración y rendimiento no cargan modelos.

![Inicio y navegación entre las tareas](../../results/figures/04_app_inicio.png)

## Exploración del conjunto limpio

Explorar datos ofrece filtros combinables de oferta, cadena y categoría, indicadores del tamaño de selección y la recompra, y una referencia global fija de 27.14%. Una primera gráfica compara clientes con y sin historial de compra del producto. Las tasas y tamaños se recalculan con cada selección; la anotación aclara el hallazgo del conjunto completo. Una segunda gráfica compara las tasas de los 15 grupos más frecuentes de la dimensión elegida. La distribución de variables prioriza las entradas declaradas en el manifiesto de modelos y permite consultar faltantes, estadísticas y el diccionario. La página permite descargar los clientes filtrados. Si una combinación no tiene datos, muestra un mensaje y permite corregir los filtros.

![Exploración y hallazgo del historial del producto](../../results/figures/04_app_explorar.png)

## Clasificación de datos nuevos

La página Clasificar acepta las transacciones de un cliente o un cliente de ejemplo, prepara las entradas con las mismas funciones del entrenamiento y compara las probabilidades de uno o de todos los algoritmos congelados. Su funcionamiento, sus reglas de validación y su verificación se describen en el apartado siguiente.

![Ingreso de datos para clasificar](../../results/figures/09_app_clasificar_datos.jpg)

![Resultados de la clasificación](../../results/figures/09_app_clasificar_resultado.jpg)

## Rendimiento interactivo

Rendimiento lee las métricas, probabilidades y divisiones agrupadas de las tres tablas de evaluación. Muestra curvas ROC de los modelos seleccionados, una matriz de confusión con umbral ajustable, precisión, sensibilidad y F1 recalculados, y una comparación de las métricas oficiales. Las métricas oficiales conservan los umbrales elegidos en validación; el control de umbral es exploratorio y no cambia la evaluación oficial ni selecciona hiperparámetros con la prueba. La AUC permanece independiente del umbral.

![Curvas ROC de la semana de prueba](../../results/figures/04_app_rendimiento.png)

![Errores al cambiar el umbral](../../results/figures/04_app_umbral.png)

Se distingue explícitamente la semana de prueba temporal, del 24 al 30 de abril de 2013, de la validación agrupada por oferta. Esta última muestra la media y desviación estándar de AUC y el número de divisiones. También se presentan las AUC temporales por ofertas vistas y no vistas cuando están disponibles. El conjunto de prueba de Kaggle carece de etiquetas y no se presenta como una evaluación supervisada.

Un interruptor situado antes de los resultados oculta toda la información de rendimiento. Si faltan las tablas requeridas, la página muestra cómo generarlas, sin entrenar ni estimar resultados de reemplazo.

![Rendimiento oculto a petición del usuario](../../results/figures/04_app_oculto.png)

## Color y presentación

Se utiliza la paleta Okabe-Ito del módulo común de estilo del proyecto. El azul `#0072B2` guía la navegación y las comparaciones generales; el bermellón `#D55E00` destaca el hallazgo y el gris proporciona contexto. Regresión logística, Random Forest, Gradient Boosting y perceptrón multicapa conservan sus colores azul, naranja, verde y rosa en todas las páginas. Nombres, cifras y etiquetas acompañan el color para no depender de su percepción. Las barras de tasas parten de cero y la correlación utiliza una escala divergente centrada en cero. El tema tiene fondo claro, texto oscuro y rejillas discretas.

## Verificación y alcance

Las páginas se verificaron con pruebas automáticas que abren cada una y ejercitan sus controles: los filtros, una combinación de filtros sin clientes, el cambio de umbral, la selección vacía de modelos y el interruptor que oculta el rendimiento. Además, las AUC recalculadas con las probabilidades guardadas coinciden con las de la evaluación en los cuatro modelos. La comparación de variables y probabilidades de la página Clasificar en 100 clientes se describe en el apartado siguiente.

Estas comprobaciones no sustituyen una prueba de uso con personas. La prueba con una persona externa al grupo queda pendiente.
