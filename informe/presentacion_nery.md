# Presentación de Nery · cuatro minutos

## Bloque 4a · 1 minuto 30 segundos

Abrir Explorar datos. Mostrar la tasa global y la comparación con y sin compra previa. Decir: «La recompra pasa de aproximadamente 24.2% a 42.8% cuando existe historial con el producto; es una asociación, no una causa». Filtrar una categoría y señalar cómo cambian la tasa y el número de clientes. Comparar categorías; explicar que los códigos son anónimos y que el tamaño acompaña cada tasa. Entregar a Joel para clasificar un cliente.

## Bloque 4c · 2 minutos 30 segundos

Abrir Rendimiento. Mostrar que la regresión logística presenta la mayor AUC temporal, aproximadamente 0.647. Señalar que son 35,499 clientes etiquetados de la última semana, y no el test de Kaggle. Mostrar las cuatro curvas ROC. Mover el umbral de 0.50 a 0.70 y explicar cómo se intercambian falsos positivos y falsos negativos, sin cambiar el AUC. Mostrar la comparación agrupada por oferta y aclarar que es otro protocolo. Ocultar toda la información con el interruptor y volver a activarlo.

Explicar el diseño: Streamlit reutiliza Python; Plotly permite inspección interactiva. La paleta Okabe-Ito mantiene un color por modelo, un acento azul, contexto gris y el hallazgo en bermellón. Las etiquetas permiten entender el gráfico sin depender solo del color.

## Explicación de la aplicación para el lunes 19 de octubre de 2026

Guion breve para acompañar la demostración y las capturas de la sección `08_aplicacion.md`:

«La aplicación permite explorar los datos, probar los cuatro algoritmos con un cliente y comparar su rendimiento. En Explorar podemos filtrar clientes y revisar su comportamiento. En Clasificar elegimos un ejemplo o cargamos un CSV de compras; la aplicación prepara los datos con las mismas funciones del entrenamiento y muestra la probabilidad y la decisión de cada modelo. Joel explica esta parte de la demostración. En Rendimiento comparamos las curvas ROC y los errores: al mover el umbral cambian las decisiones, pero no el AUC.

Elegimos Python para reutilizar el trabajo del proyecto y Streamlit para construir las páginas, los filtros y la carga de archivos. pandas y NumPy manejan las tablas y los cálculos. Plotly permite consultar valores y comparar las gráficas de forma interactiva. scikit-learn y joblib permiten usar los modelos guardados con sus transformaciones, sin entrenarlos durante la demostración. Las tablas se intercambian en CSV y Parquet. Las versiones están fijadas para reproducir el entorno».

Si no se puede realizar la demostración en vivo, mostrar las capturas de Inicio, Explorar, Clasificar y Rendimiento incluidas en el informe y seguir el mismo recorrido. La exposición corresponde al equipo; este texto queda como apoyo para el ensayo.

## Prueba con una persona externa (pendiente)

No se registra como realizada hasta observar a una persona ajena al grupo. Pedirle, sin indicar qué botones tocar: filtrar una categoría, encontrar el tamaño del grupo, consultar una variable usada por los modelos, identificar el modelo con mayor AUC, cambiar el umbral y ocultar el rendimiento. Registrar fecha, duración, tareas completadas sin ayuda, confusiones y cambios propuestos. No recoger información personal innecesaria.

| Tarea | Completada sin ayuda | Tiempo | Observación |
|---|---|---|---|
| Filtrar y encontrar el tamaño del grupo | Pendiente | — | — |
| Consultar una variable del modelo | Pendiente | — | — |
| Identificar el mayor AUC | Pendiente | — | — |
| Cambiar el umbral e interpretar errores | Pendiente | — | — |
| Ocultar el rendimiento | Pendiente | — | — |

Usar las capturas `04_app_*` como respaldo y las capturas `09_app_*` de Joel para Clasificar. El ensayo completo con los cuatro integrantes y la exposición se realizan por el equipo.
