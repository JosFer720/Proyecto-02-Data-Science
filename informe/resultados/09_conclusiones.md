# Conclusiones

La regresión logística versión 2 obtuvo el mejor resultado en la semana de prueba, con AUC 0.6470, sensibilidad 0.8848 y F1 0.5448. El perceptrón multicapa quedó prácticamente empatado en discriminación, con AUC 0.6461, y alcanzó la mayor precisión, 0.4154. Se selecciona la regresión logística como modelo de referencia por su mayor AUC y F1, aunque el perceptrón constituye una alternativa si se busca reducir la cantidad de contactos positivos.

El objetivo específico 1 se cumplió al evaluar regresión logística, Random Forest, Gradient Boosting y perceptrón multicapa sobre la misma semana futura. Sus AUC fueron 0.6470, 0.6151, 0.6121 y 0.6461, respectivamente. El reentrenamiento hasta el 23 de abril cambió el orden observado con la primera versión y confirmó la importancia de incorporar toda la información disponible antes de la prueba final.

El objetivo específico 2 se cumplió mediante la búsqueda de hiperparámetros en la semana de validación y el reentrenamiento posterior con entrenamiento y validación. Los hiperparámetros y umbrales se conservaron al generar la versión 2, por lo que la semana de prueba no intervino en su elección.

El objetivo específico 3 se cumplió al evaluar ofertas vistas y no vistas y realizar cinco divisiones agrupadas. El 96.3% de la prueba correspondió a ofertas no vistas. La regresión logística y el perceptrón mostraron las menores degradaciones entre ofertas vistas y no vistas, de 0.0169 y 0.0198. En las divisiones agrupadas, Gradient Boosting obtuvo el mayor AUC promedio, 0.6197, pero con una desviación estándar de 0.0659. Esto confirma que el rendimiento debe monitorearse por campaña.

El objetivo específico 4 se cumplió mediante importancia por permutación. El gasto previo en el producto ofertado fue la señal principal de la regresión logística y del perceptrón multicapa. Para Gradient Boosting dominó la categoría de la oferta, seguida del gasto en el producto. El historial relacionado con el producto y su categoría sostiene una parte importante de la predicción.

Para apoyar la integración de la aplicación se generó `08_predicciones_prueba.parquet`, con 35,499 filas y una probabilidad por modelo. La coincidencia entre la aplicación y estos resultados debe verificarse mediante una prueba de integración antes de declararla completa.

Los umbrales de validación clasificaron como recompra entre 64.6% y 80.7% de los clientes. Esta amplitud favorece la sensibilidad, pero produce numerosos falsos positivos. Antes del despliegue se recomienda calibrar las probabilidades y elegir el umbral mediante una función de valor esperado que incorpore el costo del descuento, el margen de la recompra y el costo de contactar a un cliente que no volverá.
