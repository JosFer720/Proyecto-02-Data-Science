# Herramientas utilizadas

## Recursos de cómputo

El análisis exploratorio, el entrenamiento y el ajuste de los modelos se ejecutaron en una computadora personal con sistema operativo Windows 11, procesador Intel Core i5-12400 de 6 núcleos físicos y 12 hilos, 32 GB de memoria RAM y una tarjeta gráfica NVIDIA GeForce RTX 4060 Ti con 8 GB. Los algoritmos de scikit-learn usados en este proyecto se ejecutan en el procesador, por lo que la tarjeta gráfica no intervino en el entrenamiento.

La evaluación de los modelos en la semana de prueba se hizo en otra computadora con Windows 11 Home, procesador AMD Ryzen 5 7600 de 6 núcleos y 12 hilos, 32 GB de memoria RAM y una tarjeta gráfica NVIDIA GeForce RTX 3060 Ti, además de los gráficos integrados AMD Radeon. La evaluación se ejecutó también en el procesador, con Python 3.13.2 y las mismas versiones de scikit-learn, pandas, numpy, scipy y joblib de la tabla siguiente.

La mayor carga del proyecto fue la lectura de los 349,655,789 registros de transacciones, que se reducen a una fila por cliente antes de entrenar. Con el conjunto ya reducido, ningún algoritmo tarda más de cinco minutos en entrenar, de modo que la búsqueda de hiperparámetros cabe en una sola sesión de trabajo sin necesidad de un servicio de cómputo en la nube.

## Lenguaje, bibliotecas y paquetes

El entrenamiento y el ajuste se hicieron con Python 3.12.4 y todo el código se versiona en GitHub. El trabajo se organizó en cuadernos de Jupyter, uno por etapa, y en módulos reutilizables dentro de la carpeta `src`. Los paquetes que intervienen en los modelos y sus versiones son los siguientes.

| Paquete | Versión | Uso en el proyecto |
|---|---|---|
| scikit-learn | 1.6.1 | Pipelines de preparación, los cuatro algoritmos, búsqueda aleatoria y métricas |
| pandas | 2.2.3 | Carga, agregación por cliente y tablas de resultados |
| numpy | 2.1.2 | Cálculo numérico y transformación logarítmica |
| scipy | 1.15.1 | Distribuciones de los rangos de la búsqueda aleatoria |
| joblib | 1.4.2 | Guardado y carga de los modelos entrenados |
| matplotlib | 3.9.2 | Figuras estáticas del informe |
| pyarrow | 25.0.1 | Lectura y escritura de tablas en formato Parquet |

Pedregosa et al. (2011) describen scikit-learn, la biblioteca que aporta la regresión logística, el Random Forest, el Gradient Boosting basado en histogramas y el perceptrón multicapa usados en este trabajo.

Los modelos finales se guardan con las versiones exactas de la tabla anterior. Un archivo de joblib solo es confiable con la misma versión de scikit-learn con que se creó, así que cada modelo tiene una ficha con la huella SHA-256 del archivo y las versiones del entorno. El cargador común comprueba ambas cosas antes de usar un modelo y se detiene si algo no coincide. Esto garantiza que la aplicación, las tablas del informe y las gráficas usen exactamente los mismos modelos.
