# Proyecto MLOps de Steam
## Introducción
¡Bienvenido al proyecto MLOps de Steam! En este proyecto, asumiremos el rol de un Ingeniero MLOps en Steam, una plataforma de juegos multinacional. Nuestra misión es crear un sistema de recomendación de videojuegos utilizando aprendizaje automático. Los datos necesitan ser refinados, y nuestra tarea es transformarlos en un estado utilizable, desarrollar un Producto Mínimo Viable (MVP) y desplegarlo como una API RESTful.

## Descripción del Problema
Como Científico de Datos en Steam, nuestra tarea es crear un modelo de aprendizaje automático para un sistema de recomendación de videojuegos. El estado actual de los datos es crudo y no procesado, lo que dificulta el trabajo. Nuestro objetivo es empezar desde cero, realizar tareas rápidas de Ingeniería de Datos y entregar un MVP al final del proyecto.

## Información del Juego en Steam
En este proyecto, trabajamos con tres archivos JSON que contienen datos cruciales sobre los juegos en la plataforma Steam. Cada archivo aporta una perspectiva única:
* australian_user_reviews.json: <br>
Este conjunto de datos es como un caleidoscopio de opiniones de usuarios sobre los juegos que han experimentado en Steam. Ofrece detalles sobre si recomendaron o no un juego y estadísticas sobre la utilidad de los comentarios. Además, revela el ID del usuario, su URL de perfil y el ID del juego que están comentando.

* australian_users_items.json:<br>
Aquí, obtenemos una visión panorámica de los juegos que cada usuario ha jugado y cuánto tiempo les han dedicado.

* output_steam_games.json: <br>
Este conjunto de datos proporciona una ventana a los propios juegos en Steam. Incluye información vital como títulos, desarrolladores, precios, características técnicas y etiquetas.

## Flujo de Trabajo Propuesto

### Ingeniería de Datos

- **Limpieza y Transformación de Datos:** Enfoque inicial en leer el conjunto de datos en el formato correcto. Eliminar columnas innecesarias para optimizar el rendimiento de la API y el entrenamiento del modelo. En este proyecto el etl se divide entre los tres conjuntos de datos que fueron proporcionados y ofrecen informacion acerca de las distintas caracteristicas y opiniones de los juegos presentes en la plataforma.
["ETL Output"](01_ETL_Output.ipynb)
["ETL Users Items"](01_ETL_user_items.ipynb)
["ETL Users Reviews"](01_ETL_user_reviews.ipynb)

- **Análisis de Sentimiento:** Crear una nueva columna, 'sentiment_analysis', aplicando análisis de sentimiento mediante Procesamiento de Lenguaje Natural (NLP) a las reseñas de usuarios. La escala que se utilizo fue: '0' para comentarios negativos, '1' para neutrales y '2' para positivos.

### Análisis Exploratorio de Datos (EDA)

- **Exploración Manual:** Realizar un EDA manual después del ETL para investigar las relaciones entre variables, identificar valores atípicos y descubrir patrones interesantes dentro del conjunto de datos, para esta tarea se utilizan diferentes librerias para hacer visualizaciones y medidas estadisticas. ["EDA"](02_EDA.ipynb)


### Ingeniería de Características

- **Creación de DataFrames Auxiliares:** Antes de desarrollar las funciones de la API, se crearon DataFrames auxiliares para optimizar el espacio y mejorar el rendimiento de las funciones. Estos DataFrames se utilizaron para almacenar datos específicos necesarios para las consultas de la API. ["Dataframes Auxiliares](03_DataFrames_Auxiliares.ipynb)


### Desarrollo de la API

- **Framework:** Utilizar el framework FastAPI para exponer los datos de la empresa a través de endpoints RESTful.
- **Endpoints:**
  - `PlayTimeGenre(genero: str)`: Devuelve el año de lanzamiento con más horas jugadas para el género especificado.
  - `UserForGenre(genero: str)`: Devuelve el usuario con más horas jugadas para el género dado y una lista de acumulación de horas jugadas por año.
  - `UsersRecommend(año: int)`: Devuelve el top 3 de juegos más recomendados por usuarios para el año especificado.
  - `UsersWorstDeveloper(año: int)`: Devuelve el top 3 de desarrolladoras con juegos menos recomendados por usuarios para el año especificado.
  - `sentiment_analysis(empresa_desarrolladora: str)`: Devuelve un diccionario con el recuento de análisis de sentimiento para reseñas asociadas con el desarrollador de juegos especificado.

  - Dentro de la carpeta Datasets se encuentran los dataframes utilizados para cada funcion y en este archivo se probaron la funciones antes de utilizar FastAPI ["Funciones API"](04_API_Funciones.ipynb)

### Modelo de Aprendizaje Automático

- **Sistema de Recomendación:** Se implementa un sistema de recomendación de filtrado colaborativo item-item. En este caso el sistema de recomendacion funciona tomando un item y encontrando cinco similares a este. Para poder lograr deployar el modelo con espacio de memoria limitado se utiliza la tecnica de muestreo, esto significa que se usa solo una muestra de los datos para realizar la recomendacion aunque esto puede conllevar a predicciones no tan acertadas.
["Modelo de Recomendacion"](05_Modelo_Recomendacion.ipynb)
- **Integración con la API:** Asegurar que las recomendaciones del modelo puedan accederse a través de un endpoint API GET/POST, como `recomendacion_juego(id_juego)`.

### Implementación en la Nube
Para poner en funcionamiento la API que hemos desarrollado, optamos por la plataforma Render.Una vez que hemos completado y probado nuestra API localmente, Render nos brinda la capacidad de llevarla a la web de manera sencilla y automatizada. Esto simplifica significativamente el proceso de poner nuestra aplicación en línea y asegura una implementación rápida y confiable.
Se deja a continuacion el link para ingresas a ["Render](https://asguzman-app-mlops-steam.onrender.com)

### Video
Apartir de este ["Video](https://www.youtube.com/watch?v=YuxJReXzwhM) procedemos a explicar el principal desarrollo del proyecto y demostrar el funcionamiento de la API deployada.

## Conclusión

Este proyecto integral de MLOps tiene como objetivo transformar datos de juegos en bruto en un sistema funcional de recomendación desplegado como una API RESTful. La optimización del espacio mediante DataFrames auxiliares es una estrategia clave para mejorar el rendimiento de las funciones al igual que utilizar el muestreo en el modelo de recomendacion. Al abordar la ingeniería de datos, la ingeniería de características, el desarrollo de la API, el EDA y el aprendizaje automático, buscamos proporcionar información valiosa y recomendaciones a los usuarios de Steam acerca de los juegos presentes en la plataforma.

## Tecnologías Utilizadas
En el desarrollo de este proyecto, aprovechamos varias tecnologías para llevar a cabo las distintas etapas del proceso:
* Python: Utilizamos el lenguaje de programación Python como piedra angular para la implementación de la lógica del proyecto.
* Pandas y NumPy: Estas bibliotecas de Python fueron esenciales para la manipulación eficiente y el análisis de datos, permitiéndonos realizar operaciones complejas en los conjuntos de datos.
* Matplotlib y Seaborn: Estas poderosas librerías de visualización nos permitieron explorar patrones y relaciones en los datos a través de gráficos y visualizaciones claras.
* FastAPI: Para exponer nuestros datos al mundo, optamos por FastAPI, un marco moderno y rápido para la construcción de APIs en Python.
* Cosine Similarity (Scikit-learn): Implementamos la similitud del coseno utilizando Scikit-learn para construir un sistema de recomendación basado en ítems.
* Render: Para la implementación en la nube y el despliegue automático desde GitHub, seleccionamos Render como plataforma de elección.
* Análisis de Sentimientos con NLP: Aplicamos técnicas de Procesamiento del Lenguaje Natural (NLP) para realizar análisis de sentimientos en las reseñas de usuarios y asignar etiquetas correspondientes.
