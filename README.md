Sleep Disorder Classifier
===
Este repositorio contiene un proyecto de clasificación de desórdenes del sueño, el cuál utiliza un [dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset) sintético y dos modelos para comparar sus resultados:
-  __XGBoost__: Algoritmo de ML que utiliza árboles de decisión como la base del aprendizaje y emplea técnicas de regularización para mejorar la generalización del modelo.
- __Red neuronal artificial__: Modelo que simula el funcionamiento del cerebro humano. Está constituido por varios _perceptrones_ (neuronas artificiales) que forman capas. Son capaces de aprender patrones complejos de los datos.  

Además, administramos su ciclo de vida por medio de [MLflow](https://mlflow.org/).  
 
 
Preparación del entorno de trabajo  
---
Para este proyecto es necesario:  
- __MySQL__:dolphin:: Ingestaremos los datos procesados en una BD. Será necesario crear la base de datos con la instrucción `CREATE TABLE`.
- __Docker__:whale2:: Levantaremos dos contenedores:
  - __pyspark__: Un contenedor con la imagen de _jupyter/pyspark-notebook_, donde realizaremos la extracción, transformación y carga de datos en la BD. 

    Ejcución del contenedor (si no existe la imagen, se descargará automáticamente):
    ```
    docker run -p 127.0.0.1:8888:8888 -p 4040:4040 -v <ruta-a-la-carpeta-pyspark>:/home/jovyan/work --name pyspark -e DB_HOST=<IP-local> -e DB_USER=<usuario> -e DB_PASSWORD=<constraseña> -e DB_NAME=<nombre-BD> jupyter/pyspark-notebook
    ```
    A mayores, es necesario el Conector/J 8.0 en el contenedor para poder establecer una conexión con la BD. Después de descargar el .jar, se realizará la siguiente ejecución desde el directorio donde se guardó:
    ```
    docker cp mysql-connector-java-8.0.28.jar pyspark:/usr/local/spark-3.5.0-bin-hadoop3/jars/
    ```
  - __mlflow__: Un contenedor con la imagen de _python_, en donde se registran los modelos, las métricas y parámetros con MLflow. La imagen es definida en el archivo _Dockerfile_.  

    Construcción de la imagen:
    ```
    docker build -t python .
    ```  
    Ejecución del contenedor:
    ```
    docker run --rm -e DB_HOST=<IP-local> -e DB_USER=<usuario> -e DB_PASSWORD=<constraseña> -e DB_NAME=<nombre-BD> --name mlflow python
    ```
    En el archivo _requirements.txt_ están definidas todas las dependencias necesarias para esta imagen.  

Esquema de directorios
---
```
/ SleepDisorderClassifier
├── ANN_scripts
│   ├── NeuralNetwork.py
│   ├── test_ANN.py
│   └── train_ANN.py
├── pkl
├── pyspark
│   ├── ETL.ipynb
│   └── Sleep_health_and_lifestyle_dataset.csv
├── SQL_scripts
│   ├── PKs_FKs.sql
│   └── raneg_restrictions.sql
├── XGB_scripts
│   ├── test_XGB.py
│   └── train_XGB.py
├── Dockerfile  
├── mlflow.ipynb  
├── README.md  
├── requirements.txt  
├── run.sh
└── tfmPBI.pbix 
```
- Los directorios _ANN_sripts_ y _XGB_scripts_ contienen los scripts necesarios para entrenar los modelos y guardar los metadatos pertinentes en archivos _pickle_.  
- _SQL_scripts_ contiene dos archivos SQL que endurecen las restricciones de la BD. Deben de ejecutarse después de ingestar los datos procesados.  
- El directorio _pyspark_ contiene el dataset y el archivo .ipynb encargado del tratamiento de los datos y de su ingesta en la BD.  
- En el directorio _pkl_ se guardan todos los archivos .pkl.  
- El archivo _mlflow.ipynb_ registra todos los metadatos que obtiene de los archivos _pickle_.  
- El archivo _run.sh_ se encarga de ejecutar todos los archivos de entreno y testeo de los modelos, y también del archivo _mlflow.ipynb_.

Ejecución de la aplicación  
---
1. Limpieza e ingesta de datos:
    - Levantar el contenedor pyspark.
    - Copiar ahí el Connector/J 8.0.
    - Ejecutar el notebook ETL.ipynb.
    - Ejecutar los scripts del directorio SQL_scripts.
2. Entrenamiento, testeo y registro de metadatos:
    - Construir la imagen de _python_ con el archivo _Dockerfile_.
    - Levantar el contenedor _mlflow_.
    - El archivo _run.sh_ se ejecutará automáticamente y entrenará los modelos, los testeará y registrará en MLflow.
3. Visualizar por pantalla las predicciones realizadas a partir de los modelos que anteriormente han sido registrados en MLflow.
    
#### _Tres estudiantes han trabajado en este proyecto_.

