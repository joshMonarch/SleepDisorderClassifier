# SleepDisorderClassifier
En este proyecto utilizaremos el dataset [Sleep Health and Lifestyle Dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset)
para clasificar segun el desorden del sueño, que reúne tres opciones:  
* Sleep Apnea
* Insomnia
* None
 
Las herramientas que utilizaremos serán:
* Docker: Levantaremos dos contenedores:
    + PySpark-Notebook: Aquí desarrollaremos el codigo para limpiar e inyectar los datos en la base de datos.
    + MLflow: Aquí resgistraremos los modelos con sus métricas y parámetros.
* MySQL: Nuestra base de datos en local se llamará *sleep*.
* Power BI Desktop: Para visualizar los datos.  
 
## Recolección, transformación e inyección de datos  
Se levantará el contenedor de *PySpark-Notebook*, creando además un volumen que será el lugar de trabajo de Jupyter. En el archivo *starModelToMySQL.ipynb* tendrá lugar este paso del proyecto. Una vez ejecutado deberemos realizar las restricciones necsarias a las tablas creadas con los
scripts SQL que se encuentran en el directorio *scripts_sql*:  
* PKs_FKs.sql: Aquí marcaremos las columnas que harán de primary key y foreign key.
* range_restrictions.sql: Aquí marcaremos los límites que deberan cumplir los datos para poder formar parte de las tablas.  

## Visualización de datos  
Obtendremos los datos ya limpios desde la BD utilizando Power BI Desktop y generaremos el archivo *tfmPBI.pbix*.  

## Entreno y testeo de los modelos
Se levantará el contenedor *mlflow* a partir del archivo *Dockerfile*. Una vez levantado, este contenedor ejecutará *run.sh*, el cual ejecutara los scripts en el siguiente orden:
1. train_ANN.py 
2. test_ANN.py
3. train_XGB.py
4. test_XGB.py
5. mlflow.ipynb

Se utilizarán archivos *.pkl* para guardar los modelos y las variables pertinentes para su posterior uso. En el archivo *mlflow.ipynb* se utilizará MLflow para registrar los modelos, los
parámetros y las métricas; y por último, se realizarán predicciones con los modelos ya registrados.  

## Otros archivos y directorios  
* requirements.txt: Los requerimientos que necesitamos para levantar el contenedor con Dockerfile.
* credenciales.json: Las credenciales para conectarse a la base de datos tanto para inyectar los datos como para extraerlos.
* loss_accuracy_plot.png: Gráfico que representa la variacion de la precisión y la pérdida de la red neuronal durante las distintas épocas del entreno.
* NeuralNetwork.py: Script que define el modelo de la red neuronal.
* pkl: Directorio donde se guardan todo los archivos *pkl* generados.
