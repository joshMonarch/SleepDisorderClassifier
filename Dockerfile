# Descargar imagen de python 3.9
FROM python:3.9

# Establecer app como directorio de trabajo
WORKDIR /app

COPY requirements.txt ./requirements.txt

# Instalar las dependencias
RUN pip install -r ./requirements.txt

# Copiar el resto de los archivos y directorios a utilizar
COPY ANN_scripts ./ANN_scripts
COPY XGB_scripts ./XGB_scripts
COPY pkl ./pkl
COPY run.sh Dockerfile mlflow.ipynb .

# Ejecutar el script de shell para entrenar y testear los modelos, además de
# hacer un seguimiento de las métricas, artefactos y parámetros con mlflow 
CMD ["sh", "run.sh"]