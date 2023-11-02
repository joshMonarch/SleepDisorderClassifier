FROM python:3.9

WORKDIR .

# Copiar el archivo de requisitos
COPY requirements.txt ./requirements.txt

# Instalar las dependencias
RUN pip install -r ./requirements.txt

# Copiar el resto de los archivos del proyecto
COPY . .

# Ejecutar el script de shell para entrenar el modelo usando MLflow
CMD ["sh", "run.sh"]