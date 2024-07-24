# Download image
FROM python:3.9

# stablish app as workplace
WORKDIR /app

COPY requirements.txt ./requirements.txt

# Install dependencies
RUN pip install -r ./requirements.txt

# Copy the necessary files
COPY ANN_scripts ./ANN_scripts
COPY XGB_scripts ./XGB_scripts
COPY pkl ./pkl
COPY run.sh Dockerfile mlflow.ipynb .

# Execute shell script to train and test models, and to monitor metrics, artifacts and parameters using mlflow
CMD ["sh", "run.sh"]