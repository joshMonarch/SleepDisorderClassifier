Sleep Disorder Classifier
===
This repository contains a sleep disorder classification project that uses a synthetic [dataset](https://www.kaggle.com/datasets/uom190346a/sleep-health-and-lifestyle-dataset) and two models to compare their results:
-  __XGBoost__: An ML algorithm that uses decision trees as the base of learning and employs regularization techniques to improve the model's generalization.
- __Artificial Neural Network__: A model that simulates the functioning of the human brain. It is made up of several _perceptrons_ (artificial neurons) that form layers. They are capable of learning complex patterns from the data.  

Additionally, we manage its life cycle using [MLflow](https://mlflow.org/).  
 
 
Setting Up the Work Environment  
---
For this project, you will need:  
- __MySQL__:dolphin:: Ingestion of processed data into a database. You will need to create the database using the `CREATE TABLE` statement.
- __Docker__:whale2:: Two containers will be launched:
  - __pyspark__: Using the _jupyter/pyspark-notebook_ image. Here, it takes place the extraction, transformation, and loading of data into the database. 

    Running the container (if the image does not exist, it will be downloaded automatically):
    ```
    docker run -p 127.0.0.1:8888:8888 -p 4040:4040 -v <path-to-pyspark-folder>:/home/jovyan/work --name pyspark -e DB_HOST=<local-IP> -e DB_USER=<user> -e DB_PASSWORD=<password> -e DB_NAME=<db-name> jupyter/pyspark-notebook
    ```
    Additionally, the Connector/J 8.0 is needed in the container to establish a connection with the database. After downloading the .jar file, execute the following command from the directory where it is saved:
    ```
    docker cp mysql-connector-java-8.0.28.jar pyspark:/usr/local/spark-3.5.0-bin-hadoop3/jars/
    ```
  - __mlflow__: Using the _python_ image; models, metrics, and parameters are registered using MLflow. The image is defined in the _Dockerfile_.  

    Building the image:
    ```
    docker build -t python .
    ```  
    Running the container:
    ```
    docker run --rm -e DB_HOST=<local-IP> -e DB_USER=<user> -e DB_PASSWORD=<password> -e DB_NAME=<db-name> --name mlflow python
    ```
    The _requirements.txt_ file defines all the dependencies needed for this image.  

Directory Structure
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
│   └── range_restrictions.sql
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
- The _ANN_scripts_ and _XGB_scripts_ directories contain the scripts needed to train the models and save the relevant metadata in _pickle_ files.  
- _SQL_scripts_ contains two SQL files that enforce database restrictions. They must be executed after ingesting the processed data.  
- The _pyspark_ directory contains the dataset and the .ipynb file responsible for data processing and ingestion into the database.  
- The _pkl_ directory stores all the .pkl files.  
- The _mlflow.ipynb_ file logs all the metadata obtained from the _pickle_ files.  
- The _run.sh_ file is responsible for running all the training and testing scripts for the models, as well as the _mlflow.ipynb_ file.

Application Execution  
---
1. Data Cleaning and Ingestion:
    - Start the pyspark container.
    - Copy the Connector/J 8.0 there.
    - Run the ETL.ipynb notebook.
    - Execute the scripts in the SQL_scripts directory.
2. Training, Testing, and Metadata Registration:
    - Build the _python_ image with the _Dockerfile_.
    - Start the _mlflow_ container.
    - The _run.sh_ file will run automatically, train the models, test them, and register them in MLflow.
3. Display the predictions made from the models previously registered in MLflow.
    
#### _Three students worked on this project_.
