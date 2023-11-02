import json
import pandas as pd
import pymysql
from xgboost import XGBClassifier;
from sklearn.model_selection import train_test_split
import mlflow

if __name__ == "__main__":

    with open("./credenciales.json", "r") as archivo:
        cred = json.load(archivo)

    # Establecer la conexión a la base de datos
    connection = pymysql.connect(
        host=cred["host_docker"],
        user=cred["user"],
        password=cred["password"],
        db=cred["db"]
    )

    # Consultar los datos
    query = """
    SELECT f.*, d1.Gender, d2.Occupation, d3.`BMI Category`
    FROM `facts` f
    JOIN `gender` d1 ON f.`id Gender` = d1.`id Gender`
    JOIN `occupation` d2 ON f.`id Occupation` = d2.`id Occupation`
    JOIN `bmi` d3 ON f.`id BMI Category` = d3.`id BMI Category`
    """

    data = pd.read_sql_query(query, connection)
    data.set_index('Person ID', inplace=True)
    cols = ['id Gender', 'id Occupation', 'id BMI Category']
    data.drop(cols, axis=1, inplace=True)

    connection.close()

    #LLamado a la función
    columnas_one_hot = ["BMI Category","Occupation","Gender"]
    data = pd.get_dummies(data, columns=columnas_one_hot)

    # Dividir los datos en características (X) y variable objetivo (y)
    X = data.drop("id Sleep Disorder", axis=1)
    y = data["id Sleep Disorder"]

    # Dividir los datos en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    params = {'n_estimators': 100, 'random_State': 42}
    xgb_clf = XGBClassifier(params)
    xgb_clf.fit(X_train, y_train)

    # Guardar las variables en un archivo
    var_to_pkl = {"X_train_xgb" : X_train, 
                "y_train_xgb" : y_train,
                "X_test_xgb" : X_test, 
                "y_test_xgb" : y_test, 
                "xgb_clf" : xgb_clf, 
                "params_xgb" : params
                }

    for nombre, var in var_to_pkl.items():
        pd.to_pickle(var, f"./pkl/{nombre}.pkl")

