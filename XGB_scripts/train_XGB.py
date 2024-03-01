import pandas as pd
import pymysql
from xgboost import XGBClassifier;
from sklearn.model_selection import train_test_split
import os

if __name__ == "__main__":

    # Establecer la conexión a la base de datos
    connection = pymysql.connect(
        host=os.environ['DB_HOST'],
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD'],
        db=os.environ['DB_NAME']
    )

    # Consultar los datos
    query = """
    SELECT f.*, d1.Gender, d2.Occupation, d3.BMI_Category
    FROM facts f
    JOIN gender d1 ON f.ID_Gender = d1.ID_Gender
    JOIN occupation d2 ON f.ID_Occupation = d2.ID_Occupation
    JOIN bmi d3 ON f.ID_BMI_Category = d3.ID_BMI_Category
    """

    data = pd.read_sql_query(query, connection)

    data.set_index('Person_ID', inplace=True)
    cols = ['ID_Gender', 'ID_Occupation', 'ID_BMI_Category']
    data.drop(cols, axis=1, inplace=True)

    connection.close()

    #LLamado a la función
    columnas_one_hot = ["BMI_Category","Occupation","Gender"]
    data = pd.get_dummies(data, columns=columnas_one_hot)

    # Dividir los datos en características (X) y variable objetivo (y)
    X = data.drop("ID_Sleep_Disorder", axis=1)
    y = data["ID_Sleep_Disorder"]

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

