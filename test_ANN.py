import torch
import pandas as pd
import numpy as np

if __name__ == "__main__":

    # Cargar las variables desde el archivo
    pkl_to_var = ["./pkl/X_train_ann.pkl",
                "./pkl/y_train_ann.pkl",
                "./pkl/X_test_ann.pkl",
                "./pkl/y_test_ann.pkl",
                "./pkl/ann_model.pkl"]
    variables = []

    for pkl in pkl_to_var:
        variables.append(pd.read_pickle(pkl))

    X_train, y_train, X_test, y_test, model = variables

    X_train_tensor = torch.tensor(X_train.values.astype(np.float32), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values.astype(np.float32), dtype=torch.long)
    X_test_tensor  = torch.tensor(X_test.values.astype(np.float32), dtype=torch.float32)
    y_test_tensor  = torch.tensor(y_test.values.astype(np.float32), dtype=torch.long)

    # Realizar predicciones en el conjunto de prueba
    X_train_pred = model(X_train_tensor)
    X_test_pred = model(X_test_tensor)
    predicted_train_labels = torch.argmax(X_train_pred, dim=1)
    predicted_test_labels = torch.argmax(X_test_pred, dim=1)

    # Calcular la precisión en el conjunto de prueba
    accuracy_train = (predicted_train_labels == y_train_tensor).float().mean().item() * 100
    accuracy_test = (predicted_test_labels == y_test_tensor).float().mean().item() * 100
    print("Accuracy on train set: {:.2f}%".format(accuracy_train))
    print("Accuracy on test set: {:.2f}%".format(accuracy_test))

    pd.to_pickle(accuracy_train, "./pkl/accuracy_train_ann.pkl")
    pd.to_pickle(accuracy_test, "./pkl/accuracy_test_ann.pkl")
