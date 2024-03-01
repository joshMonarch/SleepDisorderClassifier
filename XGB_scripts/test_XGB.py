import pandas as pd
from sklearn.metrics import precision_score, recall_score;
from sklearn.model_selection import cross_val_predict;
import numpy as np

if __name__ == "__main__":

    pkl_to_var = ["./pkl/X_train_xgb.pkl",
                "./pkl/y_train_xgb.pkl",
                "./pkl/X_test_xgb.pkl",
                "./pkl/y_test_xgb.pkl",
                "./pkl/xgb_clf.pkl"]
    variables = []

    for pkl in pkl_to_var:
        variables.append(pd.read_pickle(pkl))

    X_train, y_train, X_test, y_test, xgb_clf = variables

    y_train_pred_xgb = cross_val_predict(xgb_clf, X_train, y_train, cv=5)
    y_test_pred_xgb  = cross_val_predict(xgb_clf, X_test, y_test, cv=5)

    precision_train  = precision_score(y_train, y_train_pred_xgb, average=None)
    recall_train     = recall_score(y_train, y_train_pred_xgb, average=None)
    precision_test   = precision_score(y_test, y_test_pred_xgb, average=None)
    recall_test      = recall_score(y_test, y_test_pred_xgb, average=None)

    print("Precisión entreno:\t", np.round(precision_train, 2), np.mean(precision_train))
    print("Recall entreno:\t\t", np.round(recall_train, 2), np.mean(recall_train))

    print("Precisión testeo:\t", np.round(precision_test, 2), np.mean(precision_test))
    print("Recall testeo:\t\t", np.round(recall_test, 2), np.mean(recall_test))

    metrics_dict = {
        "Train_precision" : np.mean(precision_train),
        "Test_precision" : np.mean(precision_test),
        "Train_recall" : np.mean(recall_train),
        "Test_recall" : np.mean(recall_test)
    }

    for nombre, var in metrics_dict.items():
        pd.to_pickle(var, f"./pkl/{nombre}.pkl")