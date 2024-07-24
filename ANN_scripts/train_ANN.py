import torch
import torch.nn as nn
import pandas as pd
import numpy as np
import pymysql
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from NeuralNetwork import NeuralNetwork
import os


class ModelTrainer:
    def __init__(self, model, loss_function, optimizer):
        self.model = model
        self.loss_function = loss_function
        self.optimizer = optimizer
        self.losses = []
        self.accuracies = []
    
    def train(self, X_train_tensor, y_train_tensor, num_epochs):
        for epoch in range(num_epochs):
            # Forward pass
            outputs = self.model(X_train_tensor)
            loss = self.loss_function(outputs, y_train_tensor)
            
            # Calculating accuracy
            predicted_labels = torch.argmax(outputs, dim=1)
            accuracy = 100 * (predicted_labels == y_train_tensor).float().mean()

            # Backward pass y optimizacion
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Saving the loss and accuracy per epoch
            self.losses.append(loss.item())
            self.accuracies.append(accuracy)

            # Showing loss and accuracy every 10 epochs
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

            # Validating loss thresholder
            if target_loss is not None and loss.item() <= target_loss:
                print(f"Desired loss of {target_loss} reached. Stopping training.")
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
                break

            # Validating accuracy thresholder
            if target_accuracy is not None and accuracy.item() >= target_accuracy:
                print(f"Desired accuracy of {target_accuracy}% reached. Stopping training.")
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
                break
        
    def plot_loss_accuracy(self):
        plt.figure(figsize=(10, 4))
        
        # Adding losses
        plt.subplot(1, 2, 1)
        plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        
        # Adding accuracies
        plt.subplot(1, 2, 2)
        plt.plot(self.accuracies)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training Accuracy')
        
        plt.tight_layout()
        plt.savefig('loss_accuracy_plot.png')

if __name__ == "__main__":

    # Create a connection to the database
    """connection = pymysql.connect(
        host=os.environ['DB_HOST'],
        user=os.environ['DB_USER'],
        password=os.environ['DB_PASSWORD'],
        db=os.environ['DB_NAME']
    )"""
    connection = pymysql.connect(
        host='localhost',
        user='root',
        password='root',
        db='sleep'
    )

    # Extracting data from db
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
    # Closing connection
    connection.close()

    columnas_one_hot = ["BMI_Category","Occupation","Gender"]
    data = pd.get_dummies(data, columns=columnas_one_hot)

    # Spliting data into features (X) and target variable (y)
    X = data.drop("ID_Sleep_Disorder", axis=1)
    y = data["ID_Sleep_Disorder"]

    # Generating trainig and testing data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

    # Showing shapes
    print("X_train shape:", X_train.shape)
    print("y_train shape:", y_train.shape)
    print("X_test shape:", X_test.shape)
    print("y_test shape:", y_test.shape)

    X_train_tensor = torch.tensor(X_train.values.astype(np.float32), dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train.values.astype(np.float32), dtype=torch.long)

    input_size = 25
    hidden_size = 128
    output_size = 3
    target_accuracy = 95.0  
    target_loss = 0.5       
    model = NeuralNetwork(input_size, hidden_size, output_size)
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    # Inicializing ModelTrainer object and training
    trainer = ModelTrainer(model, loss_function, optimizer)
    trainer.train(X_train_tensor, y_train_tensor, num_epochs=3000)
    trainer.plot_loss_accuracy()

    # Saving variables in pkl files
    dict_var_to_pkl = {"X_train_ann" : X_train, 
                "y_train_ann" : y_train,
                "X_test_ann" : X_test, 
                "y_test_ann" : y_test, 
                "ann_model" : model,
                "params_ann" : {
                    "input_size" : input_size,
                    "hidden_size": hidden_size,
                    "output_size": output_size,
                    "optimizer_lr": 0.01,
                    "num_epochs" : 3000
                    } 
                }

    for name, var in dict_var_to_pkl.items():
        pd.to_pickle(var, f"./pkl/{name}.pkl")

