import torch
import torch.nn as nn
import pandas as pd
import json
import numpy as np
import pymysql
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from NeuralNetwork import NeuralNetwork



with open("./credenciales.json", "r") as archivo:
    cred = json.load(archivo)

# Establecer la conexión a la base de datos
connection = pymysql.connect(
    host=cred["host_docker"],
    user=cred["user"],
    password=cred["password"],
    db=cred["db"]
)

# Consultar los datos de una tabla específica
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
# Cerrar la conexión a la base de datos
connection.close()

# Realizar el preprocesamiento de datos si es necesario
# Por ejemplo, puedes realizar la limpieza de datos, transformaciones, etc.

# Comenzar a entrenar tu modelo de IA utilizando los datos del DataFrame
# Puedes utilizar cualquier biblioteca de machine learning como scikit-learn, TensorFlow, PyTorch, etc.
columnas_one_hot = ["BMI Category","Occupation","Gender"]
data = pd.get_dummies(data, columns=columnas_one_hot)

# Dividir los datos en características (X) y variable objetivo (y)
X = data.drop("id Sleep Disorder", axis=1)
y = data["id Sleep Disorder"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# Imprimir la forma de los conjuntos de entrenamiento y prueba
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print("X_test shape:", X_test.shape)
print("y_test shape:", y_test.shape)

X_train_tensor = torch.tensor(X_train.values.astype(np.float32), dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values.astype(np.float32), dtype=torch.long)

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
            
            # Calculate accuracy
            predicted_labels = torch.argmax(outputs, dim=1)
            accuracy = 100 * (predicted_labels == y_train_tensor).float().mean()

            # Backward pass and optimization
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            # Store loss and accuracy values
            self.losses.append(loss.item())
            self.accuracies.append(accuracy)

            # Print training loss and accuracy
            if (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")

            # Check if desired loss is reached
            if target_loss is not None and loss.item() <= target_loss:
                print(f"Desired loss of {target_loss} reached. Stopping training.")
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
                break

            # Check if desired accuracy is reached
            if target_accuracy is not None and accuracy.item() >= target_accuracy:
                print(f"Desired accuracy of {target_accuracy}% reached. Stopping training.")
                print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.2f}%")
                break
        
    def plot_loss_accuracy(self):
        plt.figure(figsize=(10, 4))
        
        # Plot loss
        plt.subplot(1, 2, 1)
        plt.plot(self.losses)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        
        # Plot accuracy
        plt.subplot(1, 2, 2)
        plt.plot(self.accuracies)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.title('Training Accuracy')
        
        plt.tight_layout()
        plt.savefig('loss_accuracy_plot.png')
        plt.show()

input_size = 25
hidden_size = 128
output_size = 3
target_accuracy = 95.0  # Precisión objetivo
target_loss = 0.5       # Pérdida objetivo
model = NeuralNetwork(input_size, hidden_size, output_size)
loss_function = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Crear el objeto ModelTrainer y entrenar el modelo
trainer = ModelTrainer(model, loss_function, optimizer)
trainer.train(X_train_tensor, y_train_tensor, num_epochs=3000)
trainer.plot_loss_accuracy()

# Guardar las variables en un archivo
var_to_pkl = {"X_train_ann" : X_train, 
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

for nombre, var in var_to_pkl.items():
    pd.to_pickle(var, f"./pkl/{nombre}.pkl")

