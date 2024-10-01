# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '0' 
# import tensorflow as tf
# import keras
# from keras import layers
import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np

from DataAccess import DataAccess as d_a

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


class Classify:
    def __init__(self, X_train, X_test, y_train, y_test):
        self.train_features = X_train
        self.train_labels = y_train
        self.test_features = X_test
        self.test_labels = y_test
        # self.data = data

    def randomForest_classification(self):
        classifier = RandomForestClassifier(n_estimators=100, random_state=42)

        classifier.fit(self.train_features, self.train_labels)

        y_pred = classifier.predict(self.test_features)

        accuracy = accuracy_score(self.test_labels, y_pred)
        print(f"Accuracy : {accuracy}")
        return classifier

    def nn_classification(self, learningRate=0.01, epochs=50):
        # Step 1: Normalize the data using StandardScaler
        scaler = StandardScaler()
        train_features = scaler.fit_transform(self.train_features)
        test_features = scaler.transform(self.test_features)

        # Step 2: Define the model
        class ClassificationModel(nn.Module):
            def __init__(self):
                super(ClassificationModel, self).__init__()
                self.fc1 = nn.Linear(10, 64)
                self.fc2 = nn.Linear(64, 32)
                self.dropout = nn.Dropout(0.001)
                self.fc3 = nn.Linear(32, 1)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = torch.relu(self.fc1(x))
                x = torch.relu(self.fc2(x))
                x = self.dropout(x)
                x = self.sigmoid(self.fc3(x))
                return x

        # Step 3: Initialize model, loss function, and optimizer
        model = ClassificationModel()
        optimizer = optim.Adamax(model.parameters(), lr=learningRate)
        loss_fn = nn.BCELoss()

        # Step 4: Training loop
        train_features_tensor = torch.tensor(train_features, dtype=torch.float32)
        train_labels_tensor = torch.tensor(self.train_labels.values, dtype=torch.float32).unsqueeze(1)  # Convert Series to NumPy array

        for epoch in range(epochs):
            model.train()
            optimizer.zero_grad()

            outputs = model(train_features_tensor)
            loss = loss_fn(outputs, train_labels_tensor)

            loss.backward()
            optimizer.step()

            predictions = (outputs > 0.5).float()
            accuracy = (predictions == train_labels_tensor).float().mean()

            print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item()}, Accuracy: {accuracy.item()}")

        # Step 5: Evaluation
        model.eval()
        with torch.no_grad():
            test_features_tensor = torch.tensor(test_features, dtype=torch.float32)
            test_labels_tensor = torch.tensor(self.test_labels.values, dtype=torch.float32).unsqueeze(1)  # Convert Series to NumPy array

            outputs = model(test_features_tensor)
            predictions = (outputs > 0.5).float()

            accuracy = (predictions == test_labels_tensor).float().mean()
            print(f"\nAccuracy: {accuracy.item()}")

        return model

    

    # def prediction(self, model, data):
    #     result = model.predict(data)
    #     return result
    
    


if __name__ == "__main__":
    data = d_a()  
    X_train, X_test, y_train, y_test = data.read_with_parameters()
    classif = Classify(X_train, X_test, y_train, y_test)
    classif.nn_classification()
    