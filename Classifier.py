import tensorflow as tf
import keras as keras
from keras import layers
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from DataAccess import DataAccess as d_a

 
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

    def nn_classification(self, learningRate=0.01, epochs=30):      
        normalizer = tf.keras.layers.Normalization(axis=-1)
        normalizer.adapt(np.array(self.train_features))

        classification_model = keras.models.Sequential([
            normalizer,
            layers.InputLayer(input_shape=(10)),
            layers.Dense(64, activation='relu'),
            layers.Dense(32, activation='relu'),
            layers.Dense(1, activation='sigmoid')
        ])

        classification_model.compile(
            optimizer=keras.optimizers.Adamax(learning_rate=learningRate),
            loss="binary_crossentropy",
            metrics=['accuracy'])

        history = classification_model.fit(self.train_features, self.train_labels, epochs=epochs)

        loss,acc = classification_model.evaluate(self.test_features, self.test_labels)
        print(f"\nloss: {loss} / acc: {acc}")
    
        return classification_model
     

    # def prediction(self, model, data):
    #     result = model.predict(data)
    #     return result
    
    


if __name__ == "__main__":
    # classif = Classify()
    # print(tf.version.VERSION)
    print(keras.__version__)