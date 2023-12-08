# import tensorflow as tf
# from tensorflow import keras
# from keras import layers

# import numpy as np


# class Classify:
#     def __init__(self, train_features, train_labels, validation_features, validation_labels, test_features, test_labels):
#         self.train_features = train_features
#         self.train_labels = train_labels
#         self.validation_features = validation_features
#         self.validation_labels = validation_labels
#         self.test_features = test_features
#         self.test_labels = test_labels

#     def classification(self, units=2, learningRate=0.01, epochs=50, data=None):      
#         normalizer = tf.keras.layers.Normalization(axis=-1)
#         normalizer.adapt(np.array(self.train_features))

#         classification_model = tf.keras.Sequential([
#             normalizer,
#             layers.Flatten(),
#             layers.Dense(units=units, activation=keras.activations.relu),
#         ])

#         classification_model.compile(
#             optimizer=keras.optimizers.Adam(learning_rate=learningRate),
#             loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
#             metrics=['accuracy'])

#         history = classification_model.fit(self.train_features, self.train_labels, epochs=epochs, validation_data=(self.validation_features, self.validation_labels))

#         test_loss, test_acc = classification_model.evaluate(self.test_features, self.test_labels, verbose=2)

#         predictions = classification_model.predict(data)
#         return predictions

#     def prediction(self, data):
#         return self.classification(data=data)
