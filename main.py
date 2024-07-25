import Detectors as detectors
import DataAccess as data_access
import Classifier as classifier
import numpy as np
import os


def rf_predicter(repeat):
    d_a = data_access.DataAccess()
    X_train, X_test, y_train, y_test= d_a.read_with_parameters()

    _classifier = classifier.Classify(X_train, X_test, y_train, y_test)
    model = _classifier.randomForest_classification()

    for i in range(repeat):
        lms = detectors.calculate_positions()
        lms = np.array(lms)
        lms = np.reshape(lms, (1,-1))
        res = model.predict(lms)
        print(res)

    return




def predicter(repeat):
    d_a = data_access.DataAccess()
    X_train, X_test, y_train, y_test= d_a.read_with_parameters()

    _classifier = classifier.Classify(X_train, X_test, y_train, y_test)
    model = _classifier.nn_classification()
    for i in range(repeat):
        succ, lms = detectors.calculate_positions()
        if succ:
            input_data = np.array([lms])
            prediction = model.predict(input_data)
            print(prediction)
            threshold = 0.5
            if prediction - threshold > 0:
                print(f"You were biting your nail!")
            else:
                print(f"You were not biting your nail") 
        else:
            print("No faces detected!")        
def trainer():
    d_a = data_access.DataAccess()
    train_features, train_labels, validation_features, validation_labels, test_features, test_labels = d_a.read_with_parameters()

    classification_model = classifier.Classify(train_features, train_labels, validation_features, validation_labels, test_features, test_labels)

    model = classification_model.classification()
    
    

if __name__ == "__main__":
    # rf_predicter(5)
    predicter(1)



    

