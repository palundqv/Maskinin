import matplotlib.pyplot as plt
import datasetreader
import numpy as np
import PCA
import tensorflow as tf
from tensorflow.keras import datasets, layers, models

from sklearn.decomposition import PCA

import PCA
import sys
import numpy
from sklearn.metrics import classification_report
numpy.set_printoptions(threshold=sys.maxsize)


# KNN Classifier ##############################################################################################################

# MLP classifier #######################################################################################################################


########################################################################################################################################


def evaluate(y_test, y_pred):
    # printar ut tabell med precision, recall, accuracy och f-measure
    target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    print(classification_report(y_test, y_pred,target_names=target_names))



if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X, Y = datasetreader.get_dataset(
        '/Sign-Language-Digits-Dataset-master/Dataset')
    X_train_pca, X_test_pca = features.apply_PCA(X_train, X_test, 30)


    #apply_cnn_classifier(X_train, X_test, y_train, y_test)
    """
    preds = apply_MLP_classifier(X_train_pca, X_test_pca, y_train)
    print("MLP success rate = ", (preds==y_test).sum() / len(y_test))


    y_pred, knn = apply_KNeighborsClassifier(X_train_pca, X_test_pca, y_train)
    print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
    print(knn.best_params_)
    """
 
    #Kneighbors_plotter(40, X_train_pca, y_train, X_test_pca, y_test)
    
    y_pred, MLP = MLP_param(X_train_pca, y_train, X_test_pca)
    print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
    print(MLP.best_params_)


