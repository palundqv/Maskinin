import matplotlib.pyplot as plt
import datasetreader
import numpy as np
import features
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import features
import sys
import numpy
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
numpy.set_printoptions(threshold=sys.maxsize)


# KNN Classifier ##############################################################################################################
def apply_KNeighborsClassifier(X_train_pca, X_test_pca, y_train):
    # https://stackoverflow.com/questions/59830510/defining-distance-parameter-v-in-knn-crossval-grid-search-seuclidean-mahalano
    # Hittar bästa kombinationen av hyperparametrarna k och distance av kNN och predictar på dem.

    knn = KNeighborsClassifier()
    
    grid_params = [
        {'n_neighbors': np.arange(1, 51), 'metric': ['euclidean', 'minkowski', 'manhattan', 'chebyshev', 'hamming']}]

    knn_gscv = GridSearchCV(estimator=knn, param_grid=grid_params, cv=5)
    knn_gscv.fit(X_train_pca, y_train)

    y_pred = knn_gscv.predict(X_test_pca)
    
    return y_pred, knn_gscv


def Kneighbors_plotter(n_neighbors, X_train_pca, y_train, X_test_pca, y_test):
    # Plottar accuracy mot antalet k i kNN.
    test_accuracy = []
    training_accuracy = []
    neighbors = np.arange(1,n_neighbors,1)
    for k in neighbors:
        clf = KNeighborsClassifier(n_neighbors = k).fit(X_train_pca, y_train)
        test_accuracy.append(clf.score(X_test_pca, y_test))
        training_accuracy.append(clf.score(X_train_pca, y_train))

    plt.plot(neighbors, test_accuracy, label="test accuracy")
    plt.plot(neighbors, training_accuracy, label="training accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.grid()
    plt.show()

# MLP classifier #######################################################################################################################

def apply_MLP_classifier(X_train_pca, X_test_pca, y_train):
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train_pca, y_train)
    MPL_predicts = clf.predict(X_test_pca)
    return MPL_predicts

def MLP_param(X_val_pca, y_val, X_test_pca):
    MLP = MLPClassifier(max_iter=70)
    grid_params = [{
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,100,100), (100,50,100)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive']}]
    MLP_gscv = GridSearchCV(estimator=MLP, param_grid=grid_params, n_jobs=-1, cv=5)
    
    MLP_gscv.fit(X_train_pca, y_train)
    y_pred = MLP_gscv.predict(X_test_pca)

    return y_pred, MLP_gscv

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



