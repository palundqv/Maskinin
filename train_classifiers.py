import matplotlib.pyplot as plt
import datasetreader
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV



# KNN Classifier ##############################################################################################################
def apply_KNeighborsClassifier(X_train_pca, X_test_pca, y_train, n_neighbors):
    # https://stackoverflow.com/questions/59830510/defining-distance-parameter-v-in-knn-crossval-grid-search-seuclidean-mahalano
    
    knn = KNeighborsClassifier()
    
    grid_params = [
        {'n_neighbors': np.arange(1, 51), 'metric': ['euclidean', 'minkowski']},
        {'n_neighbors': np.arange(1, 51), 'metric': ['mahalanobis', 'seuclidean'],
        'metric_params': [{'V': np.cov(X_train_pca)}]}]

    knn_gscv = GridSearchCV(estimator=knn, param_grid=grid_params[0], cv=5)

    knn_gscv.fit(X_train_pca, y_train)
    
    
    # appling PCA transformation
    # clf = KNeighborsClassifier(n_neighbors).fit(X_train_pca, y_train)
    y_pred = knn_gscv.predict(X_test_pca)
    
    return y_pred, knn_gscv


def Kneighbors_plotter(n_neighbors, X_train_pca, y_train, X_test_pca, y_test):
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

# def apply mlp classifier

# CNN classifier #######################################################################################################################

# def apply cnn classifier