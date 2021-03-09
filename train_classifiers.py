import matplotlib.pyplot as plt
import datasetreader
import numpy as np
import features
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import features
import sys
import numpy
numpy.set_printoptions(threshold=sys.maxsize)


# KNN Classifier ##############################################################################################################
def apply_KNeighborsClassifier(X_train_pca, X_test_pca, y_train):
    # https://stackoverflow.com/questions/59830510/defining-distance-parameter-v-in-knn-crossval-grid-search-seuclidean-mahalano
    
    knn = KNeighborsClassifier()
    
    grid_params = [
        {'n_neighbors': np.arange(1, 51), 'metric': ['euclidean', 'minkowski', 'manhattan']},
        {'n_neighbors': np.arange(1, 51), 'metric': ['mahalanobis', 'seuclidean'],
        'metric_params': [{'V': np.cov(X_train_pca, rowvar=False)}]}]

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

def apply_MLP_classifier(X_train_pca, X_test_pca, y_train):
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train_pca, y_train)
    MPL_predicts = clf.predict(X_test_pca)
    return MPL_predicts

def find_best_interations():
    pass


# CNN classifier #######################################################################################################################

# def apply cnn classifier

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, X, Y = datasetreader.get_dataset(
        'Sign-Language-Digits-Dataset-master\Dataset')

    n_components = 10

    X_train_pca, X_test_pca = features.apply_PCA(X_train, X_test, n_components)

    y_pred, knn = apply_KNeighborsClassifier(X_train_pca, X_test_pca, y_train)

    print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
    print(knn.best_params_)
def apply_cnn_classifier():
    pass


########################################################################################################################################

if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X, Y = datasetreader.get_dataset(
        '/Sign-Language-Digits-Dataset-master/Dataset')
    X_train_pca, X_test_pca = features.apply_PCA(X_train, X_test, 30)
    preds = apply_MLP_classifier(X_train_pca, X_test_pca, y_train)
    print(preds)
    print((preds==y_test).sum() / len(y_test))








