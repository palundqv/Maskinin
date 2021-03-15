from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
import matplotlib.pyplot as plt
import PCA
import datasetreader
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap


def kNN_param(X_train, X_val, y_train):
    # https://stackoverflow.com/questions/59830510/defining-distance-parameter-v-in-knn-crossval-grid-search-seuclidean-mahalano
    # Gridsearch hittar bästa kombinationen av hyperparametrarna k och distance av kNN och predictar på dem.

    knn = KNeighborsClassifier()
    
    grid_params = [
        {'n_neighbors': np.arange(1, 51), 'metric': ['euclidean', 'minkowski', 'manhattan', 'chebyshev', 'hamming']}]

    knn_gscv = GridSearchCV(estimator=knn, param_grid=grid_params, cv=5)
    knn_gscv.fit(X_train, y_train)

    y_pred = knn_gscv.predict(X_val)
    
    return y_pred, knn_gscv


def apply_knn_classifier(X_trainval_pca, X_test_pca, y_trainval, neighbors, distance):
    clf = KNeighborsClassifier(n_neighbors=neighbors, metric=distance).fit(X_trainval_pca, y_trainval)
    knn_predicts = clf.predict(X_test_pca)
    return knn_predicts, clf


def Kneighbors_plotter(n_neighbors, X_train_pca, y_train, X_test_pca, y_test):
    # Plottar accuracy mot antalet k i kNN.
    test_accuracy = []
    training_accuracy = []
    neighbors = np.arange(1, n_neighbors, 1)
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


def evaluate(y_test, y_pred):
    # printar ut tabell med precision, recall, accuracy och f-measure
    target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    print(classification_report(y_test, y_pred,target_names=target_names))


if __name__ == '__main__':
    X_trainval, X_test, y_trainval, y_test, X, Y = datasetreader.get_dataset(
        'Sign-Language-Digits-Dataset-master\Dataset')

    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.5, random_state=0)

    #lista = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    #for i in lista:
    X_trainval_pca, X_train_pca, X_val_pca, X_test_pca, pca = PCA.apply_PCA(X_trainval, X_train, X_val, X_test, i)


    y_pred, knn = kNN_param(X_train_pca, X_val_pca, y_train)
    print("Test set score: {:.2f}".format(np.mean(y_pred == y_val)))
    print(knn.best_params_)

    y_predict, clf = apply_knn_classifier(X_trainval_pca, X_test_pca, y_trainval, 
    knn.best_params_['n_neighbors'], knn.best_params_['metric'])
    evaluate(y_test, y_predict)



    