from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier

def apply_KNeighborsClassifier(X_train, X_test, y_train):
    # https://stackoverflow.com/questions/59830510/defining-distance-parameter-v-in-knn-crossval-grid-search-seuclidean-mahalano
    # Gridsearch hittar bästa kombinationen av hyperparametrarna k och distance av kNN och predictar på dem.

    knn = KNeighborsClassifier()
    
    grid_params = [
        {'n_neighbors': np.arange(1, 51), 'metric': ['euclidean', 'minkowski', 'manhattan', 'chebyshev', 'hamming']}]

    knn_gscv = GridSearchCV(estimator=knn, param_grid=grid_params, cv=5)
    knn_gscv.fit(X_train, y_train)

    y_pred = knn_gscv.predict(X_test)
    
    return y_pred, knn_gscv

    

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

