import matplotlib.pyplot as plt
import datasetreader
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV


def plot_gallery(X_test, preds, cols=4):
    rows = cols
    plt.figure()
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(X_test[i][:].reshape((64, 64)), cmap=plt.cm.gray)
        plt.title(preds[i])
        plt.xticks(())
        plt.yticks(())

    plt.show()
    
def titles(y_pred, y_test):
    predicted_names = []
    for i in range(y_pred.shape[0]):
        predicted_names.append('predicted: {0}\ntrue: {1}'.format(y_pred[i], y_test[i]))
    return predicted_names

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

def find_best_components(max_comp, d2_train_dataset, d2_test_dataset, y_test, X_train, y_train):
    best_score = 0
    best_comp = 0
    for comp in range(1, max_comp):
    
        pca = PCA(n_components=comp, whiten=True).fit(d2_train_dataset)

        X_train_pca = pca.transform(d2_train_dataset)
        X_test_pca = pca.transform(d2_test_dataset)

        clf = KNeighborsClassifier(n_neighbors = 3).fit(X_train_pca, y_train)
        

        score = clf.score(X_test_pca, y_test)
        if score > best_score:
            best_score = score
            best_comp = comp
            #print(best_score)
    return best_score, best_comp

def apply_PCA(X_train, X_test, n_components):
    # Computing a PCA
    n_components = 30
    pca = PCA(n_components=n_components, whiten=True).fit(X_train)

    # appling PCA transformation
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_train_pca

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

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, X, Y = datasetreader.get_dataset(
        '/Sign-Language-Digits-Dataset-master/Dataset')
    
    n_components = 5
    n_neighbors = 5

    X_train_pca, X_test_pca = apply_PCA(X_train, X_test, n_components)
    y_pred, knn = apply_KNeighborsClassifier(X_train_pca, X_test_pca, y_train, n_neighbors)
    #knn.score(X_test, y_test)
    #print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
    plot_gallery(X_test, titles(y_pred, y_test), 5)
