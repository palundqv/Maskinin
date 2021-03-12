import matplotlib.pyplot as plt
import datasetreader
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import train_classifiers
from sklearn.model_selection import StratifiedKFold, cross_val_score

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


def find_best_components(max_comp, d2_train_dataset, d2_test_dataset, y_test, X_train, y_train):
    # predictar med kNN med olika antal pca komponenter.
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


def apply_PCA(X_train, X_test, n_components=30):
    # Computing a PCA
    pca = PCA(n_components=n_components, whiten=True).fit(X_train)
    # appling PCA transformation
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca


def vis_num_pca(X):
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
    # Plottar varians i datan mot antalet komponenter hos PCA.
    pca = PCA(1000)
    pca.fit(X)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()



if __name__ == '__main__':
    X_train, X_test, y_train, y_test, X, Y = datasetreader.get_dataset(
        '/Sign-Language-Digits-Dataset-master/Dataset')
    
    n_components = 2
    n_neighbors = 5

    #X_train_pca, X_test_pca = apply_PCA(X_train, X_test, n_components)

    #y_pred, knn = apply_KNeighborsClassifier(X_train_pca, X_test_pca, y_train, n_neighbors)
    #knn.score(X_test, y_test)
    #print("Test set score: {:.2f}".format(np.mean(y_pred == y_test)))
    #plot_gallery(X_test, titles(y_pred, y_test), 5)

    #vis_num_pca(X)