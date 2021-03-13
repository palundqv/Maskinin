import datasetreader
import numpy as np
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
import train_classifiers
import plot_gallery

def find_best_amount_components_PCA(max_components, n_neighbors,  X_train, X_test, y_train, y_test):
    best_score = 0
    best_comp = 0
    for comp in range(1, max_components):
    
        pca = PCA(n_components=comp, whiten=True).fit(X_train)

        X_train_pca, X_test_pca = apply_PCA(X_train, X_test)

        clf = KNeighborsClassifier(n_neighbors).fit(X_train_pca, y_train)

        score = clf.score(X_test_pca, y_test)
        if score > best_score:
            best_score = score
            best_comp = comp

    return best_score, best_comp

def find_best_amount_neighbors_with_PCA(max_neighbors, max_components, X_train, X_test, y_train, y_test):
    comp = 0
    score = 0
    for k_neighbors in range(1, max_neighbors):
        best_score, best_comp = find_best_amount_components_PCA(max_components, k_neighbors,  X_train, X_test, y_train, y_test)
        print('Best score: (', best_score, ')best component: (', best_comp, ') K neighbor: ', k_neighbors)
        print(' ')
        if score > best_score:
            best_score = score
            best_comp = comp
            best_neighbor = k_neighbors
        
    return best_score, best_neighbor, best_comp

def apply_PCA(X_train, X_test, n_components=30):
    # Computing a PCA
    pca = PCA(n_components=n_components, whiten=True).fit(X_train)
    # appling PCA transformation
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)

    return X_train_pca, X_test_pca

def vis_num_pca(X):
    # https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
    pca = PCA(1000)
    pca.fit(X)

    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()


if __name__ == '__main__':

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test, X, Y = datasetreader.get_dataset()
    
    X_train_train, X_validation, y_train_train, y_validation = train_test_split(X_train, y_train, test_size=0.3, random_state=0)

    print(find_best_amount_components_PCA(10, 10, X_validation, X_train_train, y_validation, y_train_train))
    print(find_best_amount_neighbors_with_PCA(15, 495, X_validation, X_train_train, y_validation, y_train_train))
    print('Done')
