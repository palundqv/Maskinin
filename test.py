'''
import datasetreader
import K_neighbors
import numpy as np

X_train, X_test, y_train, y_test, new_X, new_Y = datasetreader.get_dataset()

ypred, knn = apply_KNeighborsClassifier(X_train, X_test, y_train):

rho = np.corrcoef(x)
rho.style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)
import visualize_data
from sklearn.neighbors import KNeighborsClassifier
'''
#X_train, X_test, y_train, y_test, X, Y = datasetreader.get_dataset()
#X_train_pca, X_test_pca = features.apply_PCA(X_train, X_test)
k = 1
best_score = '0.42'
print("best score: ", best_score, 'best component', k)
print(' ')
print("best score: ", best_score, 'best component', k)
'''
knn = KNeighborsClassifier(10).fit(X_train, y_train)
print(knn.score(X_test, y_test))
knn_PCA = KNeighborsClassifier(10).fit(X_train_pca, y_train)
print(knn_PCA.score(X_test_pca, y_test))
'''
#visualize_data.vis_PCA_components(X_train, 30)
