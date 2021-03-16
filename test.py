import datasetreader
import K_neighbors
import numpy as np
import visualize_data
from sklearn.neighbors import KNeighborsClassifier
import K_means
from scipy import cluster
import numpy as np
import matplotlib.pyplot as plot
import K_neighbors
from sklearn.model_selection import train_test_split

X_trainval, X_test, y_trainval, y_test, X, Y = datasetreader.get_dataset()
X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.5, random_state=0)


X_trainval_kmeans, X_train_kmeans, X_val_kmeans, X_test_kmeans, kmeans = K_means.apply_Kmeans(X_trainval, X_train, X_val, X_test, n_components=30)
y_pred, knn_gscv = K_neighbors.kNN_param(X_train_kmeans, y_train)
print(knn_gsvc.score(y_pred, y_train))