import numpy as np
import train_classifiers
import features
import datasetreader
import visualize_data

X_train, X_test, y_train, y_test, X, Y = datasetreader.get_dataset()
X_train_pca, X_test_pca = features.apply_PCA(X_train, X_test)
print(X_train.shape)
print(X_train_pca.shape)
visualize_data.vis_components(X)