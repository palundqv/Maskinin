import datasetreader
import K_neighbors
import MLP
from sklearn.neighbors import KNeighborsClassifier

X_trainval, X_test, y_trainval, y_test, X, Y = datasetreader.get_dataset(
        'Sign-Language-Digits-Dataset-master\Dataset')

clf = KNeighborsClassifier(n_neighbors=9).fit(X_trainval, y_trainval)
knn_predicts = clf.predict(X_test)



MLP.find_components_from_pic(X_trainval, X_test, y_test, knn_predicts)