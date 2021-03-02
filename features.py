import matplotlib.pyplot as plt
import datasetreader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier

X_train, X_test, y_train, y_test = datasetreader.get_dataset(
    '/Users/per/Documents/Dev/python/memory/Maskinin_FinalProject/Sign-Language-Digits-Dataset-master/Dataset')


target_names = np.array((0, 1, 2, 3, 4, 5, 6, 7, 8, 9))
#_, h, w = lfw_dataset.images.shape
# target_names = lfw_dataset.target_names
nsamples, nx, ny = X_train.shape
d2_train_dataset = X_train.reshape((nsamples,nx*ny))

nsamples, nx, ny = X_test.shape
d2_test_dataset = X_test.reshape((nsamples,nx*ny))

# Compute a PCA
n_components = 100
pca = PCA(n_components=n_components, whiten=True).fit(d2_train_dataset)

# apply PCA transformation
X_train_pca = pca.transform(d2_train_dataset)
X_test_pca = pca.transform(d2_test_dataset)

# train a neural network
print("Fitting the classifier to the training set")
clf = MLPClassifier(hidden_layer_sizes=(1024,), batch_size=256, verbose=True, early_stopping=True).fit(X_train_pca, y_train)


y_pred = clf.predict(X_test_pca)
# print(classification_report(y_test, y_pred, target_names=target_names))

def plot_gallery(images, titles, h, w, rows=3, cols=4):
    plt.figure()
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i][:].reshape((h, w)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())
 
def titles(y_pred, y_test, target_names):
    for i in range(y_pred.shape[0]):
        pred_name = y_pred[i]
        true_name = y_test[i]
        yield 'predicted: {0}\ntrue: {1}'.format(pred_name, true_name)
 
prediction_titles = list(titles(y_pred, y_test, target_names))
plot_gallery(X_test, prediction_titles, 64, 64)
plt.show()