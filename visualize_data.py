import datasetreader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA


X_train, X_test, y_train, y_test, X = datasetreader.get_dataset(
    '/Sign-Language-Digits-Dataset-master/Dataset')

nsamples, nx, ny = X.shape
d2_dataset = X.reshape((nsamples,nx*ny))


# Plotta datan i två dimensioner
pca = PCA(2)  # project from 64 to 2 dimensions
projected = pca.fit_transform(d2_dataset)

plt.scatter(projected[:, 0], projected[:, 1],
            c=digits.target, edgecolor='none', alpha=0.5,
            cmap=plt.cm.get_cmap('spectral', 10))
plt.xlabel('component 1')
plt.ylabel('component 2')
plt.colorbar()


# visualisera ett antal komponenter
from sklearn.decomposition import RandomizedPCA
pca = RandomizedPCA(150)
pca.fit(X)

fig, axes = plt.subplots(3, 8, figsize=(9, 4), 
subplot_kw={'xticks':[], 'yticks':[]},
gridspec_kw=dict(hspace=0.1, wspace=0.1))
for i, ax in enumerate(axes.flat):
    ax.imshow(pca.components_[i].reshape(62, 47), cmap='bone')


#visualiserar kluster
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
# build the clustering model
kmeans = KMeans(n_clusters=10)
kmeans.fit(X)

mglearn.discrete_scatter(X[:, 0], X[:, 1], kmeans.labels_, markers='o')
mglearn.discrete_scatter(
kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], [0, 1, 2],
markers='^', markeredgewidth=2)





