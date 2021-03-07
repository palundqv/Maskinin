import datasetreader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import RandomizedPCA
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn import datasets


X_train, X_test, y_train, y_test, X = datasetreader.get_dataset(
    '/Sign-Language-Digits-Dataset-master/Dataset')

nsamples, nx, ny = X.shape
d2_dataset = X.reshape((nsamples,nx*ny))

def vis_pca():
    # Plotta datan i två dimensioner
    pca = PCA(2)  # project from 64 to 2 dimensions
    projected = pca.fit_transform(d2_dataset)

    plt.scatter(projected[:, 0], projected[:, 1],
                c=digits.target, edgecolor='none', alpha=0.5,
                cmap=plt.cm.get_cmap('spectral', 10))
    plt.xlabel('component 1')
    plt.ylabel('component 2')
    plt.colorbar()

def vis_components():
    # visualisera ett antal komponenter
    pca = RandomizedPCA(150)
    pca.fit(X)

    fig, axes = plt.subplots(3, 8, figsize=(9, 4), 
    subplot_kw={'xticks':[], 'yticks':[]},
    gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(pca.components_[i].reshape(62, 47), cmap='bone')

def vis_clusters(X):
    
    K = 10
    kmeans = KMeans(n_clusters=K)

    #numClasses = 5
    #numObservations = numClasses*100
    #cluster_std = 0.5

    #X,y = datasets.make_blobs(numObservations,centers=numClasses,cluster_std=cluster_std)

    kmeans.fit(X)

    params = {'legend.fontsize': 'x-large','figure.figsize': (12, 6),
            'axes.labelsize': 'x-large','axes.titlesize':'x-large',
            'xtick.labelsize':'x-large','ytick.labelsize':'x-large'}
    plt.rcParams.update(params)
    fig, ax = plt.subplots()

    plt.subplot(121); plt.scatter(X[:,0],X[:,1],alpha=0.5,c=y); plt.grid()
    plt.subplot(122);plt.scatter(X[:,0],X[:,1],alpha=0.5,c=kmeans.labels_)
    plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],
            s=50,c='black',marker='x')
    plt.grid(); plt.show()

if __name__ == '__main__':

    vis_clusters(X)

