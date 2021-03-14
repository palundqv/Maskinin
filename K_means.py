import matplotlib.pyplot as plt
import datasetreader
from sklearn.cluster import KMeans
import numpy as np
import PCA
from yellowbrick.cluster import KElbowVisualizer


def print_cluster_curve(X_train, amount_of_interations):
    # gammal kod, locateOtimalElbow är typ bättre
    cluster_scores = []
    for i in range(1, amount_of_interations):
        kmeans_pca = KMeans(n_clusters=i, random_state=0)
        kmeans_pca.fit(X_train_pca)
        cluster_scores.append(kmeans_pca.inertia_)

    plt.figure(figsize=(10, 8))
    plt.plot(range(1, amount_of_interations), cluster_scores, marker='o', linestyle='--')
    plt.show()

def plotOptimalElbow(X_train):
    # https://www.scikit-yb.org/en/latest/api/cluster/elbow.html#:~:text=The%20elbow%20method%20runs%20k,point%20to%20its%20assigned%20center.
    # med PCA.apply_PCA(X_train, X_test, 0.60) blev resultatet k = 32
    model = KElbowVisualizer(KMeans(), k=200)
    model.fit(X_train)
    model.show()

def showTestScoreKmeans():



if __name__ == '__main__':
    X_train, X_test, y_train, y_test, X, Y = datasetreader.get_dataset()
    X_train_pca, X_test_pca, pca = PCA.apply_PCA(X_train, X_test, 0.60)




'''
print(kmeans2.cluster_centers_.shape)

fig, ax = plt.subplots(2, 5, figsize=(8, 3))
print(kmeans.cluster_centers_.shape)
centers = kmeans.cluster_centers_.reshape(10, 17, 24)

for axi, center in zip(ax.flat, centers):
    axi.set(xticks=[], yticks=[])
    axi.imshow(center, interpolation='nearest')
#plt.show()
'''
'''
for ax, comp_kmeans in zip(axes.T, kmeans.cluster_centers_): 
    print(kmeans.cluster_centers_.shape)
    print(kmeans.labels_.shape)
    ax[0].imshow(comp_kmeans.reshape(10, 64, 64)) 

axes[0, 0].set_ylabel("kmeans") 
'''



'''
fig, axes = plt.subplots(3, 5, figsize=(15, 12), subplot_kw={'xticks': (), 'yticks': ()})
for ax, component in enumerate(kmeans.cluster_centers_[kmeans.labels_]):

    ax.imshow(component.reshape(64, 64), cmap='gray')
    ax.set_title("{}. Cluster_center".format((ax + 1)))
plt.show()
'''