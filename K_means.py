import matplotlib.pyplot as plt
import datasetreader
from sklearn.cluster import KMeans
import numpy as np
import PCA

X_train, y_train, X_test, y_test, X, Y = datasetreader.get_dataset()
print(X_train.shape)
X_train_pca, X_test_pca, pca = PCA.apply_PCA(X_train, X_test)

kmeans = KMeans(n_clusters=10, random_state=0)
kmeans.fit(X_train_pca)

image_shape = np.arrray((64, 64))

X_reconstructed_pca = pca.inverse_transform(pca.transform(X_test)) 
X_reconstructed_kmeans = kmeans.cluster_centers_[kmeans.predict(X_test)]

fig, axes = plt.subplots(2, 5, figsize=(8, 8), subplot_kw={'xticks': (), 'yticks': ()})
fig.suptitle("Extracted Components")
for ax, comp_kmeans, comp_pca in zip(axes.T, kmeans.cluster_centers_, pca.components_): 
    ax[0].imshow(comp_kmeans.reshape(image_shape)) 
    ax[1].imshow(comp_pca.reshape(image_shape), cmap='viridis') 


axes[0, 0].set_ylabel("kmeans") 
axes[1, 0].set_ylabel("pca") 


fig, axes = plt.subplots(3, 5, subplot_kw={'xticks': (), 'yticks': ()}, figsize=(8, 8))
fig.suptitle("Reconstructions")
for ax, orig, rec_kmeans, rec_pca in zip(axes.T, X_test, X_reconstructed_kmeans, X_reconstructed_pca):
    ax[0].imshow(orig.reshape(image_shape)) 
    ax[1].imshow(rec_kmeans.reshape(image_shape)) 
    ax[2].imshow(rec_pca.reshape(image_shape))

axes[0, 0].set_ylabel("original") 
axes[1, 0].set_ylabel("kmeans") 
axes[2, 0].set_ylabel("pca") 