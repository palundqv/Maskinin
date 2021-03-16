import datasetreader
import PCA
import K_neighbors
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd 
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix

def vis_kmeans_components(X, n_components):
    # varje cluster kan ses som en component och fungerar då som en PCA
    # Här plottas de första 3 x 5 components
    kmeans = KMeans(n_components, random_state=0).fit(X)
    fig, axes = plt.subplots(3, 5, figsize=(10, 6), subplot_kw={'xticks': (), 'yticks': ()})
    for i, (component, ax) in enumerate(zip(kmeans.cluster_centers_, axes.ravel())): 
        ax.imshow(component.reshape(64, 64))
        ax.set_title("{}. component".format(i+1))
    plt.show()

def vis_pca(X, y, n_components=2):
    # https://www.kaggle.com/vinayjaju/t-sne-visualization-sign-language-digit-dataset 
    #Plottar de två första principal components som tagits fram av PCA.
    pca = PCA(n_components)
    principal_components =  pca.fit_transform(X)
    pc_df = pd.DataFrame(data = principal_components, columns=['principal_component1', 'principal_component2'])

    labels_temp =[]
    for i in range(len(y)):
        labels_temp.append(y[i])
    
    pc_df['labels'] = labels_temp
    pc_df['Labels'] = (pc_df['labels']).astype(str)

    sns.scatterplot(pc_df['principal_component1'], pc_df['principal_component2'], hue=pc_df['Labels'], alpha=0.7)
    plt.show()


def vis_PCA_components(X, n_components):
    # Boken s. 152
    # visualisera ett antal komponenter i bilderna
    pca = PCA(n_components, whiten=True, random_state=0)
    pca.fit(X)

    fix, axes = plt.subplots(3, 5, figsize=(15, 12),
    subplot_kw={'xticks': (), 'yticks': ()})
    for i, (component, ax) in enumerate( zip(pca.components_, axes.ravel()) ):
        ax.imshow(component.reshape(64, 64))
        ax.set_title("{}. component".format((i + 1)))
    plt.show()


def vis_clusters(X, y):
    # Lecture 9 Kmeans
    # Visualiserar datan med k-clusters i 2 dim med k antal center.

    K = 10
    kmeans = KMeans(n_clusters=K)

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
    plt.grid()
    plt.show()


def vis_tSNE(X, y):
    # https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    # Visualiserar datan med tSNE.
    feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]
    df = pd.DataFrame(data = X, columns=feat_cols)
    df['y'] = y

    data_subset = df[feat_cols].values
    
    tsne = TSNE(n_components=2, verbose=1, perplexity=50, n_iter=1000, learning_rate=1000)
    tsne_results = tsne.fit_transform(data_subset)

    df['tsne-2d-one'] = tsne_results[:,0]
    df['tsne-2d-two'] = tsne_results[:,1]
    plt.figure(figsize=(16,10))
    sns.scatterplot(
    x="tsne-2d-one", y="tsne-2d-two",
    hue="y",
    palette=sns.color_palette("hls", 10),
    data=df,
    legend="full",
    alpha=0.3)
    plt.show()

def vis_classifiers_confusion(X_train, X_test, y_train, y_test):
    X_train_pca , X_test_pca = apply_PCA(X_train, X_test)
    mlp_p = MLP.apply_MLP_classifier(X_train_pca, X_test_pca, y_train)
    knn_p = K_neighbors.apply_knn_classifier(X_train_pca, X_test_pca, y_train)

    vis_confusion_matrix(mlp_p, y_test)
    vis_confusion_matrix(knn_p, y_test)

def vis_confusion_matrix(y_pred, y_true):
    lables = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] # fick ej att fungera
    return confusion_matrix(y_true, y_pred, labels=lables)

def conf_accuracy(conf_matrix):
   diagonal_sum = 0
   for i in range(len(conf_matrix[:])):
        diagonal_sum += conf_matrix[i][i]
   return diagonal_sum / summed_elements

    

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, X, Y = datasetreader.get_dataset()

    #vis_pca(X, Y)

    #vis_PCA_components(X,0.95)
    pca = PCA(0.95)
    principal_components =  pca.fit_transform(X)

    #vis_clusters(principal_components, Y)
    vis_tSNE(principal_components,Y)


