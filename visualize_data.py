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
from sklearn import manifold
from sklearn.neighbors import KNeighborsClassifier

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
    data=df,
    legend=False,
    alpha=0.3)
    colors = ["#476A2A", "#7851B8", "#BD3430", "#4A2D4E", "#875525", "#A83683", "#4E655E", "#853541", "#3A3120", "#535D8E"]
    for i in range(len(X)):
        # actually plot the digits as text instead of using scatter
        plt.text(tsne_results[i, 0], tsne_results[i, 1], str(y[i]), color = colors[y[i]], fontdict={'weight': 'bold', 'size': 9})
    plt.show()


def vis_classifiers_confusion(X_train, X_test, y_train, y_test):
    X_train_pca , X_test_pca = apply_PCA(X_train, X_test)
    mlp_p = MLP.apply_MLP_classifier(X_train_pca, X_test_pca, y_train)
    knn_p = K_neighbors.apply_knn_classifier(X_train_pca, X_test_pca, y_train)

    vis_confusion_matrix(mlp_p, y_test)
    vis_confusion_matrix(knn_p, y_test)


def print_vis_confusion_matrix(y_pred, y_true):

    confusion = confusion_matrix(y_true, y_pred)

    labels = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    print_cm(confusion, labels)

def print_cm(cm, labels, hide_zeroes=False, hide_diagonal=False, hide_threshold=None):
    """pretty print for confusion matrixes"""
    """taken from https://gist.github.com/zachguo/10296432"""
    columnwidth = max([len(x) for x in labels] + [5])  # 5 is value length
    empty_cell = " " * columnwidth
    
    # Begin CHANGES
    fst_empty_cell = (columnwidth-3)//2 * " " + "t\p" + (columnwidth-3)//2 * " "
    
    if len(fst_empty_cell) < len(empty_cell):
        fst_empty_cell = " " * (len(empty_cell) - len(fst_empty_cell)) + fst_empty_cell
    # Print header
    print("    " + fst_empty_cell, end=" ")
    # End CHANGES
    
    for label in labels:
        print("%{0}s".format(columnwidth) % label, end=" ")
        
    print()
    # Print rows
    for i, label1 in enumerate(labels):
        print("    %{0}s".format(columnwidth) % label1, end=" ")
        for j in range(len(labels)):
            cell = "%{0}.2f".format(columnwidth) % cm[i, j]
            if hide_zeroes:
                cell = cell if float(cm[i, j]) != 0 else empty_cell
            if hide_diagonal:
                cell = cell if i != j else empty_cell
            if hide_threshold:
                cell = cell if cm[i, j] > hide_threshold else empty_cell
            print(cell, end=" ")
        print()


def conf_accuracy(conf_matrix):
   diagonal_sum = 0
   for i in range(len(conf_matrix[:])):
        diagonal_sum += conf_matrix[i][i]
   return diagonal_sum / summed_elements


def vis_MDS_cluster():
    # Stora delar taget från Lecture 09 MDS.ipynb
    embedding = manifold.MDS(2, max_iter=100, n_init=1)
    Xprime = embedding.fit_transform(X)

    params = {'legend.fontsize': 'x-large','figure.figsize': (10, 10),
             'axes.labelsize': 'x-large','xtick.labelsize':'x-large',
             'ytick.labelsize':'x-large'}
    plt.rcParams.update(params)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    labels_temp =[]
    for i in range(len(y)):
        labels_temp.append(y[i])
    hand = ax.scatter(Xprime[:,0], Xprime[:,1], s=15 ,alpha=0.5, c=labels_temp, cmap=plt.cm.brg)
    plt.legend(handles=hand.legend_elements()[0],labels=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
    plt.xlabel('Embedding Dimension 1'); plt.ylabel('Embedding Dimension 2')
    ax.grid(); plt.show()
    

if __name__ == '__main__':
    X_train, X_test, y_train, y_test, X, Y = datasetreader.get_dataset()
    KNN = KNeighborsClassifier(10).fit(X_train, y_train)
    y_pred = KNN.predict(X_test)
    print_vis_confusion_matrix(y_pred, y_test)
    #vis_kmeans_components(X, 15)

    '''
    #X_pca = PCA(n_components=10).fit_transform(X)
    X_kmeans = KMeans(n_clusters=35).fit_transform(X)
    vis_tSNE(X, Y)
    

    #vis_pca(X, Y)

    #vis_PCA_components(X,0.95)
    pca = PCA(0.95)
    principal_components =  pca.fit_transform(X)

    #vis_clusters(principal_components, Y)
    vis_tSNE(principal_components,Y)   
    '''