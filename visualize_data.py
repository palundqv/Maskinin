import datasetreader
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import pandas as pd 
import seaborn as sns
#from sklearn import datasets




def vis_pca(X, y):
    pca = PCA(n_components=2)
    principal_components =  pca.fit_transform(X)
    pc_df = pd.DataFrame(data = principal_components, columns=['principal_component1', 'principal_component2'])

    labels_temp =[]
    for i in range(len(y)):
        labels_temp.append(np.argmax(y[i]))
    
    pc_df['labels'] = labels_temp
    pc_df['str_labels'] ='Label_'+(pc_df['labels']).astype(str)

    sns.scatterplot(pc_df['principal_component1'], pc_df['principal_component2'], hue=pc_df['str_labels'])
    plt.show()

def vis_components():
    # visualisera ett antal komponenter
    pca = RandomizedPCA(150)
    pca.fit(X)

    fig, axes = plt.subplots(3, 8, figsize=(9, 4), 
    subplot_kw={'xticks':[], 'yticks':[]},
    gridspec_kw=dict(hspace=0.1, wspace=0.1))
    for i, ax in enumerate(axes.flat):
        ax.imshow(pca.components_[i].reshape(62, 47), cmap='bone')

def vis_clusters(X, y):
    
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
    plt.grid()
    plt.show()

def vis_pearsoncorr(X):
    X_dataframe = pd.DataFrame(X, columns=['0','1','2','3','4','5','6','7','8','9'])
    corr = X_dataframe.corr()
    corr.style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)

if __name__ == '__main__':
    #X_train, X_test, y_train, y_test, X, Y = datasetreader.get_dataset(
    #    'Sign-Language-Digits-Dataset-master\Dataset')

    import os
    for dirname, _, filenames in os.walk('/kaggle/input'):
        for filename in filenames:
            print(os.path.join(dirname, filename))

    X = np.load('/kaggle/input/sign-language-digits-dataset/Sign-language-digits-dataset/X.npy')
    y = np.load('/kaggle/input/sign-language-digits-dataset/Sign-language-digits-dataset/Y.npy')

    #nsamples, nx, ny = X.shape
    #d2_dataset = X.reshape((nsamples,nx*ny))
    
    #Y = np.where(Y == 1)[0].astype(int)

    #vis_clusters(d2_dataset, Y)

    X_df = X.reshape(-1,64*64)

<<<<<<< HEAD
    vis_pca(X_df, y)
=======
    #vis_clusters(X, Y)
    #hej hej
>>>>>>> 071fe2b6cb74ceeeb2f932e5745a134c70c9059c
