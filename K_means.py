import matplotlib.pyplot as plt
import datasetreader
from sklearn.cluster import KMeans
import numpy as np
import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import accuracy_score

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


def evaluate_print(y_test, y_pred):
    # printar ut tabell med precision, recall, accuracy och f-measure
    target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    print(classification_report(y_test, y_pred, target_names=target_names))


def apply_Kmeans(X_train, X_test):
    kmeans = KMeans(n_clusters=10, random_state=0).fit(X_train)
    y_pred = kmeans.predict(X_test)
    return y_pred, kmeans



def retrieve_info(cluster_labels,y_train):
    '''
    Associates most probable label with each cluster in KMeans model
    returns: dictionary of clusters assigned to each label
    '''
    # Initializing
    reference_labels = {}
    # For loop to run through each label of cluster label
    for i in range(len(np.unique(kmeans.labels_))):
        index = np.where(cluster_labels == i,1,0)
        num = np.bincount(y_train[index==1]).argmax()
        reference_labels[i] = num
    return reference_labels


def calculate_metrics(model,output):
 print('Number of clusters is {}'.format(model.n_clusters))
 print('Inertia : {}'.format(model.inertia_))
 print('Homogeneity :       {}'.format(metrics.homogeneity_score(output,model.labels_)))


def wierd_calculator_I_dont_understand():
    cluster_number = [10,16,36,64,144,256]
    for i in cluster_number:
        total_clusters = len(np.unique(y_test))
    
        #   Initialize the K-Means model
        kmeans = KMeans(n_clusters = i)
        # Fitting the model to training set
        kmeans.fit(X_train)
        # Calculating the metrics
 
        calculate_metrics(kmeans, y_train)
        # Calculating reference_labels
        reference_labels = retrieve_info(kmeans.labels_, y_train)
        # ‘number_labels’ is a list which denotes the number displayed in image
        number_labels = np.random.rand(len(kmeans.labels_))

    for i in range(len(kmeans.labels_)):
        number_labels[i] = reference_labels[kmeans.labels_[i]]
 
def calculate_accuracy_kmeans(n_clusters, kmeans, y_train):

    reference_labels = retrieve_info(kmeans.labels_, y_train)
    number_labels = np.random.rand(len(kmeans.labels_))
    for i in range(len(kmeans.labels_)):
        number_labels[i] = reference_labels[kmeans.labels_[i]]

    return 'Accuracy score : {}'.format(accuracy_score(number_labels, y_train))



if __name__ == "__main__":
    X_train, X_test, y_train, y_test, X, Y = datasetreader.get_dataset()
    X_train_pca, X_test_pca, pca = PCA.apply_PCA(X_train, X_test, 0.60)
    y_pred, kmeans = apply_Kmeans(X_train_pca, X_test_pca)
    print(calculate_accuracy_kmeans(34, kmeans, y_train))


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