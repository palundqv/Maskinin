import matplotlib.pyplot as plt
import datasetreader
from sklearn.cluster import KMeans
import numpy as np
import PCA
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import visualize_data


def plotOptimalElbow(X_train, max_interation):
    # https://www.scikit-yb.org/en/latest/api/cluster/elbow.html#:~:text=The%20elbow%20method%20runs%20k,point%20to%20its%20assigned%20center.
    # med PCA.apply_PCA(X_train, X_test, 0.60) blev resultatet k = 32
    model = KElbowVisualizer(KMeans(), k=max_interation)
    model.fit(X_train)
    model.show()


def evaluate_print(y_train, kmeans):
    # printar ut tabell med precision, recall, accuracy och f-measure
    target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']

    reference_labels = retrieve_info(kmeans, kmeans.labels_, y_train)
    number_labels = np.random.rand(len(kmeans.labels_))
    for i in range(len(kmeans.labels_)):
        number_labels[i] = reference_labels[kmeans.labels_[i]]
    print(number_labels.shape)

    print(classification_report(y_train, number_labels, target_names=target_names))


def apply_Kmeans(X_trainval, X_train, X_val, X_test, n_components=30):
    kmeans = KMeans(n_components, random_state=0).fit(X_train)
    X_trainval_kmeans = kmeans.transform(X_trainval)
    X_train_kmeans = kmeans.transform(X_train)
    X_val_kmeans = kmeans.transform(X_val)
    X_test_kmeans = kmeans.transform(X_test)

    return X_trainval_kmeans, X_train_kmeans, X_val_kmeans, X_test_kmeans, kmeans


def retrieve_info(kmeans, cluster_labels,y_train):
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
 

def calculate_accuracy_kmeans(kmeans, y_train):

    reference_labels = retrieve_info(kmeans, kmeans.labels_, y_train)
    number_labels = np.random.rand(len(kmeans.labels_))
    for i in range(len(kmeans.labels_)):
        number_labels[i] = reference_labels[kmeans.labels_[i]]

    return 'Accuracy score : {}'.format(accuracy_score(number_labels, y_train))


def plot_best_accuracy_score_kmeans(X_train, X_test, max_interation):
    cluster_scores = []
    for i in range(1, max_interation):
        kmeans = KMeans(n_clusters=i, random_state=0)
        kmeans.fit(X_train)
        reference_labels = retrieve_info(kmeans, kmeans.labels_, y_train)
        number_labels = np.random.rand(len(kmeans.labels_))
        for i in range(len(kmeans.labels_)):
            number_labels[i] = reference_labels[kmeans.labels_[i]]
        cluster_scores.append(accuracy_score(number_labels, y_train))

    plt.figure(figsize=(10, 8))
    plt.plot(range(1, max_interation), cluster_scores, marker='1', linestyle='--')
    plt.show()


if __name__ == "__main__":
    X_trainval, X_test, y_trainval, y_test, X, Y = datasetreader.get_dataset()

    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.5, random_state=0)

    #X_trainval_kmeans, X_train_kmeans, X_val_kmeans, X_test_kmeans, kmeans = apply_Kmeans(X_trainval, X_train, X_val, X_test, n_components=30)
    #X_trainval_pca, X_train_pca, X_val_pca, X_test_pca, pca = PCA.apply_PCA(X_trainval_kmeans, X_train_kmeans, X_val_kmeans, X_test_kmeans)
    #visualize_data.vis_PCA_components(X_trainval_kmeans, 15)

    #plot_best_accuracy_score_kmeans(X_train_pca, X_test_pca, 500)
    
    X_trainval_kmeans, X_train_kmeans, X_val_kmeans, X_test_kmeans, kmeans = apply_Kmeans(X_trainval, X_train, X_val, X_test, n_components=30)
    print(kmeans.predict(X_test).shape)
    evaluate_print(y_train, kmeans)