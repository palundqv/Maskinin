print('Loadin...')
import sys
import matplotlib.pyplot as plt
import datasetreader
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier

X_train, X_test, y_train, y_test, X = datasetreader.get_dataset(
    '/Sign-Language-Digits-Dataset-master/Dataset')


target_names = ['9', '0', '7', '6', '1', '8', '4', '3', '2', '5']
nsamples, nx, ny = X_train.shape
d2_train_dataset = X_train.reshape((nsamples,nx*ny))

nsamples, nx, ny = X_test.shape
d2_test_dataset = X_test.reshape((nsamples,nx*ny))

def plot_gallery(images, titles, cols=4):
    rows = cols
    plt.figure()
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(images[i][:].reshape((64, 64)), cmap=plt.cm.gray)
        plt.title(titles[i])
        plt.xticks(())
        plt.yticks(())

    plt.show()
    
def titles(y_pred, y_test, target_names):
    for i in range(y_pred.shape[0]):
        pred_name_ind =  np.where(y_pred[i] == 1)
        true_name_ind = np.where(y_test[i] == 1)
        pred_name = np.array(target_names)[pred_name_ind[0]].astype(int)
        true_name = np.array(target_names)[true_name_ind[0]].astype(int)
        yield 'predicted: {0}\ntrue: {1}'.format(pred_name, true_name)

def Kneighbors_plotter(n_neighbors, X_train_pca, y_train, X_test_pca, y_test):
    test_accuracy = []
    training_accuracy = []
    neighbors = np.arange(1,n_neighbors,1)
    for k in neighbors:
        clf = KNeighborsClassifier(n_neighbors = k).fit(X_train_pca, y_train)
        test_accuracy.append(clf.score(X_test_pca, y_test))
        training_accuracy.append(clf.score(X_train_pca, y_train))

    plt.plot(neighbors, test_accuracy, label="test accuracy")
    plt.plot(neighbors, training_accuracy, label="training accuracy")
    plt.ylabel("Accuracy")
    plt.xlabel("n_neighbors")
    plt.legend()
    plt.grid()
    plt.show()

def find_best_components(max_comp, d2_train_dataset, d2_test_dataset, y_test, X_train, y_train):
    best_score = 0
    best_comp = 0
    for comp in range(1, max_comp):
    
        pca = PCA(n_components=comp, whiten=True).fit(d2_train_dataset)

        X_train_pca = pca.transform(d2_train_dataset)
        X_test_pca = pca.transform(d2_test_dataset)

        clf = KNeighborsClassifier(n_neighbors = 3).fit(X_train_pca, y_train)
        

        score = clf.score(X_test_pca, y_test)
        if score > best_score:
            best_score = score
            best_comp = comp
            #print(best_score)
    return best_score, best_comp

# Computing a PCA
n_components = 30
pca = PCA(n_components=n_components, whiten=True).fit(d2_train_dataset)

# appling PCA transformation
X_train_pca = pca.transform(d2_train_dataset)
X_test_pca = pca.transform(d2_test_dataset)

# appling PCA transformation
clf = KNeighborsClassifier(n_neighbors = 3).fit(X_train_pca, y_train)

# Predicting y
y_pred = clf.predict(X_test_pca)

#print(X_test_pca)

#Koden mellan linjerna är endast för debug och kan tas bort utan att paja något
#########################################################################################
np.set_printoptions(threshold=sys.maxsize)
print(sum(sum(y_test)))
# Bunch of prints testing stuff
print("######################################################")   # https://stackoverflow.com/questions/16929203/python-using-scikit-learn-to-predict-gives-blank-predictions
print("Len of x training data: ", len(X_train_pca))
print("Len of y training data: ", len(y_train))
print("Amount of testdata to predict on: ", len(X_test_pca))
print("Actual predicts: ", sum(sum(y_pred)))
print("Y_train: ", y_train)
#print(y_pred)
#########################################################################################
# Använd funktioner nedan - - - - - - - - - - - - - - - - - 

#print(find_best_components(30, d2_train_dataset, d2_test_dataset, y_test, X_train, y_train))

#Kneighbors_plotter(60, X_train_pca, y_train, X_test_pca, y_test)

#plot_gallery(X_test, list(titles(y_pred, y_test, target_names)), 4)

#print(classification_report(y_test, y_pred, target_names=target_names))

# k-means clustering för att visualisera datan

