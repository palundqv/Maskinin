import datasetreader
import PCA 
import matplotlib.pyplot as plt
import numpy as np
import visualize_data as vis
import plot_gallery
from sklearn.decomposition import PCA as prins_comp
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split

def apply_MLP_classifier(X_train, X_test, y_train, layers, activation, solver, alpha, learning_rate):
    clf = MLPClassifier(random_state=1, max_iter=300, hidden_layer_sizes=layers,
    activation=activation,solver=solver, alpha=alpha, learning_rate=learning_rate).fit(X_train, y_train)
    MPL_predicts = clf.predict(X_test)
    return MPL_predicts

def MLP_param(X_train_pca, y_train, X_val_pca):
    MLP = MLPClassifier(max_iter=150)
    grid_params = [{
    'hidden_layer_sizes': [(300,300,300)],
    'activation': ['tanh', 'relu', 'logistic'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive']}]
    MLP_gscv = GridSearchCV(estimator=MLP, param_grid=grid_params, n_jobs=-1, cv=5)
    MLP_gscv.fit(X_train_pca, y_train)
    y_pred = MLP_gscv.predict(X_val_pca)
    return y_pred, MLP_gscv


def plot_MPL_params(X_train_pca, X_test_pca, y_train, y_test):
    max_miters = 500
    training_error = []
    validation_error = []
    t = np.linspace(0, max_miters, max_miters)
    for iteration in range(max_miters):
        clf = MLPClassifier(random_state = 1, max_iter=iteration + 1).fit(X_train_pca, y_train)
        t_predicts = clf.predict(X_train_pca)
        v_predicts = clf.predict(X_test_pca)
        training_error.append(1 - ((t_predicts == y_train).sum() / len(y_train)))
        validation_error.append(1 - ((v_predicts == y_test).sum() / len(y_test)))
    plt.plot(t, training_error)
    plt.plot(t, validation_error)
    plt.show()


def evaluate(y_test, y_pred):
    # printar ut tabell med precision, recall, accuracy och f-measure
    target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    print(classification_report(y_test, y_pred, target_names=target_names))
 
def find_components_from_pic(X_train, X_test, y_test, y_predict):
    # Finds the indicies where
    pca = prins_comp(n_components=0.6, whiten=True).fit(X_train)
    X_test_pca = pca.transform(X_test)
    indicies = []
    for i in range(len(y_test)):
        if (y_test[i] == 6 or y_test[i] == 4) and y_test[i] != y_predict[i]:
            indicies.append(i)
    print(X_test_pca[indicies[0]])

    X_inv = pca.inverse_transform(X_test_pca) 
    #reshaping as len of train images of 64x64 dimension
    X_proj_img = np.reshape(X_inv, (len(X_test),64,64)) 
    
    fig = plt.figure(figsize=(6,6)) 
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=0.05, wspace=0.05) # plot the faces, each image is 64 by 64 dimension but 8x8 pixels
    print(y_test[indicies[1]])
    print(y_predict[indicies[1]])
    plt.imshow(X_proj_img[indicies[1]], cmap=plt.cm.bone, interpolation='nearest') 
    plt.show()
    X_test_img = np.reshape(X_test, (len(X_test),64,64))
    plt.imshow(X_test_img[indicies[1]], cmap=plt.cm.bone, interpolation='nearest') 
    plt.show()
    
if __name__ == "__main__":
    X_trainval, X_test, y_trainval, y_test, X, Y = datasetreader.get_dataset(
        'Sign-Language-Digits-Dataset-master\Dataset')

    X_train, X_val, y_train, y_val = train_test_split(X_trainval, y_trainval, test_size=0.5, random_state=0)

    X_trainval_pca, X_train_pca, X_val_pca, X_test_pca, pca = PCA.apply_PCA(X_trainval, X_train, X_val, X_test)
    #plot_MPL_params(X_train_pca, X_test_pca, y_train, y_test)

    y_pred, MLP = MLP_param(X_train_pca, y_train, X_val_pca)
    print("Test set score: {:.2f}".format(np.mean(y_pred == y_val)))
    print(MLP.best_params_)

    y_predict = apply_MLP_classifier(X_trainval_pca, X_test_pca, y_trainval,
    MLP.best_params_['hidden_layer_sizes'],
    MLP.best_params_['activation'],
    MLP.best_params_['solver'],
    MLP.best_params_['alpha'],
    MLP.best_params_['learning_rate'])
    evaluate(y_test, y_predict)
    
    #print(vis.conf_accuracy(vis.vis_confusion_matrix(y_predict, y_test)))
    #plot_gallery.plotGallery(X_test, y_predict, y_test, 3)
    find_components_from_pic(X_trainval_pca, X_test, y_test, y_predict)
    






