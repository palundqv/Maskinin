from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier

def apply_MLP_classifier(X_train_pca, X_test_pca, y_train):
    clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train_pca, y_train)
    MPL_predicts = clf.predict(X_test_pca)
    return MPL_predicts

def MLP_param(X_train_pca, y_train, X_test_pca):
    MLP = MLPClassifier()
    grid_params = [{'max_iter': [1000],
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive']}]
    MLP_gscv = GridSearchCV(estimator=MLP, param_grid=grid_params, cv=5)
    
    MLP_gscv.fit(X_train_pca, y_train)
    y_pred = MLP_gscv.predict(X_test_pca)

    return y_pred, MLP_gscv