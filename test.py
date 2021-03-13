import numpy as np
import datasetreader
import train_classifiers
import numpy as np

X_train, X_test, y_train, y_test, new_X, new_Y = datasetreader.get_dataset()

rho = np.corrcoef(X_train)
rho.style.background_gradient(cmap='coolwarm', axis=None).set_precision(2)
print('done!')