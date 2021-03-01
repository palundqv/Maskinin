import datasetreader
from sklearn.decomposition import PCA


x_trainval, x_test, y_trainval, y_test = datasetreader.get_dataset(
    '/Users/per/Documents/Dev/python/memory/Maskinin_FinalProject/Sign-Language-Digits-Dataset-master/Dataset')

GroupKFold,