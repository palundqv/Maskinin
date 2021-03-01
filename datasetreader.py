# Arda Mavi
import os
import numpy as np
from os import listdir
from cv2 import imread, resize, cvtColor, COLOR_BGR2GRAY
from keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# Settings:
img_size = 64
grayscale_images = True
num_class = 10
test_size = 0.2

def get_img(data_path):
    # Getting image array from path:
    img = imread(data_path)
    img = cvtColor(img, COLOR_BGR2GRAY)
    img = resize(img, (img_size, img_size, 1 if grayscale_images else 3))
    return img

def get_dataset(dataset_path='Dataset'):
    # Getting all data from data path:
    try:
        X = np.load('npy_dataset/X.npy')
        Y = np.load('npy_dataset/Y.npy')
    except:
        labels = listdir(dataset_path) # Geting labels
        X = []
        Y = []
        for i, label in enumerate(labels):
            datas_path = dataset_path+'/'+label
            for data in listdir(datas_path):
                img = get_img(datas_path+'/'+data)
                X.append(img)
                Y.append(i)
        # Create dateset:
        X = 1-np.array(X).astype('float32')/255.
        Y = np.array(Y).astype('float32')
        Y = to_categorical(Y, num_class)
        if not os.path.exists('npy_dataset/'):
            os.makedirs('npy_dataset/')
        np.save('npy_dataset/X.npy', X)
        np.save('npy_dataset/Y.npy', Y)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size, random_state=0)
    return X_train, X_test, Y_train, Y_test

if __name__ == '__main__':
    get_dataset('/Users/per/Documents/Dev/python/memory/Maskinin_FinalProject/Sign-Language-Digits-Dataset-master/Dataset')

print(X_train)