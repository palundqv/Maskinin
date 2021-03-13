
import matplotlib.pyplot as plt

def plotGallery(X_test, y_pred, y_test, amountOfPictures):
    preds = titles(y_pred, y_test)
    cols = rows = amountOfPictures
    plt.figure()
    for i in range(rows * cols):
        plt.subplot(rows, cols, i + 1)
        plt.imshow(X_test[i][:].reshape((64, 64)), cmap=plt.cm.gray)
        plt.title(preds[i])
        plt.xticks(())
        plt.yticks(())

    plt.show()

def titles(y_pred, y_test):
    predicted_names = []
    for i in range(y_pred.shape[0]):
        predicted_names.append('predicted: {0}\ntrue: {1}'.format(y_pred[i], y_test[i]))
    return predicted_names
