import mnist_load_show as mnist
import numpy as np

from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix

# Load the first 5000 images from mnist.
X, y = mnist.read_mnist_training_data(5000)

'''
Uncomment if you need a sanity check to see if the data is in a right format.
Should print the first 100 digits and their right labels.
'''
# mnist.visualize(X[0:100])
# print(y[0:100])

# split data in training and test sets.
training_data, test_data = np.array_split(X, 2)
training_labels, test_labels = np.array_split(y, 2)

#print(training_labels[0:12])
#mnist.visualize(training_data[0:12])
#print(test_labels[0:12])
#mnist.visualize(test_data[0:12])

def prototype(label):
    """
    Creates a prototype image of the training set images with a given label.
    """
    digits = training_data[training_labels == label]
    return np.array(np.mean(digits, axis=0, dtype=np.int32))

prototypes = np.array([prototype(n) for n in range(0,10)])
#mnist.visualize(prototypes)

def simple_EC_classifier():
    """
    Implement the classifier based on the Euclidean distance
    :return: the confusing matrix obtained regarding the result obtained using simple Euclidean distance method
    """
    predictions = np.array([np.argmin(dist) for dist in cdist(test_data, prototypes)])
    simple_EC_conf_martix = confusion_matrix(test_labels, predictions)
    return simple_EC_conf_martix

print(simple_EC_classifier())

