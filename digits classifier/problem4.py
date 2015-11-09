import mnist_load_show as mnist
import numpy as np

from scipy.spatial.distance import cdist

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

def prototype(digit):
    digits = training_data[training_labels == digit]
    return np.array(np.mean(digits, axis=0, dtype=np.int32))

mnist.visualize(prototype(4))
