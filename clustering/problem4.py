import numpy as np
import mnist_load_show as mnist

from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist

X, y = mnist.read_mnist_training_data(500)

'''
Split data to training and test sets.
training_data, test_data = np.array_split(X, 2)
training_labels, test_labels = np.array_split(y, 2)
'''

def k_means(data, centers):
    '''
    Takes data matrix and initial cluster means, outputs final cluster means and the assignments
    specifying which data vectors are assigned to which cluster after convergence of the K-means
    algorithm.
    '''
    while True:
        distances = cdist(data, centers, 'sqeuclidean')
        clusters = np.argmin(distances, 1)
        new_centers = np.array([data[clusters == i].mean(0) for i in range(len(centers))])
        if (new_centers == centers).all():
            return new_centers, clusters
        centers = new_centers

'''
Print cluster means and assignments for cluster for each digit
'''
centers, clusters = k_means(X, X[:10])
mnist.visualize(centers)
print clusters
