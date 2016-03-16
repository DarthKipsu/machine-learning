import numpy as np
import mnist_load_show as mnist

from sklearn.metrics import confusion_matrix
from scipy.spatial.distance import cdist

X, y = mnist.read_mnist_training_data(500)

def k_means(data, means):
    '''
    Takes data matrix and initial cluster means, outputs final cluster means and the assignments
    specifying which data vectors are assigned to which cluster after convergence.
    '''
    while True:
        distances = cdist(data, means, 'sqeuclidean')
        clusters = np.argmin(distances, 1)
        new_means = np.array([data[clusters == i].mean(0) for i in range(len(means))])

        if (new_means == means).all():
            return means, clusters
        means = new_means

'''
Print cluster means and assignments for cluster for each digit when original cluster
centers are the first 10 digits.
'''
means, cluster = k_means(X, X[:10])
#mnist.visualize(means)
#for i in range(10):
#    mnist.visualize(np.concatenate((np.array([means[i]]), X[cluster==i]), axis=0))

'''
Print cluster means and assignments for cluster for each digit when original cluster
centers are the first instances of each 10 digits.
'''
means, cluster = k_means(X, np.array([X[np.argmax(y==i)] for i in range(10)]))
#mnist.visualize(means)
#for i in range(10):
#    mnist.visualize(np.concatenate((np.array([means[i]]), X[cluster==i]), axis=0))

def k_medoids(distances, medoids):
    '''
    Takes a dissimilarity matrix and the initial medoids, and outputs the final medoids and the
    assignments specifying which data vectors are assigned to which clusters after convergence.
    '''
    while True:
        clusters = np.argmin(distances[:, medoids], 1)
        new_medoids = np.copy(medoids)

        for i in range(len(medoids)):
            cluster_data = distances[clusters == i][:, clusters == i]
            minimum = np.argmin(cluster_data.sum(axis=1))
            new_medoids[i] = np.where(clusters == i)[0][minimum]

        if (new_medoids == medoids).all():
            return medoids, clusters
        medoids = new_medoids

'''
Print cluster medoids and assignments for cluster for each digit when original cluster
centers are the first 10 digits.
'''
medoids, cluster = k_medoids(cdist(X, X, 'euclidean'), range(10))
#mnist.visualize(np.array([X[medoid] for medoid in medoids]))
#for i in range(len(medoids)):
#    mnist.visualize(np.concatenate((np.array([X[medoids[i]]]), X[cluster==i]), axis=0))

'''
Print cluster medoids and assignments for cluster for each digit when original cluster
centers are the first instances of each 10 digits.
'''
medoids, cluster = k_medoids(cdist(X, X, 'euclidean'), np.array([np.argmax(y==i) for i in range(10)]))
#mnist.visualize(np.array([X[medoid] for medoid in medoids]))
#for i in range(len(medoids)):
#    mnist.visualize(np.concatenate((np.array([X[medoids[i]]]), X[cluster==i]), axis=0))
