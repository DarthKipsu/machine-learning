import numpy as np
import mnist_load_show as mnist
from sklearn.metrics import confusion_matrix

X, y = mnist.read_mnist_training_data()

epoch_count = 10

'''
Split data to training and test sets.
'''
training_data, test_data = np.array_split(X, 2)
training_labels, test_labels = np.array_split(y, 2)

'''
Implement the Perceptron algorithm, including the "pocket" modification for use on non-separable data.
'''
def perceptron(data, label):
    w = np.zeros(len(data[0]), np.int32)
    pocket = np.copy(w)
    pocket_count = 0
    for epoch in range(epoch_count):
        converged = True
        correct_count = 0
        for i in range(len(data)):
            prediction = np.sign(w.dot(data[i]))
            if prediction != label[i]:
                if correct_count > pocket_count:
                    pocket = np.copy(w)
                    pocket_count = correct_count
                correct_count = 0
                w += label[i] * data[i]
                converged = False
            else:
                correct_count += 1
        if converged:
            return w
    if pocket_count == 0:
        print 'zero pockets'
        return w
    return pocket

def one_vs_all_labels(i):
    labels = np.ones(len(training_data), np.int32)
    labels[training_labels != i] = -1
    return labels

def one_vs_all():
    """
    Implement the the multi label classifier using one_vs_all paradigm and return the confusion matrix
    :return: the confusion matrix regarding the result obtained using the classifier
    """
    weights = np.array([perceptron(training_data, one_vs_all_labels(i)) for i in range(10)])
    predictions = np.array([test_data.dot(weights[i]) for i in range(10)])
    predictions = np.argmax(predictions, axis=0)
    one_vs_all_conf_matrix = confusion_matrix(test_labels, predictions)
    print 'error rate one vs all: ', (1 - np.sum(np.diagonal(one_vs_all_conf_matrix)) / float(len(test_data)))
    return one_vs_all_conf_matrix

def all_vs_all():
    """
    Implement the multi label classifier based on the all_vs_all paradigm and return the confusion matrix
    :return: the confusing matrix obtained regarding the result obtained using the classifier
    """
    predictions = np.array([np.empty(len(training_data))])
    votes = np.zeros((10, len(training_data)))
    for i in range(10):
        for j in range(i+1, 10):
            data = training_data[(training_labels == i) | (training_labels == j)]
            true_labels = training_labels[(training_labels == i) | (training_labels == j)]
            labels = np.ones(len(true_labels), np.int32)
            labels[true_labels == j] = -1
            weights = perceptron(data, labels)
            prediction = test_data.dot(weights)
            empty_votes = np.zeros(len(training_data))
            empty_votes[prediction > 0] = 1
            votes[i] += empty_votes
            empty_votes = np.zeros(len(training_data))
            empty_votes[prediction <= 0] = 1
            votes[j] += empty_votes

    all_vs_all_conf_matrix = confusion_matrix(test_labels, np.argmax(votes, axis=0))
    print 'error rate all vs all: ', (1 - np.sum(np.diagonal(all_vs_all_conf_matrix)) / float(len(test_data)))
    return all_vs_all_conf_matrix

print one_vs_all()
print all_vs_all()
