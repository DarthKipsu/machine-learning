from scipy.spatial.distance import cdist
from sklearn.neighbors import KNeighborsClassifier

import math
import matplotlib.pyplot as plt
import numpy as np

# Analytically computed optimal Bayes classifier
boundary = 64 * math.log(2) / 15

'''
create random data points from the given list of distributions
'''
def draw_datapoints(labels):
    data = np.empty(len(labels))
    data[labels == 0] = np.random.normal(0, 1, len(labels[labels == 0]))
    data[labels == 1] = np.random.normal(0, 4, len(labels[labels == 1]))
    return data

'''
Generate 10k samples, classify with the boundary above and compute error rate.
'''
boundary_labels = np.random.randint(2, size=10000)
b_x1 = draw_datapoints(boundary_labels)
b_x2 = draw_datapoints(boundary_labels)
b_errors_y0 = b_x1[(boundary_labels==0) & (b_x1 * b_x1 + b_x2 * b_x2 > math.pow(boundary, 2))]
b_errors_y1 = b_x1[(boundary_labels==1) & (b_x1 * b_x1 + b_x2 * b_x2 <= math.pow(boundary, 2))]
boundary_errors = (len(b_errors_y0) + len(b_errors_y1)) / 10000.0

'''
Create training set with 500 items and print a scatterplot of them
'''
training_labels = np.random.randint(2, size=500)
train_x1 = draw_datapoints(training_labels)
train_x2 = draw_datapoints(training_labels)
training_data = np.column_stack((train_x1, train_x2))

plt.scatter(train_x1[training_labels == 1], train_x2[training_labels == 1], color='blue', edgecolor='black')
plt.scatter(train_x1[training_labels == 0], train_x2[training_labels == 0], color='red', edgecolor='black')
plt.title('Training set')
plt.show()

'''
Create test set with 2000 items and plot it
'''
test_labels = np.random.randint(2, size=2000)
test_x1 = draw_datapoints(test_labels)
test_x2 = draw_datapoints(test_labels)
test_data = np.column_stack((test_x1, test_x2))

plt.scatter(test_x1[test_labels == 1], test_x2[test_labels == 1], color='blue', edgecolor='black')
plt.scatter(test_x1[test_labels == 0], test_x2[test_labels == 0], color='red', edgecolor='black')
plt.title('Test set')
plt.show()

'''
Apply kNN classifier with selected k by Euclidean distance and return percentage of misclassifications
'''
def kNN_missclassifications(k, data, labels):
    neigh = KNeighborsClassifier(n_neighbors=k, metric='euclidean')
    neigh.fit(training_data, training_labels)
    predictions = neigh.predict(data)
    errors = predictions[predictions != labels]
    return len(errors) / float(len(labels))

distances = cdist(test_data, training_data)

def kNN_predictions(k):
    return training_labels[np.argmin(distances, 1)] # this does not take k into account currently

#print cdist(test_data, training_data)
print kNN_predictions(1)
#print kNN_predictions(15)

def kNN_missclassifications_my(k, data, labels):
    predictions = kNN_predictions(k)
    return predictions[predictions != labels]

print kNN_missclassifications(1, test_data, test_labels)

k_values = [57, 41, 33, 25, 21, 17, 13, 9, 7, 5, 3, 1]

errors_test_set = np.array([kNN_missclassifications(k, test_data, test_labels) for k in k_values])
errors_training_set = np.array([kNN_missclassifications(k, training_data, training_labels) for k in k_values])

plt.plot(errors_test_set, color='orange', marker='o', linestyle='--', label='Test')
plt.plot(errors_training_set, color='blue', marker='o', linestyle='--', label='Training')
plt.plot([0,12], [boundary_errors, boundary_errors], color='purple', label='Bayes', linewidth=2)
plt.xticks(range(12), k_values)
plt.xlabel('k - number of nearest neighbors')
plt.ylabel('test error')
plt.title('Missclassification curves')
plt.legend(loc=3)
plt.show()
