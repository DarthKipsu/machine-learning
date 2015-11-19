from sklearn.neighbors import KNeighborsClassifier
import math
import matplotlib.pyplot as plt
import numpy as np

# Analytically computed optimal Bayes classifier
boundary = 1.71972

'''
Generate 10k samples, classify with the boundary above and compute error rate.
'''
boundary_errors = 0
for y in np.random.randint(2, size=10000):
    if y == 0:
        x1, x2 = np.random.normal(0, 1, 2)
        if math.sqrt(x1*x1 + x2*x2) > boundary: # would classify as y = 1
            boundary_errors += 1
    else:
        x1, x2 = np.random.normal(0, 4, 2)
        if math.sqrt(x1*x1 + x2*x2) <= boundary: # would classify as y = 0
            boundary_errors += 1
boundary_errors /= 10000.0

'''
create random data points from the given list of distributions
'''
def draw_datapoints(labels):
    data = []
    for y in labels:
        if y == 0:
            data.append(np.array([np.random.normal(0, 1), np.random.normal(0, 1)]))
        else:
            data.append(np.array([np.random.normal(0, 4), np.random.normal(0, 4)]))
    return np.array(data)

'''
Create training set with 500 items and print a scatterplot of them
'''
training_labels = np.random.randint(2, size=500)
training_data = draw_datapoints(training_labels)
plt.scatter(training_data[training_labels == 1,0], training_data[training_labels == 1,1], color='blue', edgecolor='black')
plt.scatter(training_data[training_labels == 0,0], training_data[training_labels == 0,1], color='red', edgecolor='black')
plt.title('Training set')
plt.show()

'''
Create test set with 2000 items and plot it
'''
test_labels = np.random.randint(2, size=2000)
test_data = draw_datapoints(test_labels)
plt.scatter(test_data[test_labels == 1,0], test_data[test_labels == 1,1], color='blue', edgecolor='black')
plt.scatter(test_data[test_labels == 0,0], test_data[test_labels == 0,1], color='red', edgecolor='black')
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
    #print len(errors) / 2000.0
    return len(errors) / 2000.0

errors_test_set = np.array([kNN_missclassifications(k, test_data, test_labels) for k in [57, 41, 33, 25, 21, 17, 13, 9, 7, 5, 3, 1]])
errors_training_set = np.array([kNN_missclassifications(k, training_data, training_labels) for k in [57, 41, 33, 25, 21, 17, 13, 9, 7, 5, 3, 1]])

plt.plot(errors_test_set, color='orange', marker='o', linestyle='--', label='Test set')
plt.plot(errors_training_set, color='blue', marker='o', linestyle='--', label='Training set')
plt.plot([0,12], [boundary_errors, boundary_errors], color='red', label='Estimated error')
plt.xticks(range(12), ['57', '41', '33', '25', '21', '17', '13', '9', '7', '5', '3', '1'])
plt.xlabel('k - number of nearest neighbors')
plt.ylabel('test error')
plt.title('Missclassification curves')
plt.legend(loc=5)
plt.show()
