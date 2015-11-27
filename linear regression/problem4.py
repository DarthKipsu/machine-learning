import math
import matplotlib.pyplot as plt
import numpy as np

'''
Randomly generate 30 x from uniform distribution on the interval [-3,3]
Then randomly generate y using y=f(x)+n where f(x) = 2+x-0.5x^2 and n i.i.d.
normal random variables with 0 mean and standard deviation 0.4
'''
X = np.mat(np.sort(np.random.uniform(-3, 3, (30,1)), axis=0))
y = np.mat(np.copy(X))
y[:] = 2 + X - 0.5 * np.power(X, 2)
y += np.random.normal(0, 0.4, (30, 1))

'''
Fit polynomials of order 0 to 10 to this dataset using linear regression,
minimizing the sum of squares error and plot them seperately.
Also display the coefficient of determination R2
'''
Xm = np.mat(np.ones((len(X), 1)))
X_fit = np.mat(np.ones((100, 1)))

for K in range(11):
    w = (Xm.T * Xm).I * Xm.T * y
    prediction = Xm * w
    fit = X_fit * w

    R2_D1 = np.sum(np.power(y - prediction, 2))
    R2_D2 = np.sum(np.power(y - np.mean(y), 2))

    fig, ax = plt.subplots()
    ax.plot(X, y, 'ro')
    ax.plot(np.linspace(-3, 3, 100), fit, 'b-')
    ax.text(0, -4, 'R2: ' + str(1-R2_D1/R2_D2))
    ax.set_xlim([-3, 3])
    ax.set_ylim([-5, 5])
    plt.title('K = ' + str(K))
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

    print '\nCoefficients, k: ', K, w

    Xm = np.concatenate((Xm, np.power(X, K+1)), axis=1)
    X_fit = np.concatenate((X_fit, np.mat(np.power(np.linspace(-3, 3, 100), K+1)).T), axis=1)

'''
Divide dataset into 10 equal size data sets.
'''
subsets = np.split(X, 10)

'''
Calculate the sum of squared errors for given k and subsets using the j:th
subset as a test set
'''
def SSE(K, j):
    training_set = np.concatenate((X[:j*3-3], X[j*3:]))
    training_fit = np.mat(np.ones((len(training_set), 1)))
    test_X = np.mat(np.ones((3, 1)))

    for k in range(1, K+1):
        training_fit = np.concatenate((training_fit, np.power(training_set, k)), axis=1)
        test_X = np.concatenate((test_X, np.power(X[j*3-3:j*3], k)), axis=1)

    w = (training_fit.T * training_fit).I * training_fit.T * np.concatenate((y[:j*3-3], y[j*3:]))

    predictions = test_X * w
    actual = y[j*3-3:j*3]
    return np.sum(np.power(predictions - actual, 2))

errors = [np.sum([SSE(K, j) for j in range(1,11)]) for K in range(11)]
print '\nSum of squared errors: ', errors

plt.plot(range(11), np.log2(errors))
plt.title('Squared errors with 10-fild cross-vaidation')
plt.xlabel('K')
plt.ylabel('Sum of squared errors (in logarithmic scale)')
plt.show()
