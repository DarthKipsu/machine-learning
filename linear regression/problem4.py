import math
import matplotlib.pyplot as plt
import numpy as np

'''
Randomly generate 30 x from uniform distribution on the interval [-3,3]
Then randomly generate y using y=f(x)+n where f(x) = 2+x-0.5x^2 and n i.i.d.
normal random variables with 0 mean and standard deviation 0.4
'''
X = np.sort(np.random.uniform(-3, 3, 30))
y = np.array([2 + x - 0.5 * math.pow(x, 2) for x in X])
y += np.random.normal(0, 0.4, 30)

'''
Fit polynomials of order 0 to 10 to this dataset using linear regression,
minimizing the sum of squares error and plot them seperately.
Also display the coefficient of determination R2
'''
for K in range(11):
    x_matrix = np.array([[math.pow(x, k) for k in range(K + 1)] for x in X])
    x_t = x_matrix.transpose()
    w = np.dot(np.linalg.inv(np.dot(x_t, x_matrix)), np.dot(x_t, y))
    fit = np.array([sum(np.multiply(x, w)) for x in x_matrix])

    R2_n = np.sum(np.array([math.pow(y[i] - fit[i], 2) for i in range(30)]))
    R2_d = np.sum(np.array([math.pow(y[i] - 1/30*np.sum(y), 2)]))
    print 1 - R2_n/R2_d

    fig, ax = plt.subplots()
    ax.plot(X, y, 'ro')
    ax.plot(X, fit, 'b-')
    ax.text(0,-4, 'R2: ' + str(1-R2_n/float(R2_d)))
    ax.set_xlim([-3, 3])
    plt.title('K = ' + str(K))
    plt.xlabel('X')
    plt.ylabel('y')
    plt.show()

'''
'''
