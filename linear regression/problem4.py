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
Xm = np.mat(np.ones((len(X),1)))
for K in range(11):
    w = (Xm.T * Xm).I * Xm.T * y
    prediction = Xm * w

    R2_D1 = np.sum(np.power(y - prediction, 2))
    R2_D2 = np.sum(np.power(y - np.mean(y), 2))

    fig, ax = plt.subplots()
    ax.plot(X, y, 'ro')
    ax.plot(X, prediction, 'b-')
    ax.text(0, -4, 'R2: ' + str(1-R2_D1/R2_D2))
    ax.set_xlim([-3, 3])
    plt.title('K = ' + str(K))
    plt.xlabel('X')
    plt.ylabel('y')
    #plt.show()

    Xm = np.concatenate((Xm, np.power(X, K+1)), axis=1)

'''
'''
