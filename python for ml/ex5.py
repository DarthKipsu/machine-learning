import matplotlib.pyplot as plt
import numpy as np
import random

# Write a function to select from two given distributions with the given probability pi

def function(n, pi, exp1, var1, exp2, var2):
    norm1 = np.random.normal(exp1, var1, n)
    norm2 = np.random.normal(exp2, var2, n)

    x = np.arange(n)
    y = np.arange(n)

    for i in range(0, n):
        if random.random() < pi:
            x[i] = norm1[i]
            y[i] = 0
        else:
            x[i] = norm2[i]
            y[i] = 1

    return x, y

# plot some result to see what they are like

x, y = function(10, 0.5, 1, 1, 3, 2)
plt.subplot(3, 1, 1)
plt.scatter(x, y, c='r')

x, y = function(10, 0.2, 0, 1, 0, 5)
plt.subplot(3, 1, 2)
plt.scatter(x, y, c='b')

x, y = function(10, 0.7, 2, 3, 4, 6)
plt.subplot(3, 1, 3)
plt.scatter(x, y, c='g')

plt.show()
