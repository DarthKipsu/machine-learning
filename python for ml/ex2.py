import matplotlib.pyplot as plt
import numpy as np
import sys

# Create random 10x5 matrix and save it in file

write_matrix = np.random.random((10, 5))
np.save('matriisi.npy', write_matrix)

# Read created matrix from the file given as parameter

matrix = np.load(sys.argv[1])

# plot row and column sums in a same window

column_sums = np.sum(matrix, axis=0)
row_sums = np.sum(matrix, axis=1)

plt.subplot(2, 1, 1)
plt.plot(column_sums)

plt.subplot(2, 1, 2)
plt.plot(row_sums)

plt.show()

# print matrix dimensions

print(matrix.shape)

# print the sum of all elements in the matrix

print(np.sum(matrix))
