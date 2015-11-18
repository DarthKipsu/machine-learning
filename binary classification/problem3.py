import math
import numpy as np

# Analytically computed optimal Bayes classifier
boundary = 1.71972

# Generate 10k samples, classify with the classifier above and compute error rate.
errors = 0
for y in np.random.randint(2, size=10000):
    if y == 0:
        x1, x2 = np.random.normal(0, 1, 2)
        if math.sqrt(x1*x1 + x2*x2) > boundary: # would classify as y = 1
            errors += 1
    else:
        x1, x2 = np.random.normal(0, 4, 2)
        if math.sqrt(x1*x1 + x2*x2) <= boundary: # would classify as y = 0
            errors += 1

print '%.3f' %(errors / 10000.0)
