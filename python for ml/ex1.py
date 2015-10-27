import matplotlib.pyplot as plt
import numpy as np

# 100 samples from normal distribution with 0 mean and 1 variance

samples = np.random.normal(0, 1, 100)

# Visualize data with a histogram

plt.hist(samples, bins=50)
plt.show()

# Sort them to increasing order

sorted_samples = np.sort(samples)

# Draw a line plot for x=i:th sample and y is i/100

y = np.arange(0,1,0.01)

plt.plot(sorted_samples, y)
plt.show()
