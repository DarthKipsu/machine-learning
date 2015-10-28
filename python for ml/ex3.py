import matplotlib.pyplot as plt
import numpy as np

# Generate samples from normal distribution with 0 mean and 1 variance

s1 = np.random.normal(0, 1, 10)
s2 = np.random.normal(0, 1, 100)
s3 = np.random.normal(0, 1, 1000)
s4 = np.random.normal(0, 1, 10000)
s5 = np.random.normal(0, 1, 100000)
s6 = np.random.normal(0, 1, 1000000)
s7 = np.random.normal(0, 1, 10000000)

# Plot N vs. the empirical mean

x = np.arange(1, 8)
y = [np.sum(s1)/10,
     np.sum(s2)/100,
     np.sum(s3)/1000,
     np.sum(s4)/10000,
     np.sum(s5)/100000,
     np.sum(s6)/1000000,
     np.sum(s7)/10000000]

plt.plot(x, y, color='b')
plt.show()
