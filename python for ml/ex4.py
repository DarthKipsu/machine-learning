import matplotlib.pyplot as plt
import numpy as np

# create a sample of random permutation with random fixed points

def perms(n):
    "Creates a permutation sample with length n and F^n fixed points."
    return np.random.permutation(n)

samples = [perms(100),
           perms(150),
           perms(200),
           perms(250),
           perms(300),
           perms(350),
           perms(400),
           perms(450),
           perms(500),
           perms(550)]

# Get percent of fixed points for a sample

def fixed_points(sample):
    fixed = 0
    for i in sample:
        if sample[i] == i + 1:
            fixed += 1
    return fixed * 100

fixed = list(map(lambda x: fixed_points(x) / len(x), samples))

# Draw a histogram to display the percent of fixed points

plt.hist(fixed, bins=5)
plt.show()
