# Here we generate the data in the form of linear regression (y = a*x + b) for the demonstration of gradient descent.

import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import pandas as pd


# init

a = 3 # y = a*x + b
b = 5
n = 1000 # n: number of data
mean = 0 # add normal distribution noise
sigma = 2

# creat data
x = np.linspace(-5, 5, n)

y = a*x + b + np.random.normal(mean, sigma, n)


# y = np.array(y)
data = np.concatenate((np.transpose([x]), np.transpose([y])), axis=1)
# shuffle data
data = shuffle(data)


# save data
df = pd.DataFrame(data)
df.to_csv('test.csv')


