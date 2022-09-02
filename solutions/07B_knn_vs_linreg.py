from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
import numpy as np
import pandas as pd
import matplotlib as plt


dataurl = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(dataurl, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])  # why odd rows have [:2] columns  
target = raw_df.values[1::2, 2]

print('data.shape:', data.shape)

x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=42)

linreg = LinearRegression()
knnreg = KNeighborsRegressor(n_neighbors=3)

linreg.fit(x_train, ytrain)
print('Linear Regression Train/Test: %.3f/%.3f' % (linreg.score(x_train, y_train), linreg.score(x_test, y_test)))

knnreg.fit(x_train, y_train)
print('KNeighborsRegressor Train/Test: %.3f/%.3f' % (knnreg.score(x_train, y_train), knnreg.score(x_test, y_test)))

plt.plot(xtest, ytest, 'o', label="data")
plt.legend(loc='best');
