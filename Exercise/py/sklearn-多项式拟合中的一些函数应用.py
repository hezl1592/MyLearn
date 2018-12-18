# -*- coding: utf-8 -*-
# Author: Zilch
# Creation Date: 2018/12/18

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

'''
关于sk-learn库中多项式拟合的一些函数用法
'''
print(__doc__)


# 产生一维数组x， 以及相应的y
def get_data(N=100, p=1):
    np.random.seed(0)
    x = np.random.uniform(0, 6, size=N).reshape(-1, 1)
    x = np.sort(x, axis=0)
    y = x ** 2 - 4 * x - 3 + np.random.randn(N).reshape(-1, 1)
    y.shape = -1, 1
    return x, y, N, p


x, y, n, p = get_data()
plt.plot(x, y, 'r.')
plt.show()

poly = PolynomialFeatures(degree=2)
poly.fit(x)
x1= poly.transform(x)
