# -*- coding: utf-8 -*-
# Author: Zilch
# Creation Date: 2018/12/17

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False


def getdata(N=6, p=1):
    np.random.seed(0)
    x1 = np.linspace(0, 6, N)
    y = x1 ** 2 - 4 * x1 - 3 + np.random.randn(N)
    x1.shape = -1, 1
    y.shape = -1, 1
    # if p != 1:
    #     #     x = x1
    #     #     for i in range(p, p + 1):
    #     #         x = np.concatenate((x, x1 ** i), axis=1)
    #     # else:
    #     #     x = x1
    return x1, y, N, p


if __name__ == "__main__":
    x1, y1, N, p = getdata(17, 2)
    plt.figure()
    plt.plot(x1[:, 0], y1, 'r*')
    plt.show()
    models = [Pipeline([
        ('poly', PolynomialFeatures()),
        ('linear', LinearRegression(fit_intercept=False))]),
        Pipeline([
            ('poly', PolynomialFeatures()),
            ('linear', RidgeCV(alphas=np.logspace(-3, 2, 10), fit_intercept=False))]),
        Pipeline([
            ('poly', PolynomialFeatures()),
            ('linear', LassoCV(alphas=np.logspace(-3, 2, 10), fit_intercept=False))]),
        Pipeline([
            ('poly', PolynomialFeatures()),
            ('linear', ElasticNetCV(alphas=np.logspace(-3, 2, 10), l1_ratio=[.1, .5, .7, .9, .95, .99, 1],
                                    fit_intercept=False))])
    ]

    d_pool = np.arange(1, p, 1)  # 阶
    m = d_pool.size

    clrs = []  # 颜色
    for c in np.linspace(16711680, 255, m, dtype=int):
        clrs.append('#%06x' % c)

    for i, model in enumerate(models):
        regr = model
        regr.set_params(poly__degree=p)
        regr.fit(x1[:, 0], y1)
        # print(regr.intercept_, regr.coef_)
        # f1 = np.linspace(0, 6, N*10)
        # for i in range(p, p + 1):
        #     f1 = np.concatenate((f1, x ** i), axis=1)
        # f1.shape = -1, 1
        x_hat = np.linspace(x1[:, 0].min(), x1[:, 0].max(), num=100)
        x_hat.shape = -1, 1
        pred = regr.predict(x_hat)

        plt.plot(x_hat, pred, color=clrs[i])
        plt.show()
#
#
# c = linear_model.LinearRegression()
# c.set_params(poly__degree=3)
