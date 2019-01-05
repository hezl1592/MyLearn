# -*- coding: utf-8 -*-
# Author: Zilch
# Creation Date: 2019/01/05

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Perceptron
from sklearn.datasets import load_iris
import matplotlib as mpl
import time


# 指定显示字体，防止中文乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 1000  # 保存的图片像素


def getdata():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['label'] = iris.target
    df.columns = ['sepal length', 'sepal width',
                  'petal length', 'petal width', 'label']
    # print(df.label.value_counts())
    # print(df)

    x = np.zeros(((100, 2)))
    x[:, 0] = df[0:100]['sepal length']
    x[:, 1] = df[0:100]['sepal width']
    y = np.zeros((100, 1))
    y = df[0:100]['label']

    plt.figure('感知机示意')
    plt.scatter(x[:50, 0], x[:50, 1], c='b', label='+1')
    plt.scatter(x[50:, 0], x[50:, 1], c='r', label='-1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    # plt.show()
    return x, y


if __name__ == "__main__":
    init_time = time.time()
    x, y = getdata()
    model = Perceptron(fit_intercept=True, max_iter=10000,
                       shuffle=True, tol=1e-5)
    model.fit(x, y)
    w = model.coef_
    b = model.intercept_
    x1_draw = np.linspace(
        min(x[:, 0])-0.1, max(x[:, 0])+0.1, 10).reshape(-1, 1)
    x2_draw = -(w[0, 0]*x1_draw + b)/w[0, 1]
    # print('w.shape:{}\nb.shape:{}\nx1.shape:{}\nx2.shape:{}'.format(w.shape, b.shape, x1_draw.shape, x2_draw.shape))
    plt.plot(x1_draw, x2_draw)
    # plt.scatter(x[:,0], x[:,1], c=y)
    plt.title(u'time:{:.2f}s'.format(time.time() - init_time))
    plt.show()
