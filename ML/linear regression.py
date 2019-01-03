# -*- coding: utf-8 -*-
# Author: Zilch
# Creation Date: 2018/12/17

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures

# 指定显示字体，防止中文乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 1000  # 保存的图片像素


# 制造随机数据
def get_data(N=100, p=1):
    np.random.seed(0)
    x = np.random.uniform(0, 6, size=N).reshape(-1, 1)  # 生成训练用的x
    x_pred = np.random.uniform(0, 6, size=N * 20).reshape(-1, 1)  # 生成测试用的x
    x = np.sort(x, axis=0)
    x_pred = np.sort(x_pred, axis=0)
    y = x ** 2 - 4 * x - 3 + np.random.randn(N).reshape(-1, 1)
    y.shape = -1, 1
    return x, x_pred, y, N, p


if __name__ == "__main__":
    x, x_pred, y, N, p = get_data(100, 2)  # 获取数据以及相关参数

    # 运用四种模型
    models = [LinearRegression(fit_intercept=False),
              RidgeCV(alphas=[0.1, 0.2, 0.3, 0.5, 1],
                      cv=3, fit_intercept=False),
              LassoCV(alphas=[0.1, 0.2, 0.3, 0.5, 1],
                      cv=3, fit_intercept=False),
              ElasticNetCV(alphas=[0.1, 0.2, 0.3, 0.5, 1], cv=3, fit_intercept=False)]
    models_name = ['LinearRegression', 'RidgeCV',
                   'LassoCV', 'ElasticNetCV']  # 图像标题
    clrs = ['r', 'b', 'g', 'w', 'm', 'y', 'k']  # 定义曲线颜色空间

    plt.figure('kk', figsize=(18, 12), facecolor='w')  # 产生图片窗口
    plt.axis('off')  # 主图坐标轴显示关闭

    for c, model in enumerate(models):
        regr = models[c]
        plt.subplot(2, 2, c + 1)  # 子图
        plt.title(models_name[c], fontsize=18)
        plt.plot(x, y, 'c*', label='原始数据')  # 绘制原始数据点图
        plt.xlabel('X', fontsize=18)
        plt.ylabel('Y', fontsize=18)
        plt.grid(True)  # 显示网格
        for i in range(1, 7):
            poly = PolynomialFeatures(degree=i)  # 定义多项式阶数
            x1 = poly.fit_transform(x)  # 产生多项式x矩阵
            regr.fit(x1, y)
            print('the degree:{}\n{}\nthe coef_:{}\n'.format(
                i, poly.get_feature_names(), regr.coef_))
            s = regr.score(x1, y)  # 校验拟合原始数据准确度
            pred = regr.predict(poly.fit_transform(x_pred))  # 绘出预测曲线
            label = '%d 阶拟合曲线, $R^2$=%.3f' % (i, s)
            plt.plot(x_pred, pred, color=clrs[i], label=label)
        plt.legend(loc='best')  # 开启图示

    plt.suptitle('多项式曲线拟合比较', fontsize=22)
    plt.tight_layout(1, rect=(0, 0, 1, 0.95))
    # plt.savefig('四种多项式回归比较.png')  # 保存图片
    plt.show()
