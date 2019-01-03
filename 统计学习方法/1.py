import numpy as np
from scipy.optimize import leastsq
import matplotlib.pyplot as plt
import matplotlib as mpl


# 指定显示字体，防止中文乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 1000  # 保存的图片像素


# real function实际函数
def real_func(x):
    return np.sin(x)


# polynomial多项式
def fit_func(p, x):
    f = np.poly1d(p)
    return f(x)


# 损失函数
def residuals_func(p, x, y):
    ret = fit_func(p, x) - y
    return ret


def getdata(N=10, M=1000):
    x = np.linspace(0, 2*np.pi, N).reshape(-1, 1)
    x_ = np.linspace(0, 2*np.pi, M).reshape(-1, 1)
    y_ = real_func(x)
    r1 = np.random.normal(0, 0.1)
    y = y_ + r1
    return x, y, x_, y_


def fitting(M=0):

    p_lsq = leastsq(residuals_func, [0.1], args=(x, y))
    print('fitting Parameters:', p_lsq[0])
    plt.figure(0)
    plt.plot(x_, real_func(x_), label='实际曲线')
    plt.plot(x, y, 'ro', label='数据')
    plt.plot(x_, fit_func(p_lsq[0], x_), label='拟合曲线')
    plt.legend(loc='best')
    plt.show()

    return p_lsq


if __name__ == '__mian__':
    x, y, x_, y_ = getdata()
    s = fitting(M=0)
