import numpy as np
# import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from argparse import Namespace
from sklearn.linear_model import LinearRegression, RidgeCV, LassoCV, ElasticNetCV
from sklearn.preprocessing import PolynomialFeatures
from sklearn.model_selection import train_test_split

# 指定显示字体，防止中文乱码
mpl.rcParams['font.sans-serif'] = ['SimHei']
mpl.rcParams['axes.unicode_minus'] = False
plt.rcParams['savefig.dpi'] = 1000  # 保存的图片像素

args = Namespace(seed=1234,
                 data_file='sample_data.csv',
                 num_samples=100,
                 train_size=0.75,
                 test_size=0.25,
                 num_epochs=100)


def getdata(num_samples):
    np.random.seed(0)
    x = np.linspace(0, 4 * np.pi, num_samples)
    # 产生均匀分布的噪声数据
    random_noise = np.random.uniform(-0.1, 0.1, size=num_samples)
    y = np.sin(x) + random_noise
    x.shape = (-1, 1)
    y.shape = (-1, 1)
    # print(x.shape, y.shape)
    # plt.figure('The data figure', figsize=(9, 6))
    # plt.title("Generated data")
    # plt.scatter(x, y)
    # plt.show()

    return x, y


if __name__ == "__main__":
    x, y = getdata(args.num_samples)
    x_train, x_test, y_train, y_test = train_test_split(x, y,
                                                        test_size=args.test_size,
                                                        random_state=args.seed)
    # print('x_train:{}\ny_train:{}\nx_test:{}\nx_train:{}'.format(x_train.shape, y_train.shape, x_test.shape, y_test.shape))

    # 运用四种模型
    N = 15
    models = [LinearRegression(fit_intercept=False),
              RidgeCV(alphas=[0.1, 0.2, 0.3, 0.5, 1],
                      cv=3, fit_intercept=False),
              LassoCV(alphas=[0.1, 0.2, 0.3, 0.5, 1],
                      cv=3, fit_intercept=False),
              ElasticNetCV(alphas=[0.1, 0.2, 0.3, 0.5, 1], cv=3, fit_intercept=False)]
    models_name = ['LinearRegression', 'RidgeCV',
                   'LassoCV', 'ElasticNetCV']  # 图像标题

    d_pool = np.arange(1, N, 1)  # 阶
    m = d_pool.size
    clrs = []  # 颜色
    for c in np.linspace(16711680, 255, m, dtype=int):
        clrs.append('#%06x' % c)# 定义曲线颜色空间

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
        for i in range(1, N):
            poly = PolynomialFeatures(degree=i)  # 定义多项式阶数
            x1 = poly.fit_transform(x_train)  # 产生多项式x矩阵
            regr.fit(x1, y_train)
            print('the degree:{}\n{}\nthe coef_:{}\n'.format(
                i, poly.get_feature_names(), regr.coef_))
            s = regr.score(poly.fit_transform(x_train), y_train)  # 校验拟合原始数据准确度
            pred = regr.predict(poly.fit_transform(x))  # 绘出预测曲线
            label = '%d 阶拟合曲线, $R^2$=%.3f' % (i, s)
            plt.plot(x, pred, color=clrs[i-1], label=label)
        plt.legend(loc='best')  # 开启图示

    plt.suptitle('多项式曲线拟合比较', fontsize=22)
    plt.tight_layout(1, rect=(0, 0, 1, 0.95))
    # plt.savefig('四种多项式回归比较.png')  # 保存图片
    plt.show()
