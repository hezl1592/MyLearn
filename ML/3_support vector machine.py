import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn import svm
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
    # print(df[df['label'] == 0].count())
    # print(df[df['label'] == 1].count())
    # print(df[df['label'] == 2].count())
    # print((df['label'] == 0).count())
    x = np.zeros(((100, 2)))
    x[:, 0] = df[0:100]['sepal length']
    x[:, 1] = df[0:100]['sepal width']
    y = np.zeros((100, 1))
    y = df[0:100]['label']
    plt.figure('支持向量机示意')
    plt.scatter(x[:50, 0], x[:50, 1], c='b', label='+1')
    plt.scatter(x[50:, 0], x[50:, 1], c='r', label='-1')
    # plt.scatter(df[50:100]['sepal length'], df[50:100]['sepal width'], label='1')
    plt.xlabel('sepal length')
    plt.ylabel('sepal width')
    plt.legend()
    # plt.show()

    return x, y


if __name__ == "__main__":
    init_time = time.time()
    x, y = getdata()
    model = svm.SVC(kernel='linear', C=0.5)
    model.fit(x, y)

    w = model.coef_
    b = model.intercept_.astype(np.float64)
    x1_draw = np.linspace(min(x[:, 0])-0.1, max(x[:, 0])+0.1, 10).reshape(-1, 1)

    def fuc(xx):
        xx1 = (-xx * w[0, 0] - b) / w[0, 1]
        return xx1

    # def fuc1(xx):
    #     xx1 = (-xx * w[0, 0] - b + 1) / w[0, 1]
    #     return xx1

    # def fuc_1(xx):
    #     xx1 = (-xx * w[0, 0] - b - 1) / w[0, 1]
    #     return xx1

    plt.plot(x1_draw, fuc(x1_draw), 'r-')
    # plt.plot(x1_draw, fuc1(x1_draw), 'b:')
    # plt.plot(x1_draw, fuc_1(x1_draw), 'b:')
    # plt.scatter(model.support_vectors_[:, 0], model.support_vectors_[:, 1], s=150,
    # facecolors='none', edgecolors='g')
    plt.title(u'time:{:.2f}s'.format(time.time() - init_time))
    plt.show()
