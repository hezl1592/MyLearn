# 监督学习

## 1.1 线性模型

$$
\hat{y}(w, x) = w_0 + w_1 x_1 + ... + w_p x_p
$$

参数==**coef_**==：$w=(w_1, w_2, ..., w_p)$，==**intercept_**==:$w_0$

### 1.1.1 最小二乘法

Ordinary Least Squares:
$$
 \min_{w} {|| X w - y||_2}^2
$$

```python
from sklearn import linear_model
import numpy as np


reg = linear_model.LinearRegression()
reg.fit(x_train, y_train)
pred = reg.predict(x_test, y_test)
# get 
omega_1 = reg.coef_
omega_0 = reg.intercept_
omega = np.concatenate((reg.intercept_, reg.coef_), axis=1)
print(omega)
```

### 1.1.2 岭回归

Ridge regression，改良的最小二乘法，$L2$范数正则化项：
$$
\min_{w} {{|| X w - y||_2}^2 + \alpha {||w||_2}^2}
$$

```python

from sklearn import linear_model

reg = linear_model.Ridge(alpha=.5)
reg.fit(x_train, y_train)
pred = reg.predict(x_test, y_test)
# get 
omega_1 = reg.coef_
omega_0 = reg.intercept_
omega = np.concatenate((reg.intercept_, reg.coef_), axis=1)
print(omega)
```

$\alpha$：the complexity of the ridge，we can use the function RidgeCV to set it

function **RidgeCV**  to find the best alpha.



### 1.1.3 Lasso回归

Lasso Regression，Lasso能够将一些作用比较小的特征的参数训练为0，从而获得稀疏解。也就是说用这种方法，在训练模型的过程中实现了降维(特征筛选)的目的。改良的最小二乘法，$L1$范数正则化项：
$$
\min_{w} \frac {1}{2n_{samples}}{{( X w - y)}^2 + \alpha {||w||_1}^2}
$$

### 1.1.4 Elastic Net

弹性网络是结合了岭回归和Lasso回归，由两者加权平均所得。据介绍这种方法在特征数大于训练集样本数或有些特征之间高度相关时比Lasso更加稳定。
$$
\min_{w} { \frac{1}{2n_{samples}} ||X w - y||_2 ^ 2 + \alpha \rho ||w||_1 +
\frac{\alpha(1-\rho)}{2} ||w||_2 ^ 2}
$$

## 1.2 支持向量机

支持向量机（Support Vector Machine，SVM）是一种二分类器。基本模型是定义于特征空间上的间隔最大的线性分类器，间隔最大化有别于感知机；

同时，支持向量机还包括核技巧，成为非线性分类器。

![222](C:\Users\Zilch\OneDrive - mails.jlu.edu.cn\Notebook\image\sphx_glr_plot_separating_hyperplane_0011.png)

| 优点                                                         | 缺点                                                         |
| :----------------------------------------------------------- | :----------------------------------------------------------- |
| 在高维空间有效。<br />在尺寸数量大于样本数量的情况下仍然有效。<br />在决策函数中使用训练点的子集（称为支持向量），因此它也具有内存效率<br />多功能：可以为决策功能指定不同的内核功能。提供了通用内核，但也可以指定自定义内核。 | 当特征数量远大于样本数量，则避免选择内核函数时产生过拟合，正则化项是至关重要的。<br />SVM不直接提供概率估计，这些是使用昂贵的五重交叉验证计算的（参见下面的分数和概率）。 |

### 1.2.1 数学公式

分割平面：
$$
w^T\Phi(x)+b=0
$$
训练集、目标值：
$$
\begin{matrix}
训练集：x_1,x_2,x_3,...,x_n\\ 
目标集：y_1,y_2,y_3,...,y_n,y_i\in\begin{Bmatrix}
1,-1
\end{Bmatrix}
\end{matrix}
$$
分类决策函数：
$$
f(x)=sign(w^Tx+b)
$$
有：
$$
y_i\cdot y(x_i)>0
$$
目标函数：
$$
\underset{w,b}{\arg\max}\begin{Bmatrix}
\frac {1}{||w||}\underset{i}{\min} y_i\cdot (w^T\cdot \Phi(x)+b)
\end{Bmatrix}
$$
间隔：
$$
\begin{matrix}
函数间隔：y_i\cdot (w^T\cdot \Phi(x)+b)\\ 
几何间隔：\frac {1}{||w||}y_i\cdot (w^T\cdot \Phi(x)+b)
\end{matrix}
$$
在此时，假设将$w$和$b$按比例改变，使得两类点的函数值都满足$|y|≥1$，此时几何间隔没有改变，而函数间隔改变。

新的目标函数可以转化为：
$$
\begin{matrix}
\underset{w,b}{\max}\frac{1}{||w||}\\ 
s.t.y_i\cdot (w^T\cdot \Phi(x)+b)≥1,i=1,2,3
\end{matrix}\Leftrightarrow 
\begin{matrix}
\underset{w,b}{\min}\frac{1}{2}||w||^2\\ 
s.t.y_i\cdot (w^T\cdot \Phi(x)+b)≥1,i=1,2,3
\end{matrix}
$$
加入松弛因子$\xi_i≥0$，使得函数间隔大于等于1，这样目标函数：
$$
\begin{matrix}
\underset{w,b}{\min}\frac{1}{2}||w||^2+C\sum_{i=1}^{N}\xi_i\\ 
s.t.y_i\cdot (w^T\cdot \Phi(x)+b)≥1-\xi_i,i=1,2,3...\\
\xi_i≥0，i=1,2,3...
\end{matrix}
$$

### 1.2.2 核函数

使用核函数，可以将原始输入空间映射到新的特征空间，从而使得原本线性不可分的样本可能在核空间中可分。

- 多项式核函数：

$$
K(x_1, x_2)=(x_1\cdot x_2+c)^p
$$

对应的支持向量为一个p次多项式分类器，分类决策函数变为：
$$

$$


- 高斯核RBF函数：

$$
K(x_1, x_2)=e^{-\frac{||x_1-x_2||^2}{2\sigma ^2}}
$$




In Scikit-Learn，function **SVC、NuSVC、LinearSVC**能够对数据集执行多类分类。