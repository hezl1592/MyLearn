import tensorflow as tf
import os

# 调整警告等级
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def my_regression():
    '''
    自实现一个线性回归
    '''
    # 准备数据
    x = tf.random_normal([100, 1], mean=1.75, stddev=0.5, name="x_data")

    # 矩阵相乘必须是二维的
    y_true = tf.matmul(x, [[0.7]]) + 0.8

    # 建立线性回归模型，y=wx+b
    weight = tf.Variable(tf.random_normal([1, 1], mean=0, stddev=1.0), name="w")
    bias = tf.Variable(0.0, name="b")

    y_predict = tf.matmul(x, weight) + bias

    # 损失函数
    loss = tf.reduce_mean(tf.square(y_true - y_predict))

    # 梯度下降，学习率
    train_op = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

    # 定义一个初始化op
    init_op = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init_op)

        # 打印随机初始化的权重和偏置
        print('随机初始化的权重：{}，偏置：{}'.format(weight.eval(), bias.eval()))

        # 循环优化
        for i in range(500):
            sess.run(train_op)
            print('参数权重：{}，偏置：{}, loss：{}'.format(weight.eval(), bias.eval(), loss.eval()))
            if loss.eval() < 1.0e-7:
                break
        print('训练次数：', i)

    return None


if __name__ == "__main__":
    my_regression()
