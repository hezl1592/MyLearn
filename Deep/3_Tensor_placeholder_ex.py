# -*- coding: utf-8 -*-
# Author: Zilch
# Creation Date: 2019/2/21
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with tf.Graph().as_default():
    # 定义变量
    w1 = tf.Variable(tf.random_normal([1, 2], stddev=1, seed=1))

    # 定义占位符
    x = tf.placeholder(tf.float32, shape=[None, 2])
    x1 = tf.constant([0.7, 0.9])

    # 查看占位符的类型
    print(x)

    a = x + w1
    b = x1 + w1

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        #
        y_1 = sess.run(a, feed_dict={x: [[0.7, 0.9]]})
        y_2 = sess.run(b)

        print(y_1)
        print(y_2)
        print('b:', b)
        print('------------')
        print('b:\ngraph:{}\nshape:{}\nname:{}\nop:{}'.format(b.graph, b.shape, b.name, b.op))
        print('x1:', x1)
        print('------------')
        print('x1:\ngraph:{}\nshape:{}\nname:{}\nop:{}'.format(x1.graph, x1.shape, x1.name, x1.op))

