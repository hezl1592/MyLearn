# -*- coding: utf-8 -*-
# Author: Zilch
# Creation Date: 2019/2/22
import tensorflow as tf
import os

# 调整警告等级
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with tf.Session() as sess:
    # 固定值张量
    zero = tf.zeros([3, 4], tf.float32)
    print(zero)

    sess.run(zero)
    print(zero.eval())

    print('-----------------------------------------')

    # 随机值张量
    random = tf.random_normal([3, 4], mean=0.0, stddev=1.0, dtype=tf.float32, seed=1)
    print(random)

    sess.run(random)
    print(random.eval())

    print('-----------------------------------------')

    # 张量切片与拓展
    a = [[1, 2, 3], [4, 4, 5]]
    b = [[2, 2, 2], [3, 3, 3]]
    c = tf.concat([a, b], axis=1)
    print(c)
    sess.run(c)
    print(c.eval())
