# -*- coding: utf-8 -*-
# Author: Zilch
# Creation Date: 2019/2/21
import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with tf.Graph().as_default():
    a = tf.placeholder(tf.float32, [None, 2])
    print('a:', a)

    a.set_shape([4, 2])
    print('a.setshape:', a)

    a_reshape = tf.reshape(a, [1, 8])
    print('a_reshape:', a_reshape)

    # 需注意元素数量匹配
    # a_reshape2 = tf.reshape(a, [2, 2])

    with tf.Session() as sess:
        pass
