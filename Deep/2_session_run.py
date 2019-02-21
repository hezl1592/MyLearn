# -*- coding: utf-8 -*-
# Author: Zilch
# Creation Date: 2019/2/21
import tensorflow as tf
import os


# 调整警告等级
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with tf.Graph().as_default():

    # 不是定义的OP类型，在TensorFlow中无法运行
    # var1 = 3
    # var2 = 3
    # sum2 = var1 + var2

    # TensorFlow的重载机制，默认重载成op类型
    a = tf.constant(2.0)
    var1 = 2.0
    sum2 = a + var1

    with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
        print(sess.run([sum2]))
        print(sess.graph)
