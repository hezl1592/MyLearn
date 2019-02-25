# -*- coding: utf-8 -*-
# Author: Zilch
# Creation Date: 2019/2/25
import tensorflow as tf
import os

# 调整警告等级
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# 模拟一下同步先处理数据，然后才能取数据训练
# 1.首先定义一个队列
queue1 = tf.FIFOQueue(capacity=3, dtypes=tf.float32)
queue2 = tf.RandomShuffleQueue()

# 放入一些数据
# enqueue1 = queue1.enqueue_many([[0.1], [0.2], [0.3]])
enqueue1 = queue1.enqueue_many([[0.1, 0.2, 0.3], ])
#
out_queue = queue1.dequeue()
data = out_queue + 1
en_queue = queue1.enqueue(data)

with tf.Session() as sess:
    # 初始化队列
    sess.run(enqueue1)
    # print(sess.run(enqueue1))

    for i in range(100):
        sess.run(en_queue)
        # print(sess.run(en_queue))

    for i in range(queue1.size().eval()):
        # sess.run(queue1.dequeue())
        print(sess.run(queue1.dequeue()))




