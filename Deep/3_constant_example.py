import tensorflow as tf


with tf.Graph().as_default():
    # 创建交互式的session
    sess = tf.InteractiveSession()

    # 定义常量：不可更改的张量
    a = tf.constant(value=[1.0, 2.0], shape=[3, 2])
    sess.run(a)
    print(sess.run(a))
    print(a)
