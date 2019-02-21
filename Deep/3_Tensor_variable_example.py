import tensorflow as tf
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    a = tf.constant([1.0, 2.0])
    b = tf.constant([3.0, 4.0])
    c = a*b
    sess.run(c)
    print(c.eval())
    print(sess.graph)

print('-----------分割线---------------')

with tf.Session() as sess1:
    # 创建两个变量，
    total = tf.Variable(tf.zeros([1, 2]))
    wegiht = tf.Variable(tf.random_uniform([1, 2]))

    # 初始化变量
    tf.global_variables_initializer().run()

    # 更新数据，op
    update_weights = tf.assign(wegiht, tf.random_uniform([1, 2], -1.0, 1.0))
    update_total = tf.assign(total, tf.add(total, wegiht))

    for i in range(5):

        sess1.run(update_weights)
        sess1.run(update_total)

        print(wegiht.eval(), total.eval())

    print(sess1.graph)
