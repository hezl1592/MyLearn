import tensorflow as tf

# 计算必须包含于Session的上下文中，session在graph中运行，如果没有指定图，则在默认图中运行
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    a = tf.constant([1, 1, 1, 1])
    b = tf.constant([2, 2, 2, 2])
    sum1 = tf.add(a, b)

    sess.run(sum1)
    print(sum1.eval())
    print(sess.graph)
