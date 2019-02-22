import tensorflow as tf

with tf.Graph().as_default():
    # 创建常量op
    a = tf.constant(1)
    b = tf.constant(2)
    c = tf.constant(3)
    d = tf.constant(4)

    # 创建运算op
    add1 = tf.add(a, b)
    mul1 = tf.multiply(b, c)
    add2 = tf.add(c, d)
    output = tf.add(add1, mul1)

    with tf.Session() as sess:
        sess.run(add2)
        print(output.eval())

'''
with tf.Session() as sess:
    with tf.device("/gpu:1"):
        matrix1 = tf.constant([[3., 3.]])
        matrix2 = tf.constant([[2.],[2.]])
        product = tf.matmul(matrix1, matrix2)
'''
