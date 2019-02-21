import tensorflow as tf


g = tf.Graph()
print(g)
with g.as_default():
    with tf.Session() as sess:
        a = tf.constant(1.0)
        b = tf.constant(2.0)

        c = tf.add(a, b)

        sess.run(c)
        print(c.eval())
        print('c:', c.graph)
        assert c.graph is g, '...erro'

ab = tf.constant(3.0)
print('a:', ab.graph)


# Constructing and making default:
with tf.Graph().as_default() as g1:
    c = tf.constant(5.0)
    print('g:{}\ng1:{}'.format(g, g1))

    # 判断两个graph的差别
    # assert c.graph is g, 'not'


print('\n------------------------\n')
with tf.Session(graph=g) as sess1:
    a = tf.constant(2.0)
    sess1.run(a)
    print(sess1.graph)

with tf.Session(graph=g1) as sess2:
    b = tf.constant(3.9)
    sess2.run(b)
    print(sess2.graph)