import tensorflow as tf
import os

# 调整警告等级
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


a = tf.constant([1, 2, 3, 4, 5])

# 变量必须进行显示初始化
var = tf.Variable(tf.random_normal([2, 3], mean=0, stddev=1))
init_op = tf.global_variables_initializer()

print(a, '\n', var)

with tf.Session() as sess:
    sess.run(init_op)
    print(sess.run([a, var]))