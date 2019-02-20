import tensorflow as tf
import os



# print(tf.__version__)

# 调整警告等级
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 创建一张图，上下文环境
g = tf.Graph()
print(g)
with g.as_default():
    c = tf.constant(11.0)
    print(c.graph)


# 实现一个加法运算
# 定义张量
a = tf.constant(5.0)
b = tf.constant(6.0)

sum1 = tf.add(a, b)

# 默认图，相当于给程序分配一段内存
graph = tf.get_default_graph()
print(graph)

print(sum1)

# 会话，只能运行一个graph，可以指定
with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as sess:
    print(sess.run(sum1))
    print(sess.graph)
    print(sum1.graph)
    print(a.graph)
