# -*- coding: utf-8 -*-
# Author: Zilch
# Creation Date: 2019/2/25
import tensorflow as tf
import os

# 调整警告等级
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
'''
csv文件读取：
1. 先找到文件，构造一个列表
2. 构造文件队列
3. 构造阅读器，读取队列内容（按行）
4. 解码内容
5. 批处理（多样本）
'''


def csvread(file_list):
    '''
    读取CSV文件
    ：parameter filelist:文件路径+名字的列表
    ：return    读取的内容
    '''
    # 1.构建文件的队列
    file_queue = tf.train.string_input_producer(file_list, shuffle=True)

    # 2.构建CSV阅读器
    reader = tf.TextLineReader()
    key, value = reader.read(file_queue)

    # 3.对每一行内容进行解码
    '''
    record_defaults:参数决定了所得张量的类型，并指定默认值
                    Acceptable types are `float32`, `float64`, `int32`, `int64`, `string`.
    '''
    records = [['name'], ['sex'], [160], [60]]
    # records = [['apple'], ['jpg'], [4], [2], ['apple'], [22], [22], [22], [22]]

    name, sex, height, weight = tf.decode_csv(
        value, record_defaults=records, field_delim=",")
    # name, categr, height, width, label, x1, x2, x3, x4 = tf.decode_csv(value, record_defaults=records, field_delim=",")

    # 4.想要读取多个数据， 就需要批处理
    name_batch,  sex_batch, height_batch, weight_batch = tf.train.batch(
        [name, sex, height, weight], batch_size=10, num_threads=1, capacity=10)

    # return name, sex, height, weight
    print(name_batch,  sex_batch, height_batch, weight_batch)
    return name_batch,  sex_batch, height_batch, weight_batch
    # return name, categr, height, width, label, x1, x2, x3, x4


if __name__ == "__main__":
    file_path = os.path.join(os.getcwd(), 'test_file\\csv_file')
    file_name = os.listdir(file_path)
    file_list = [os.path.join(file_path, file) for file in file_name]

    name, sex, height, weight = csvread(file_list)
    # name, categr, height, width, label, x1, x2, x3, x4 = csvread(file_list)

    # 开启会话
    with tf.Session() as sess:
        # 定义一个线程协调器
        coord = tf.train.Coordinator()

        # 开启读文件的线程
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        # 打印读取的内容
        print(sess.run([name, sex, height, weight]))
        # print(sess.run([name, categr, height, width, label, x1, x2, x3, x4]))

        coord.request_stop()
        coord.join(threads=threads)
