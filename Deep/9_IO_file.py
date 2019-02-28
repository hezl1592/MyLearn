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
    file_queue = tf.train.string_input_producer(file_list)

    # 2.构建CSV阅读器
    reader = tf.TextLineReader()
    key, value = reader.read(file_queue)

    # 3.对每一行内容进行解码
    '''
    record_defaults:参数决定了所得张量的类型，并指定默认值
                    Acceptable types are `float32`, `float64`, `int32`, `int64`, `string`.
    '''
    records = [['apple'], ['jpg'], [4], [2], ['apple'], [22], [22], [22], [22]]
    example, label = tf.decode_csv(value, record_defaults=records, field_delim=",")

    print(value)

    return None


if __name__ == "__main__":
    file_path = os.path.join(os.getcwd(), 'test_file\\csv_file')
    file_name = os.listdir(file_path)
    file_list = [os.path.join(file_path, file) for file in file_name]

    csvread(file_list)
