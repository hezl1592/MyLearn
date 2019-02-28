# -*- coding: utf-8 -*-
# Author: Zilch
# Creation Date: 2019/2/26
import tensorflow as tf

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()

    # defined before
    # PATH_TO_FROZEN_GRAPH = MODEL_NAME + '/frozen_inference_graph.pb'
    # open the file
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')