'''
Create submission

Created on Nov 8, 2016

block best solution
positivos 565 totales 1584
positivos 31 totales 2256
positivos 30 totales 2286
gran total 6127

0 and 1
positivos 726 totales 1584
positivos 3 totales 2256
positivos 17 totales 2286
gran total 6127

@author: botpi
'''
import tensorflow as tf
import numpy as np
from apiepi import *

def eval_conv(images, parameters, features):
    cv1_size = parameters["cv1_size"]
    cv2_size = parameters["cv2_size"]
    cv1_channels = parameters["cv1_channels"]
    cv2_channels = parameters["cv2_channels"]
    hidden = parameters["hidden"]
    img_resize = parameters["img_resize"]

    W_conv1 = weight_variable([cv1_size, cv1_size, 1, cv1_channels])
    b_conv1 = bias_variable([cv1_channels])
    W_conv2 = weight_variable([cv2_size, cv2_size, cv1_channels, cv2_channels])
    b_conv2 = bias_variable([cv2_channels])
    W_fc1 = weight_variable([img_resize/4 * img_resize/4 * cv2_channels, hidden])
    b_fc1 = bias_variable([hidden])
    W_fc2 = weight_variable([hidden, 2])
    b_fc2 = bias_variable([2])
    
    x = tf.placeholder(tf.float32, shape=[None, 256])
    x_image = tf.reshape(x, [-1,img_resize,img_resize,1])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    h_pool2_flat = tf.reshape(h_pool2, [-1, img_resize/4 * img_resize/4  * cv2_channels])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    h_fc1_drop = tf.nn.dropout(h_fc1, 1)
    pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)    
    
    init = tf.initialize_all_variables()
    
    with tf.Session() as sess:
        sess.run(init)
        prob = pred.eval({x:images, 
            W_conv1:features["W_conv1"],
            b_conv1:features["b_conv1"][0],
            W_conv2:features["W_conv2"],
            b_conv2:features["b_conv2"][0],
            W_fc1:features["W_fc1"],
            b_fc1:features["b_fc1"][0],
            W_fc2:features["W_fc2"],
            b_fc2:features["b_fc2"][0]                          
          })

    return prob
