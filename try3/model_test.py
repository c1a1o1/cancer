'''
Create submission

@author: botpi
'''
import tensorflow as tf
import numpy as np
from apiepi import *

def eval_conv(x, y, parameters, features):
    image_channels = parameters["image_channels"]
    cv1_size = parameters["cv1_size"]
    cv2_size = parameters["cv2_size"]
    cv1_channels = parameters["cv1_channels"]
    cv2_channels = parameters["cv2_channels"]
    hidden = parameters["hidden"]
    img_height = parameters["img_height"]
    img_width = parameters["img_width"]
    learning_rate = parameters["learning_rate"]
    dropout = parameters["dropout"]

    #x = tf.placeholder(tf.float32, shape=[None, 256])
    y = tf.placeholder(tf.float32, shape=[None, 2])  
    
    W_conv1 = weight_variable([cv1_size, cv1_size, image_channels, cv1_channels])
    b_conv1 = bias_variable([cv1_channels])    
    x_image = tf.reshape(x, [-1, img_height, img_width, image_channels])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    W_conv2 = weight_variable([cv2_size, cv2_size, cv1_channels, cv2_channels])
    b_conv2 = bias_variable([cv2_channels])    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    W_fc1 = weight_variable([img_height/4 * img_width/4 * cv2_channels, hidden])
    b_fc1 = bias_variable([hidden])    
    h_pool2_flat = tf.reshape(h_pool2, [-1, img_height/4 * img_width/4  * cv2_channels])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    W_fc2 = weight_variable([hidden, 2])
    b_fc2 = bias_variable([2])
    pred = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred + 1e-20), reduction_indices=[1]))
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    
    init = tf.global_variables_initializer() 
    with tf.Session() as sess:
        sess.run(init)
        prob = sess.run([pred], {
                    W_conv1:features["W_conv1"],
                    b_conv1:features["b_conv1"][0],
                    W_conv2:features["W_conv2"],
                    b_conv2:features["b_conv2"][0],
                    W_fc1:features["W_fc1"],
                    b_fc1:features["b_fc1"][0],
                    W_fc2:features["W_fc2"],
                    b_fc2:features["b_fc2"][0]                          
                }
            )
    return prob
