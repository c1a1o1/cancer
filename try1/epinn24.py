'''
convolutional

Created on Nov 2, 2016


@author: botpi
'''

import tensorflow as tf
import numpy as np
from apiepi import *
import scipy.io

def tf_queue(files):
    filename_queue = tf.train.string_input_producer(files, shuffle=False)
    reader = tf.TFRecordReader()
    name, data = reader.read(filename_queue)
        
    example = tf.parse_single_example(
        data,
        features = {
            'label': tf.FixedLenFeature([2], tf.int64),
            'vec': tf.FixedLenFeature([100, 512, 100], tf.float32),
            'proj': tf.FixedLenFeature([100, 100, 512], tf.float32),
            'mean': tf.FixedLenFeature([100, 512], tf.float32),
        }
    )
    image = example['vec']
    #label = example['label']

    label = [tf.cast(example['label'], tf.float32)]
    # def f1(): return tf.constant([0., 1.])
    # def f2(): return tf.constant([1., 0.])
    # label = tf.cond(label[0]>0, f1, f2)

    return image, label, name

def train_tf(files, parameters, training_epochs = 100):
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
    
    display_step = 1
    save_step = 1000
    if save_step > training_epochs:
        save_step = training_epochs
    
    best_cost = 1e99
    best_acc = 0

    #x = tf.placeholder(tf.float32, shape=[None, 256])
    #y = tf.placeholder(tf.float32, shape=[None, 2])  
    x, y, name = tf_queue(files)

    # First Convolutional Layer  
    W_conv1 = weight_variable([cv1_size, cv1_size, image_channels, cv1_channels])
    b_conv1 = bias_variable([cv1_channels])
    
    x_image = tf.reshape(x, [-1, img_height, img_width, image_channels])
    
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    print "h_conv1:", h_conv1
    print "h_pool1:", h_pool1
    
    # Second Convolutional Layer
    W_conv2 = weight_variable([cv2_size, cv2_size, cv1_channels, cv2_channels])
    b_conv2 = bias_variable([cv2_channels])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    print "h_conv2:", h_conv2
    print "h_conv2:",  h_pool2
    
    # Densely Connected Layer
    W_fc1 = weight_variable([img_height/4 * img_width/4 * cv2_channels, hidden])
    b_fc1 = bias_variable([hidden])
    
    h_pool2_flat = tf.reshape(h_pool2, [-1, img_height/4 * img_width/4  * cv2_channels])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
    
    # Dropout
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # Readout Layer
    W_fc2 = weight_variable([hidden, 2])
    b_fc2 = bias_variable([2])
    
    pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
    print "pred:", pred    

    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred + 1e-20), reduction_indices=[1]))
    print "cost", cost
    #cost = tf.reduce_sum(loss)
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    ye = y
    pe = pred

    init = tf.global_variables_initializer()    
    with tf.Session() as sess:
        sess.run(init)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord, sess=sess)

        # Training cycle
        for epoch in xrange(training_epochs):
            print epoch
            # _, c, acc, yp, pp = sess.run([optimizer, cost, accuracy, ye, pe], feed_dict={keep_prob: dropout})
            try:
                _, c, acc = sess.run([optimizer, cost, accuracy], feed_dict={keep_prob: dropout})
            except:
                print "error " + sess.run([name])

            if (epoch+1) % display_step == 0:
                print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c)
                
            if (epoch+1) % save_step == 0:
                features = {}
                features["W_conv1"] = W_conv1.eval()
                features["b_conv1"] = b_conv1.eval()
                features["W_conv2"] = W_conv2.eval()
                features["b_conv2"] = b_conv2.eval()
                features["W_fc1"] = W_fc1.eval()
                features["b_fc1"] = b_fc1.eval()
                features["W_fc2"] = W_fc2.eval()
                features["b_fc2"] = b_fc2.eval()                
                scipy.io.savemat("data/resp_%s" % (epoch+1), features, do_compression=True)
        
        print "Optimization Finished!"
    
        # Test model
     
        #acc = accuracy.eval({x:images, y: labels, keep_prob: 1})    
        #prob = pred.eval({x:images, y: labels, keep_prob: 1})
        
    print "end trainning"
    coord.request_stop()
    try: 
        coord.join(threads)
    except:
        pass
    sess.close()
    prob = 0
    #acc = 0

    return features, prob, acc, c
