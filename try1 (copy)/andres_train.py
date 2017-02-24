'''
Model Train

Created on Jan 17, 2017

@author: botpi
'''

import tensorflow as tf
import numpy as np
from apifish import *
import scipy.io
from params import param
import time

print "initialize"
training_epochs = 200000
display_step = 1

parameters = param()
cv1_size = parameters["cv1_size"]
cv2_size = parameters["cv2_size"]
cv1_channels = parameters["cv1_channels"]
cv2_channels = parameters["cv2_channels"]
hidden = parameters["hidden"]
img_width = parameters["img_width"]
img_height = parameters["img_height"]
categories = parameters["categories"]
learning_rate = parameters["learning_rate"]
dropout = parameters["dropout"]

best_cost = 1e99
best_acc = 0

filename_queue = tf.train.string_input_producer(
    tf.train.match_filenames_once("../../data/fish/train-fix/*.jpg"), shuffle=False)

filename_queue_label = tf.train.string_input_producer(
    tf.train.match_filenames_once("../../data/fish/label-train-fix/*.txt"), shuffle=False)

image_reader = tf.WholeFileReader()
label_reader = tf.WholeFileReader()
# label_reader = tf.TextLineReader()

_ , image_file = image_reader.read(filename_queue)
_ , label_file = label_reader.read(filename_queue_label)

channels_jpg = 1
ratio_jpg = 1
img_height = img_height/ratio_jpg
img_width = img_width/ratio_jpg
image = tf.image.decode_jpeg(image_file, channels=channels_jpg, ratio=ratio_jpg)
image.set_shape([img_height, img_width, channels_jpg])
# label = tf.string_to_number(label_file, out_type=tf.int32)
record_defaults = [[0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0], [0.0]]
col1, col2, col3, col4, col5, col6, col7, col8 = tf.decode_csv(label_file, record_defaults=record_defaults)
label = tf.pack([col1, col2, col3, col4, col5, col6, col7, col8])
# label = tf.pack([label])
# label = tf.cast(label, tf.float32)
# label = tf.string_to_number(label, out_type=tf.int32)
# print "label", label
# label.set_shape([categories])
# min_after_dequeue + 3 * batch_size
batch_size = 30
x, y = tf.train.shuffle_batch(
    [image, label], batch_size = batch_size, 
    capacity = 1000,
    min_after_dequeue = 600)

x = tf.cast(x, tf.float32)
y = tf.cast(y, tf.float32)
keep_prob = tf.placeholder(tf.float32)
  
W_conv1 = weight_variable([cv1_size, cv1_size, channels_jpg, cv1_channels])
b_conv1 = bias_variable([cv1_channels])
W_conv2 = weight_variable([cv2_size, cv2_size, cv1_channels, cv2_channels])
b_conv2 = bias_variable([cv2_channels])
W_fc1 = weight_variable([img_width/4 * img_height/4 * cv2_channels, hidden])
b_fc1 = bias_variable([hidden])
W_fc2 = weight_variable([hidden, categories])
b_fc2 = bias_variable([categories])
x = tf.reshape(x, [-1, img_width, img_height,channels_jpg])
h_conv1 = tf.nn.relu(conv2d(x, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
h_pool2_flat = tf.reshape(h_pool2, [-1, img_width/4 * img_height/4  * cv2_channels])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
# pred = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
pred = tf.matmul(h_fc1_drop, W_fc2) + b_fc2
print "x", x
print "y", y
print "pred", pred
print "h_conv1", h_conv1
print "h_pool1", h_pool1
print "h_conv2", h_conv2
print "h_pool2", h_pool2

# cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred + 1e-20), reduction_indices=[1]))
# cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y)) + tf.reduce_mean(tf.reduce_sum(tf.abs(tf.subtract(y,pred)),1))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
y_arg2 = tf.argmax(y, 1)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord, sess=sess)

    sleep = 10
    print "queue sleep in seg:", sleep
    time.sleep(sleep)
    print "end queue sleep"

    # Training cycle
    print "learning..."
    for epoch in xrange(training_epochs):
        # s = s[ beginning : beginning + LENGTH]
        # label = sess.run(label)
        # print label
        _, c, acc = sess.run([optimizer, cost, accuracy], feed_dict={keep_prob: dropout})

        #if (epoch+1) % display_step == 0:
        # acc = accuracy.eval({keep_prob: 1})
        print "Epoch:", '%04d' % (epoch+1), "cost=", "{:.9f}".format(c),"dropout: 0.5 bad acc:", round(acc*100.0,2),"%"
            
        if epoch%50 == 0:
            alb = 0
            bet = 0
            dol = 0
            lag = 0
            nof = 0
            other = 0
            shark = 0
            yft = 0

            alb_total = 0.00000001
            bet_total = 0.00000001
            dol_total = 0.00000001
            lag_total = 0.00000001
            nof_total = 0.00000001 
            other_total = 0.00000001
            shark_total = 0.00000001
            yft_total = 0.00000001
            # acc = accuracy.eval(feed_dict={keep_prob: 1.0})
            acc, y_arg , c_p = sess.run([accuracy,y_arg2,correct_prediction],feed_dict={keep_prob: 1.0})
            # print "len - y_arg",y_arg
            # print "len - c_p",len(c_p)
            for i in xrange(len(y_arg)):
                if y_arg[i] == 0:
                    alb_total += 1
                elif y_arg[i] == 1:
                    bet_total += 1
                elif y_arg[i] == 2:
                    dol_total += 1
                elif y_arg[i] == 3:
                    lag_total += 1
                elif y_arg[i] == 4:
                    nof_total += 1
                elif y_arg[i] == 5:
                    other_total += 1
                elif y_arg[i] == 6:
                    shark_total += 1
                elif y_arg[i] == 7:
                    yft_total += 1

                if c_p[i] == True:
                    if y_arg[i] == 0:
                        alb += 1
                    elif y_arg[i] == 1:
                        bet += 1
                    elif y_arg[i] == 2:
                        dol += 1
                    elif y_arg[i] == 3:
                        lag += 1
                    elif y_arg[i] == 4:
                        nof += 1
                    elif y_arg[i] == 5:
                        other += 1
                    elif y_arg[i] == 6:
                        shark += 1
                    elif y_arg[i] == 7:
                        yft += 1

            print "###########################################################################"
            print "                                         saving weights"
            print "                                         accuracy:", round(acc*100.0,2),"%"
            print "    ALB  ", round(alb*100.0/alb_total,2),"total good int:",alb, "of", int(alb_total)
            print "    BET  ", round(bet*100.0/bet_total,2),"total good int:",bet, "of", int(alb_total)
            print "    DOL  ", round(dol*100.0/dol_total,2),"total good int:",dol, "of", int(dol_total)
            print "    LAG  ", round(lag*100.0/lag_total,2),"total good int:",lag, "of", int(lag_total)
            print "    NoF  ", round(nof*100.0/nof_total,2),"total good int:",nof, "of", int(nof_total)
            print "    OTHER", round(other*100.0/other_total,2),"total good int:",other, "of", int(other_total)
            print "    SHARK", round(shark*100.0/shark_total,2),"total good int:",shark, "of", int(shark_total)
            print "    YFT  ", round(yft*100.0/yft_total,2),"total good int:",yft, "of", int(yft_total)
            print
            print "    total batch", (alb_total+bet_total+dol_total+lag_total+nof_total+other_total+shark_total+yft_total)
            print "###########################################################################"
            features = {}
            features["W_conv1"] = W_conv1.eval()
            features["b_conv1"] = b_conv1.eval()
            features["W_conv2"] = W_conv2.eval()
            features["b_conv2"] = b_conv2.eval()
            features["W_fc1"] = W_fc1.eval()
            features["b_fc1"] = b_fc1.eval()
            features["W_fc2"] = W_fc2.eval()
            features["b_fc2"] = b_fc2.eval()
            scipy.io.savemat("resp_100_cost", features, do_compression=True) 
        
        # acc = accuracy.eval({x:x, y: y, keep_prob: 1})     
        # if acc > best_acc:
        #     best_acc = acc
        #     best_epoch_acc = epoch
        #     features = {}
        #     features["W_conv1"] = W_conv1.eval()
        #     features["b_conv1"] = b_conv1.eval()
        #     features["W_conv2"] = W_conv2.eval()
        #     features["b_conv2"] = b_conv2.eval()
        #     features["W_fc1"] = W_fc1.eval()
        #     features["b_fc1"] = b_fc1.eval()
        #     features["W_fc2"] = W_fc2.eval()
        #     features["b_fc2"] = b_fc2.eval()
        #     scipy.io.savemat("resp_best_acc", features, do_compression=True) 
                
    print "Optimization Finished!"

    # Test model
 
    # acc = accuracy.eval({x:images, y: labels, keep_prob: 1})    
    # prob = pred.eval({x:images, y: labels, keep_prob: 1})
#         print auctf[0]
#         aucr = auctf[0].eval({x:images, y: labels, keep_prob: 1})
    
    features = {}
    features["W_conv1"] = W_conv1.eval()
    features["b_conv1"] = b_conv1.eval()
    features["W_conv2"] = W_conv2.eval()
    features["b_conv2"] = b_conv2.eval()
    features["W_fc1"] = W_fc1.eval()
    features["b_fc1"] = b_fc1.eval()
    features["W_fc2"] = W_fc2.eval()
    features["b_fc2"] = b_fc2.eval()

    coord.request_stop()
    coord.join(threads)
    sess.close()

# print "best cost", best_cost, "epoch", best_epoch_cost
# print "best acc", best_acc, "epoch", best_epoch_acc

print "epochs learning_rate cv1_size cv2_size cv1_channels cv2_channels hidden img_heigth img_width dropout"
print ("    %s      %s        %s       %s          %s            %s        %s        %s      %s      %s" 
    % (training_epochs, parameters["learning_rate"], parameters["cv1_size"], parameters["cv2_size"], parameters["cv1_channels"]
       , parameters["cv2_channels"], parameters["hidden"], parameters["img_height"], parameters["img_width"], parameters["dropout"]))

# print "Accuracy:", acc #, "Accuracy test:", test_acc
print "Cost", cost

print "saving last"
scipy.io.savemat("saving_last_resp", features, do_compression=True)    
print "end"