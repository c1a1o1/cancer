'''
Model Evaluation

@author: botpi
'''
import tensorflow as tf
import os

files = os.listdir("../../data/fish/train-fix2/")
samples = len(files)
img_path = "../../data/fish/train-fix2/"
lbl_path = "../../data/fish/label-train-fix2/"
img_queue = []
lbl_queue = []
for file in files:
	img_queue.append(img_path + file)
	lbl_queue.append(lbl_path + file +".txt")

# for i in xrange(len(files)):
# 	print "img_queue", img_queue[i], "lbl_queue", lbl_queue[i]

filename_queue = tf.train.string_input_producer(img_queue, shuffle=False)

filename_queue_label = tf.train.string_input_producer(lbl_queue, shuffle=False)

image_reader = tf.WholeFileReader()
label_reader = tf.WholeFileReader()

img_n , image_file = image_reader.read(filename_queue)
lbl_n , label_file = label_reader.read(filename_queue_label)

init = tf.global_variables_initializer()

with tf.Session() as sess:
	sess.run(init)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)
	for step in xrange(samples):
		print "step:",step + 1
		img_name2, lbl_name2 = sess.run([img_n, lbl_n])
		print "img_name:", img_name2 , "lbl_name:", lbl_name2
coord.request_stop()
coord.join(threads)
sess.close()