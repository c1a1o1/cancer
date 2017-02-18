'''
Model Evaluation

@author: botpi
'''
import tensorflow as tf
import os
import matplotlib.pyplot as plt

path = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer/stage1_100_100_200/"
files = os.listdir(path)
samples = len(files)
img_queue = [path + file for file in files if file.split('.')[1]=='tfrecords']

filename_queue = tf.train.string_input_producer(img_queue, shuffle=False)
reader = tf.TFRecordReader()
img_n , image_file = reader.read(filename_queue)
    
example = tf.parse_single_example(
    image_file,
    features = {
	    'label': tf.FixedLenFeature([1], tf.int64),
	    'vec': tf.FixedLenFeature([512, 100], tf.float32),
	    'proj': tf.FixedLenFeature([100, 512], tf.float32),
	    'med': tf.FixedLenFeature([512], tf.float32),
	}
)

image = tf.transpose(tf.matmul(example['vec'], example['proj'])) + example['med']
label = tf.cast(example['label'], tf.int32)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)
	for step in xrange(samples):
		print "step:",step + 1
		img_name, image_py, label_py = sess.run([img_n, image, label])
		print "name", img_name, label_py
		print "vec", len(image_py), image_py.shape

		#plt.gray()
		#plt.imshow(image_py)
		#plt.show()

coord.request_stop()
coord.join(threads)
sess.close()
print "end"