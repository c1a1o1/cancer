'''
Model Evaluation

@author: botpi
'''
import tensorflow as tf
import os
import matplotlib.pyplot as plt

def tfqueue(files):
	filename_queue = tf.train.string_input_producer(files, shuffle=False)
	reader = tf.TFRecordReader()
	name, data = reader.read(filename_queue)
	    
	example = tf.parse_single_example(
	    data,
	    features = {
		    'label': tf.FixedLenFeature([2], tf.int64),
		    'vec': tf.FixedLenFeature([512, 100], tf.float32),
		    'proj': tf.FixedLenFeature([100, 512], tf.float32),
		    'mean': tf.FixedLenFeature([512], tf.float32),
		}
	)

	image = tf.transpose(tf.matmul(example['vec'], example['proj'])) + example['mean']	
	label = [tf.cast(example['label'], tf.float32)]
	x, y = tf.train.shuffle_batch(
    	[image, label], batch_size = 100, 
    	capacity = 1000,
    	min_after_dequeue = 600)

	#image = example["vec"]
	# label = example['label']
	# label = tf.cast(example['label'], tf.int32)	
	# def f1(): return tf.constant([0, 1])
	# def f2(): return tf.constant([1, 0])
	# label = tf.cond(label[0]>0, f1, f2)

	return x, y, name

#path = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer/stage1_100_100_200/"
path = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer/stage1_100_100_200_test/"
files = [path + file for file in os.listdir(path)]
samples = len(files)
tfimage, tflabel, tfname = tfqueue(files)

init = tf.global_variables_initializer()
with tf.Session() as sess:
	sess.run(init)
	coord = tf.train.Coordinator()
	threads = tf.train.start_queue_runners(coord=coord, sess=sess)
	for step in xrange(5):
		print "step:", step + 1
		name, image, label = sess.run([tfname, tfimage, tflabel])
		print "name", name, label
		print "vec", len(image), image.shape

		plt.gray()
		plt.imshow(image[50])
		plt.show()

coord.request_stop()
try: 
    coord.join(threads)
except:
    pass
sess.close()
print "end"