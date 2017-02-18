'''
read tf record
'''

import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

def queue(filename):
    filename_queue = tf.train.string_input_producer(tf.train.match_filenames_once(filename))
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    features = tf.parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
        'prot': tf.FixedLenFeature([], tf.string),
        'label': tf.FixedLenFeature([], tf.int64),
    })
    return features["prot"], features["label"]


directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer"
group = "stage 1 samples_100_100_200"
group = "stage1_100_100_200"
path = directory + '/' + group 
patients = [s for s in os.listdir(path)]
#print path
#print patients
for patient in patients:
	#patient = "0a0c32c9e08cc2ea76a71649de56be6d_1.tfrecords"
	#patient = "0015ceb851d7251b8f399e39779d1e7d_1.tfrecords"
	patient = "008464bb8521d09a42985dd8add3d0d2_1.tfrecords"
	patient = "026be5d5e652b6a7488669d884ebe297_2.tfrecords"
	print patient
	i=1
	for serialized_example in tf.python_io.tf_record_iterator(path + "/" + patient):
	    example = tf.train.Example()
	    example.ParseFromString(serialized_example)

	    # traverse the Example format to get data
	    label = np.array(example.features.feature['label'].int64_list.value)
	    vec = np.array(example.features.feature['vec'].float_list.value, dtype="float32").reshape([512, 100])
	    proj = np.array(example.features.feature['proj'].float_list.value).reshape([100, 512])
	    mean = np.array(example.features.feature['med'].float_list.value)
	    #print label.shape, vec.shape, proj.shape, mean.shape, vec.dtype

	    Ar = np.dot(vec, proj).T + mean
	    Ar[Ar==2000] = 0
	    ax = plt.subplot(11, 15, i, frame_on=False)
	    ax.xaxis.set_major_locator(plt.NullLocator())
	    ax.yaxis.set_major_locator(plt.NullLocator())
	    plt.gray()
	    plt.imshow(Ar)
	    i+=1
	    print i


	plt.show()

	