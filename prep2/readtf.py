'''
read tf record
'''

import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer"
group = "stage 1 samples_100_100_200"
#group = "stage1"
path = directory + '/' + group 
patients = [s for s in os.listdir(path)]
print path
print patients
for patient in patients:
	patient = "0a0c32c9e08cc2ea76a71649de56be6d_1.tfrecords"
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
	    #print i


	plt.show()

	