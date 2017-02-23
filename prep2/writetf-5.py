'''

'''

import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

def add_file_tfrecords(name, label, part):
	print name
	part = part.T
	# print part.shape, part[2].shape, part[2][0].shape
	# vec = np.array([x for x in part[0]])
	# proj = np.array([x for x in part[2]])
	# mean = np.array([x for x in part[3]])
	# print proj.shape
	if label:
		label = [0, 1]
	else:
		label = [1, 0]

	writer = tf.python_io.TFRecordWriter(new_dir + name + ".tfrecords")
	for i in xrange(len(part[2])):
		# print part[2][i].shape
		# z()
		example = tf.train.Example(
			features=tf.train.Features(
				feature={
					'label': tf.train.Feature(
						int64_list=tf.train.Int64List(value=label)),
					'vec': tf.train.Feature(
						float_list=tf.train.FloatList(value=np.real(part[0][i]).flatten().tolist())),
					'proj': tf.train.Feature(
						float_list=tf.train.FloatList(value=np.real(part[2][i]).flatten().tolist())),
					'mean': tf.train.Feature(
						float_list=tf.train.FloatList(value=part[3][i].flatten().tolist())),
		}))
		writer.write(example.SerializeToString())
	writer.close()

def read_mat(file, mat):
	if mat:
		return mat
	else:
		try:
			return scipy.io.loadmat(old_dir + patient[0])
		except:
			return None

directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer"
#group = "stage 1 samples"
group = "stage1"

ori_dir = directory + '/' + group  + '/'
old_dir = directory + '/' + group + '_100' + '/'

#train
# labels = pd.read_csv(directory + '/' + group + '_labels.csv')
# new_dir = directory + '/' + group + '_100_100_200' + '/'

#test
labels = pd.read_csv(directory + '/stage1_sample_submission.csv')
new_dir = directory + '/' + group + '_100_100_200_test' + '/'

if not os.path.exists(new_dir):
	os.makedirs(new_dir)

labels = labels.values.tolist()
for patient in labels:
	if os.path.isfile(old_dir + patient[0] + ".mat"):
		nslices = len(os.listdir(ori_dir + patient[0]))
		mat = read_mat(old_dir + patient[0], None)
		if mat:
			add_file_tfrecords(patient[0], patient[1], mat["c"])
