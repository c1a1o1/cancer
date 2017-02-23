'''
Para manejar la variedad de slices entre los diferentes pacientes

1. Elegimos pacientes con slices entre 100 y doscientos
2. Cada paciente se parte en 2 grupos cada uno con 102 slices
   El primero va de 1 a 101 y el segundo de 100 a 199
3. A los dos grupos se les da el mismo label

En este script se vam a utilizar los archivos que contienen las imagenes reducidas por el pca

Ensayarenos con el formato .mat
'''

import numpy as np
import os
import scipy.io
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf

def add_file(name, label, part):
	slices = []
	for c1 in part:
		print "c1[2]", c1[2].shape, c1[2].dtype, c1[2][0].shape, c1[2][0][0]
		#print "c1[2]", c1[2].shape, c1[2].dtype
		slices.append((c1[2], label))

	a = {}
	a["slices"] = slices
	#scipy.io.savemat(new_dir + name, a, do_compression=True)

def add_file_tfrecords(name, label, part):
	print name
	part = part.T
	vec = np.array([x for x in part[0]])
	proj = np.array([x for x in part[2]])
	mean = np.array([x for x in part[3]])
	if label:
		label = [0, 1]
	else:
		label = [1, 0]

	writer = tf.python_io.TFRecordWriter(new_dir + name + ".tfrecords")
	example = tf.train.Example(
		features=tf.train.Features(
			feature={
				'label': tf.train.Feature(
					int64_list=tf.train.Int64List(value=label)),
				'vec': tf.train.Feature(
					float_list=tf.train.FloatList(value=np.real(vec).flatten().tolist())),
				'proj': tf.train.Feature(
					float_list=tf.train.FloatList(value=np.real(proj).flatten().tolist())),
				'mean': tf.train.Feature(
					float_list=tf.train.FloatList(value=mean.flatten().tolist())),
	}))
	writer.write(example.SerializeToString())
	writer.close()

def net_files(orig, dest):
	files_orig = {os.listdir(orig)}
	files_dest = {os.listdir(dest)}
	print files_orig - files_dest
	z()

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
			for i in xrange(nslices - 2):
				add_file_tfrecords(patient[0] + "_" + str(i), patient[1], mat["c"][i: i + 3])
