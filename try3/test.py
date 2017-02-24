'''
Create submission

@author: botpi
'''
import tensorflow as tf
import numpy as np
import model_test as model
import params
import os
import pandas as pd
import scipy.io

print "begin"
path = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer/stage1_100_100_200_test/"
files = [path + file for file in os.listdir(path)]
features_file = "data/resp_10"
sub_file = "data/stage1_submission_6.csv"

r = []
r.append(["id", "cancer"])

features = scipy.io.loadmat(features_file)
parameters = params.get()

r = []
for file in os.listdir(path):
	print file
	n = 0
	s = []
	for serialized_example in tf.python_io.tf_record_iterator(path + file):
	    example = tf.train.Example()
	    example.ParseFromString(serialized_example)

	    label = np.array(example.features.feature['label'].int64_list.value)
	    #vec = np.array(example.features.feature['vec'].float_list.value, dtype="float32").reshape([512, 100])
	    proj = np.array(example.features.feature['proj'].float_list.value, dtype="float32").reshape([100, 512])
	    #mean = np.array(example.features.feature['med'].float_list.value, dtype="float32")

	    pred = model.eval_conv(proj, label, parameters, features)
	    prob_cancer = pred[0][0][1]
	    s.append(prob_cancer)
	    n += 1
	    print n, prob_cancer
	
	d = {}
	d["id"] = file
	d["mean"] = s / n
	d["count"] = sum([1 for x in s if x>=0.5])
	d["max"] = max(a)
	d["min"] = min(a)
	r.append(d)

print d