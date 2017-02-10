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

def add_file(name, label, part):
	slices = []
	for c1 in part:
		print "c1[2]", c1[2].shape, c1[2].dtype
		slices.append((c1[2], label))

	a = {}
	a["slices"] = slices
	z()
	scipy.io.savemat(new_dir + name, a, do_compression=True)


directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer"
group = "stage 1 samples"
group = "stage1"

ori_dir = directory + '/' + group  + '/'
old_dir = directory + '/' + group + '_100' + '/'
new_dir = directory + '/' + group + '_100_100_200' + '/'
if not os.path.exists(new_dir):
	os.makedirs(new_dir)

labels = pd.read_csv(directory + '/' + group + '_labels.csv')
labels = labels.values.tolist()

for patient in labels:
	if os.path.isfile(old_dir + patient[0] + ".mat"):
		nslices = len(os.listdir(ori_dir + patient[0]))
		if nslices >=100 and nslices<200:
			print patient[0], patient[1]
			mat = scipy.io.loadmat(old_dir + patient[0])
			add_file(patient[0] + "_1", patient[1], mat["c"][:100])
			add_file(patient[0] + "_2", patient[1], mat["c"][-100:])
