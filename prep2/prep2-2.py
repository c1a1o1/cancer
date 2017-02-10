'''
Para manejar la variedad de slices entre los diferentes pacientes

1. Elegimos pacientes con slices entre 100 y doscientos
2. Cada paciente se parte en 2 grupos cada uno con 102 slices
   El primero va de 1 a 102 y el segundo de 99 a 200
3. A los dos grupos se les da el mismo label

En este script se vam a utilizar los archivos que contienen las imagenes reducidas por el pca

Ensayarenos con el formato .mat
'''

import numpy as np
import os
import scipy.io

directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer"
group = "stage 1 samples"
#group = "stage1"

old_dir = directory + '/' + group + '_100' + '/'
new_dir = directory + '/' + group + '_100_100_200' + '/'

ns = []
for patient in os.listdir(old_dir):
	mat = scipy.io.loadmat(old_dir + '/' + patient)
	nslices = len(mat['c'])
	ns.append(( patient, nslices))

	#if nslices >=100 and nslices<=200:
	#	vec = mat["vec"]




print ns