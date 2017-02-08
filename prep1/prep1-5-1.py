# read slices from files .mat (vec, val and proj ) and plot

import numpy as np
import matplotlib.pyplot as plt
import dicom
import os
import scipy.io

directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer"
group = "stage 1 samples_30"
path = directory + '/' + group 
patient = "00cba091fa4ad62cc3200a657aeb957e"

mat = scipy.io.loadmat(path + '/' + patient + "_vec")
vec = mat["vec"]
mat = scipy.io.loadmat(path + '/' + patient + "_proj")
proj= mat["proj"]
print proj.shape
n,_,_ = proj.shape
print n
for i in xrange(n):
	Ar = np.dot(vec[i], proj[i]).T
	ax = plt.subplot(11, 15, i+1, frame_on=False)
	ax.xaxis.set_major_locator(plt.NullLocator()) # remove ticks
	ax.yaxis.set_major_locator(plt.NullLocator())
	plt.gray()
	plt.imshow(Ar)
	#break

#plt.show()

	
'''
	i = 1
	for img in images:
		eigenvectors, eigenvalues, projection = princomp(img, npc)
		Ar = np.dot(eigenvectors, projection).T + np.mean(img, axis=0)
		plt.gray()
		plt.imsave(new_dir + '/' + str(i) + '.jpg', Ar)
		i += 1
		print i-1
'''
	

