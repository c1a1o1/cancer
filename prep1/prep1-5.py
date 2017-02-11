# read slices from files .mat (vec, val and proj ) and plot

import numpy as np
import matplotlib.pyplot as plt
import dicom
import os
import scipy.io

directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer"
group = "stage 1 samples_100"
path = directory + '/' + group 
patient = "00cba091fa4ad62cc3200a657aeb957e"
patient = "0a099f2549429d29b32f349e95fb2244"

mat = scipy.io.loadmat(path + '/' + patient)
n,m = mat['c'].shape
print n,m
i = 1
for slice in mat['c']:
	Ar = np.dot(slice[0], slice[2]).T + slice[3]
	Ar[Ar==2000] = 0
	#ax = plt.subplot(11, 15, i, frame_on=False)
	#ax.xaxis.set_major_locator(plt.NullLocator()) # remove ticks
	#ax.yaxis.set_major_locator(plt.NullLocator())
	plt.gray()
	plt.title(group)
	plt.imshow(Ar)
	i+=1
	break

plt.show()
