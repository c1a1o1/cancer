# save slices on one .mat file per patient

import numpy as np
import matplotlib.pyplot as plt
import dicom
import os
import scipy.io, scipy
import kronos
from scipy import linalg as LA
import pandas as pd

def princomp(A, numpc=0):
	# computing eigenvalues and eigenvectors of covariance matrix
	M = (A - np.mean(A.T, axis=1)).T # subtract the mean (along columns)
	[eigenvalues, eigenvectors] = scipy.linalg.eig(np.cov(M))
	p = np.size(eigenvectors, axis=1)
	idx = np.argsort(eigenvalues) # sorting the eigenvalues
	idx = idx[::-1]       # in ascending order
	# sorting eigenvectors according to the sorted eigenvalues
	eigenvectors = eigenvectors[:, idx]
	eigenvalues = eigenvalues[idx] # sorting eigenvalues
	projection = 0
	if numpc < p and numpc >= 0:
		eigenvectors = eigenvectors[:, range(numpc)] # cutting some PCs if needed
		projection = np.dot(eigenvectors.T, M) # projection of the data in the new space
	return eigenvectors, eigenvalues, projection

def get_pixels_hu(slices):
    image = np.stack([s.pixel_array for s in slices])
    # Convert to int16 (from sometimes int16), 
    # should be possible as values should always be low enough (<32k)
    image = image.astype(np.int16)

    # Set outside-of-scan pixels to 0
    # The intercept is usually -1024, so air is approximately 0
    image[image == -2000] = 0
    
    # Convert to Hounsfield units (HU)
    for slice_number in range(len(slices)):
        
        intercept = slices[slice_number].RescaleIntercept
        slope = slices[slice_number].RescaleSlope
        
        if slope != 1:
            image[slice_number] = slope * image[slice_number].astype(np.float64)
            image[slice_number] = image[slice_number].astype(np.int16)
            
        image[slice_number] += np.int16(intercept)
    
    return np.array(image, dtype=np.int16)

def save_patient(patient, path):
	slices = [dicom.read_file(path + '/' + patient + '/' + s) for s in os.listdir(path + '/' + patient)]
	slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
	images = get_pixels_hu(slices)
	#images = np.stack([s.pixel_array for s in slices])
	#image = images[0]
	#image[image == -2000] = 0
	#plt.imshow(image, cmap=plt.cm.gray)
	#plt.show()
	#z()

	new_dir = directory + '/' + group + '_' + str(npc) + '/'
	print new_dir
	if not os.path.exists(new_dir):
		os.makedirs(new_dir)

	label = 1
	c = []
	for img in images:
		eigenvectors, eigenvalues, projection = princomp(img, npc)
		c.append((eigenvectors, eigenvalues, projection, np.mean(img, axis=0)))
		print "princomp", k.elapsed()

	a = {}
	a["c"] = c
	scipy.io.savemat(new_dir + patient, a, do_compression=True)


k = kronos.krono()
npc = 100
directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer"
group = "stage 1 samples"
group = "stage1"
path = directory + '/' + group 
#patients = pd.read_csv('patients_carlos.csv')
patients = pd.read_csv('patients_roma.csv')
#patients = pd.read_csv('patients.csv')
patients = patients.values.tolist()
#print patients[1].tolist()

#for patient in os.listdir(path):
for patient in patients:
	name = patient[0]
	print name
	save_patient(name, path)
