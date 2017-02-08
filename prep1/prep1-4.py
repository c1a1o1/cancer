# save slices on three files .mat (vec, val and proj ) per patient

import numpy as np
import matplotlib.pyplot as plt
import dicom
import os
import scipy.io

def princomp(A, numpc=0):
	# computing eigenvalues and eigenvectors of covariance matrix
	M = (A - np.mean(A.T, axis=1)).T # subtract the mean (along columns)
	[eigenvalues, eigenvectors] = np.linalg.eig(np.cov(M))
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


npc = 30
directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer"
group = "stage 1 samples"
path = directory + '/' + group 
#patient = "4ec5ef19b52ec06a819181e404d37038"

for patient in os.listdir(path):
	slices = [dicom.read_file(path + '/' + patient + '/' + s) for s in os.listdir(path + '/' + patient)]
	images = get_pixels_hu(slices)
	#images = np.stack([s.pixel_array for s in slices])
	#image = images[1]
	#image[image == -2000] = 0
	#plt.imshow(image, cmap=plt.cm.gray)
	#plt.show()
	#z()

	new_dir = directory + '/' + group + '_' + str(npc) + '/'
	print new_dir
	if not os.path.exists(new_dir):
		os.makedirs(new_dir)

	label = 1
	vec = []
	val = []
	proj = []
	for img in images:
		eigenvectors, eigenvalues, projection = princomp(img, npc)
		vec.append(eigenvectors)
		val.append(eigenvalues)
		proj.append(projection)

	a = {}
	a["vec"] = vec
	scipy.io.savemat(new_dir + patient + '_vec', a, do_compression=True)
	a = {}
	a["val"] = val
	scipy.io.savemat(new_dir + patient + '_val', a, do_compression=True)
	a = {}
	a["proj"] = proj
	scipy.io.savemat(new_dir + patient + '_proj', a, do_compression=True)
	break
	
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
	

