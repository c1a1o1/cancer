# save slices on one file per patient in tfrecords

import numpy as np
import matplotlib.pyplot as plt
import dicom
import os
import tensorflow as tf
from scipy import linalg

def princomp(A, numpc=0):
	# computing eigenvalues and eigenvectors of covariance matrix
	M = (A - np.mean(A.T, axis=1)).T # subtract the mean (along columns)
	#[eigenvalues, eigenvectors] = np.linalg.eig(np.cov(M))
	[eigenvalues, eigenvectors] = linalg.eig(np.cov(M))
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
	image = images[1]
	#image[image == -2000] = 0
	plt.imshow(image, cmap=plt.cm.gray)
	plt.show()
	z()

	new_dir = directory + '/' + group + '_' + str(npc) + '/'
	print new_dir
	if not os.path.exists(new_dir):
		os.makedirs(new_dir)

	writer = tf.python_io.TFRecordWriter(new_dir + '/' + patient + ".tfrecords")

	label = 1
	for img in images:
		eigenvectors, eigenvalues, projection = princomp(img, npc)

        example = tf.train.Example(
            features=tf.train.Features(
              feature={
                'label': tf.train.Feature(
                    int64_list=tf.train.Int64List(value=label)),
                'image': tf.train.Feature(
                    float_list=tf.train.FloatList(value=projection)),
                'vectors': tf.train.Feature(
                    float_list=tf.train.FloatList(value=eigenvectors)),
        }))

        writer.write(example.SerializeToString())

	writer.close()
	z()
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
	

