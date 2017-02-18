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

def save_patient(patient, path, label):
	slices = [dicom.read_file(path + '/' + patient + '/' + s) for s in os.listdir(path + '/' + patient)]
	slices.sort(key = lambda x: int(x.ImagePositionPatient[2]))
	images = get_pixels_hu(slices)

	writer = tf.python_io.TFRecordWriter(path + "/" + patient + ".tfrecords")
	for img in images:
		eigenvectors, eigenvalues, projection = princomp(img, npc)
		example = tf.train.Example(
			features=tf.train.Features(
				feature={
					'label': tf.train.Feature(
						int64_list=tf.train.Int64List(value=[label])),
					'vec': tf.train.Feature(
						float_list=tf.train.FloatList(eigenvectors.flatten().tolist())),
					'proj': tf.train.Feature(
						float_list=tf.train.FloatList(value=np.mean(img, axis=0).flatten().tolist())),
					'med': tf.train.Feature(
						float_list=tf.train.FloatList(value=projection.flatten().tolist())),
		}))
		writer.write(example.SerializeToString())
	writer.close()

def add_file_tfrecords(name, label, part):
	writer = tf.python_io.TFRecordWriter(new_dir + name + ".tfrecords")
	for c1 in part:
		example = tf.train.Example(
			features=tf.train.Features(
				feature={
					'label': tf.train.Feature(
						int64_list=tf.train.Int64List(value=[label])),
					'vec': tf.train.Feature(
						float_list=tf.train.FloatList(value=c1[0].flatten().tolist())),
					'proj': tf.train.Feature(
						float_list=tf.train.FloatList(value=c1[2].flatten().tolist())),
					'med': tf.train.Feature(
						float_list=tf.train.FloatList(value=c1[3].flatten().tolist())),
		}))
		writer.write(example.SerializeToString())


k = kronos.krono()
npc = 100
directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer"
#group = "stage 1 samples"
group = "stage1"
path = directory + '/' + group 
patients = [s for s in os.listdir(path)]

new_dir = directory + '/' + group + '_' + str(npc) + '/'
print new_dir
if not os.path.exists(new_dir):
	os.makedirs(new_dir)

labels = pd.read_csv(directory + '/' + group + '_labels.csv')
labels = labels.values.tolist()

for patient in patients:
	name = patient
	print name
	save_patient(name, new_dir)
	print "elapsed", k.elapsed()
