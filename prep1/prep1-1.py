# choose number of principal components, 30 good, 50 very good, 100 max

import numpy as np
import matplotlib.pyplot as plt
import dicom

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

directory = "/media/carlos/CE2CDDEF2CDDD317/concursos/cancer/stage 1 samples/"
#patient = "4ec5ef19b52ec06a819181e404d37038"

dcm = directory + '0a38e7597ca26f9374f8ea2770ba870d/4ec5ef19b52ec06a819181e404d37038.dcm'
dcm = dicom.read_file(dcm)
img = dcm.pixel_array
#A[A == -2000] = 0
npc = np.size(img, axis=1) 

eigenvectors, eigenvalues, projection = princomp(img)

plt.figure()
perc = np.cumsum(eigenvalues) / sum(eigenvalues)
plt.plot(range(len(perc)), perc, 'b')
#plt.plot(range(len(dist)), dist, 'r')
plt.axis([0, npc, 0, 1.1])
plt.grid()

plt.figure()
plt.axis('off')
plt.imshow(img)
plt.gray()

plt.show()
