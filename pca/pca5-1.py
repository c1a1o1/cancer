import numpy as np
import matplotlib.pyplot as plt


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

A = plt.imread('../shakira.jpg') # load an image
print "A rgb", A.shape, A.size

A = np.mean(A,2) # to get a 2-D array
print "A gray", A.shape, A.size

full_pc = np.size(A, axis=1) # numbers of all the principal components
print "full_pc", full_pc

i = 1
dist = []
for numpc in range(0,full_pc+10,10): # 0 10 20 ... full_pc
	if numpc <= 50:
		eigenvectors, eigenvalues, projection = princomp(A, numpc)
		Ar = np.dot(eigenvectors, projection).T + np.mean(A, axis=0) # image reconstruction
		print
		print "reconstructed", numpc, Ar.shape, Ar.size
		print "eigenvectors", eigenvectors.shape, eigenvectors.size
		print "projection", projection.shape, projection.size

plt.figure()
perc = np.cumsum(eigenvalues) / sum(eigenvalues)
plt.plot(range(len(perc)), perc, 'b')
plt.axis([0, full_pc, 0, 1.1])
plt.show()



