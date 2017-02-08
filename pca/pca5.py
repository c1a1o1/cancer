#from numpy import mean, cov, cumsum, dot, linalg, size, flipud, argsort
#from pylab import imread, subplot, imshow, title, gray, figure, show, NullLocator, imsave
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
A = np.mean(A, 2) # to get a 2-D array
full_pc = np.size(A, axis=1) # numbers of all the principal components
i = 1 # subplots
dist = []
for numpc in range(0, full_pc+10, 10): # 0 10 20 ... full_pc
	# showing the pics reconstructed with less than 50 PCs
	if numpc <= 50:
		eigenvectors, eigenvalues, projection = princomp(A, numpc)
		Ar = np.dot(eigenvectors, projection).T + np.mean(A, axis=0) # image reconstruction

		# difference in Frobenius norm
		dist.append(np.linalg.norm(A-Ar, 'fro'))

		ax = plt.subplot(2, 3, i, frame_on=False)
		ax.xaxis.set_major_locator(plt.NullLocator()) # remove ticks
		ax.yaxis.set_major_locator(plt.NullLocator())
		i += 1 
		plt.imshow(Ar)
		plt.title('PCs # ' + str(numpc))
		plt.gray()

	if numpc == 50:
		A50 = Ar

plt.figure()
plt.imshow(A)
plt.title('numpc FULL')
plt.gray()

#imsave("shakira40.jpg", A50)
plt.figure()
plt.imshow(A50)
plt.title('numpc 50')
plt.gray()

plt.figure()
perc = np.cumsum(eigenvalues) / sum(eigenvalues)
dist = dist / np.max(dist)
plt.plot(range(len(perc)), perc, 'b')
#plt.plot(range(len(dist)), dist, 'r')
plt.axis([0, full_pc, 0, 1.1])
plt.show()
