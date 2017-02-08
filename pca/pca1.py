import numpy as np


def reduce_dim(x, k):
	sigma = np.corrcoef(x, rowvar=0)
	u,s,v = np.linalg.svd(sigma, full_matrices=False)
	ureduce = u[:, 1:k]
	z = np.transpose(ureduce) * x

	return z

# first do normalization zero mean and reduce features
print reduce_dim([[1,0],[0,1]], 1)