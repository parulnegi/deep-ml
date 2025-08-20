import numpy as np 
def pca(data: np.ndarray, k: int) -> np.ndarray:
	mean=np.mean(data, axis=0)
	sd=np.std(data, axis=0)
	data= (data-mean)/sd

	cov=np.cov(data.T)
	eigenval,eigenvector=np.linalg.eig(cov)
	ind=np.argsort(eigenval)[::-1]
	eigenvector=eigenvector[: , ind]
	principal_components=eigenvector[:, :k]

	return np.round(principal_components, 4)