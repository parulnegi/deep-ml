import numpy as np

def shuffle_data(X, y, seed=None):
	np.random.seed(seed)
	m,n=X.shape
	s=np.arange(m)
	np.random.shuffle(s)	
	
	return X[s],y[s]

	