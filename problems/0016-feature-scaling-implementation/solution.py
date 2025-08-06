import numpy as np
def feature_scaling(data: np.ndarray) -> (np.ndarray, np.ndarray):

	m,n=data.shape
	standardized_data=np.zeros((m,n)) 
	normalized_data=np.zeros((m,n))
	mean=np.mean(data,axis=0)
	std=np.std(data,axis=0)
	standardized_data = (data-mean)/std
	minval=np.min(data, axis=0)
	maxval=np.max(data,axis=0)
	normalized_data=(data-minval)/(maxval-minval)




	return np.round(standardized_data,4).tolist(), np.round(normalized_data,4).tolist()