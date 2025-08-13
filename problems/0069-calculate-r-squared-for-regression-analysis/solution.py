
import numpy as np

def r_squared(y_true, y_pred):
	mean_actual=np.mean(y_true)
	ssd=np.sum((y_true-y_pred)**2)
	sst=np.sum((y_true-mean_actual)**2)
	x=ssd/sst

	return np.round((1-x),3)
	
