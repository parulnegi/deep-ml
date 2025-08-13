
import numpy as np

def rmse(y_true, y_pred):
	if y_true.shape==y_pred.shape:
		
		mse=np.mean((y_true-y_pred)**2)
		rmse_res=mse**0.5
		return round(rmse_res,3)

	else:
		return 0


	
	
