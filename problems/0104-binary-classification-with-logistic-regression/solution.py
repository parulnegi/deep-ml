import numpy as np

def predict_logistic(X: np.ndarray, weights: np.ndarray, bias: float) -> np.ndarray:
	"""
	Implements binary classification prediction using Logistic Regression.

	Args:
		X: Input feature matrix (shape: N x D)
		weights: Model weights (shape: D)
		bias: Model bias

	Returns:
		Binary predictions (0 or 1)
	"""
	z=np.exp(-(np.dot(X,weights)+bias))
	y_pred=1/(1+z)
	ans=[]
	for i in y_pred:
		if i>=0.5:
			ans.append(1)
		else:
			ans.append(0)
	ans=np.array(ans)
	return ans

	
