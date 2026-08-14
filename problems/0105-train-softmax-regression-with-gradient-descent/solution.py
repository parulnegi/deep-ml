import numpy as np

def train_softmaxreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	m,n=X.shape
	classes= int(max(y)+1)
	X= np.hstack([np.ones((m,1)), X])
	Y= np.eye(m,classes)[y]
	losses=[]

	weights= np.zeros((classes, n+1))



	for _ in range(iterations):
		pred_y= X @ weights.T
		pred_y= pred_y - np.max(pred_y, axis=1, keepdims=True)
		sigma= np.exp(pred_y)/np.sum(np.exp(pred_y), axis=1, keepdims=True)
	
		loss= -np.sum( Y * np.log(sigma))
		weights = weights - learning_rate * ((X.T @ (sigma - Y))).T

		losses.append(loss)
	return (weights, losses)
		 





