import numpy as np

def train_logreg(X: np.ndarray, y: np.ndarray, learning_rate: float, iterations: int) -> tuple[list[float], ...]:
	"""
	Gradient-descent training algorithm for logistic regression, optimizing parameters with Binary Cross Entropy loss.
	"""
	n,d = X.shape
	losses=[]
	weights = np.zeros(d+1)
	X = np.hstack([np.ones((n,1)), X])
	for _ in range(iterations):
		z= X @ weights
		sig = 1/ (1 + np.exp( -z))
		loss= (np.sum(y* np.log(sig) + (1-y)*np.log(1-sig)))
		losses.append(-loss)
		weights= weights - learning_rate * (X.T @ (sig - y))
	
	return np.round(weights, 4) , np.round(losses, 4)



