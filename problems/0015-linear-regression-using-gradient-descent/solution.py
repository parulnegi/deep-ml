import numpy as np
def linear_regression_gradient_descent(X: np.ndarray, y: np.ndarray, alpha: float, iterations: int) -> np.ndarray:
	# Your code here, make sure to round
	m, n = X.shape
	theta = np.zeros((n, 1))
	y=y.reshape((3,1))
	for i in range(iterations):
		y_pred= X @ theta
		loss= np.sum((y_pred-y)**2)//2
		theta= theta - alpha * (X.T @ (y_pred - y))/m


	return theta