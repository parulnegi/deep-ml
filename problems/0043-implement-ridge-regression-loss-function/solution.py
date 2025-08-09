import numpy as np

def ridge_loss(X: np.ndarray, w: np.ndarray, y_true: np.ndarray, alpha: float) -> float:

	loss=np.sum((y_true-np.dot(X,w))**2)
    final_loss= loss/len(y_true)+ (alpha*np.sum(w**2))
	return final_loss
