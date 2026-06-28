import numpy as np

def gradient_descent(X, y, weights, learning_rate, n_epochs, batch_size=1, method='batch'):
    """
    Perform gradient descent optimization.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights
    """
    m,n=X.shape
    for _ in range(n_epochs):
        if method=='batch':
            y_pred= X @ weights
            new_weights = ((X.T @ (y - y_pred)) * 2)/m 
            weights+= learning_rate * new_weights
        elif method=='stochastic':
            for idx in range(m):
                y_pred= X @ weights
                new_weights= (X[idx] * (y[idx] - y_pred[idx])) * 2
                weights+= learning_rate * new_weights
        else:
            for idx in range(0,m,batch_size):
                y_pred= X @ weights
                end=min(idx + batch_size, m)
                new_weights =((X[idx:end, : ].T @ (y[idx:end] - y_pred[idx:end]))*2) / batch_size
                weights+= learning_rate * new_weights
        

    return weights

    

         


