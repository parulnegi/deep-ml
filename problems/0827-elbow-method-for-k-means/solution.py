import numpy as np

def elbow_wcss(X: np.ndarray, k_values: list, max_iters: int = 100) -> list:
    """
    Compute WCSS (inertia) for each k in k_values using K-Means.

    Args:
        X: Data of shape (n_samples, n_features)
        k_values: List of cluster counts to evaluate
        max_iters: Maximum number of Lloyd iterations

    Returns:
        List of WCSS values (rounded to 4 decimals), one per k
    """
    
    finalsum = []
    for k in k_values:
        initital_c = X[:k,:].copy()
        
        for itr in range(max_iters):
            distance = [eucDistance(X, center) for center in initital_c]
            index = np.argmin(distance, axis=0)
            dist = 0
            for val in range(k):
                points = X[index == val]
                if len(points)>0:    
                    if itr == max_iters-1:
                        dist+=wcss(points, initital_c[val]) 

                    initital_c[val] = np.mean(points ,axis = 0)
        finalsum.append(round(float(dist),4))               
                
    return finalsum     



def eucDistance(x, centroids):
    return np.sqrt(np.sum((x - centroids)**2 , axis = 1))

def wcss(x, center):
    return np.sum((x - center)**2)
