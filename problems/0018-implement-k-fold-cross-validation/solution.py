import numpy as np

def k_fold_cross_validation(X: np.ndarray, y: np.ndarray, k=5, shuffle=True):
    """
    Implement k-fold cross-validation by returning train-test indices.
    """
    n=len(X)
    ind=np.arange(n)

    if shuffle:
        np.random.shuffle(ind)
    
    data=np.array_split(ind,k)
    result=[]
    for i in range(k):
        test=data[i]
        train=np.concatenate([data[j] for j in range(k) if i!=j])
        result.append((train.tolist(),test.tolist()))
   

    return result

