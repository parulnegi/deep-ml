import numpy as np 
def min_max(x: list[float]) -> list[float]:
    """
    Perform Min-Max normalization to scale values to [0, 1].
    
    Args:
        x: A list of numerical values
    
    Returns:
        A new list with values normalized to [0, 1]
    """
    
    x= np.array(x)
    maxval, minval = np.max(x), np.min(x)
    x_norm = (x-minval)/ (maxval - minval)
    return x_norm