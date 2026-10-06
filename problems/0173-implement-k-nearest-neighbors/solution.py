import numpy as np

def k_nearest_neighbors(points, query_point, k):
    """
    Find k nearest neighbors to a query point
    
    Args:
        points: List of tuples representing points [(x1, y1), (x2, y2), ...]
        query_point: Tuple representing query point (x, y)
        k: Number of nearest neighbors to return
    
    Returns:
        List of k nearest neighbor points as tuples
        When distances are tied, points appearing earlier in the input list come first.
    """
    points = np.array(points)
    query_point = np.array(query_point)
    distance = np.sum((points - query_point)**2, axis = 1)
    sortvals = np.argsort(distance, kind ='stable')


    top_k = points[sortvals[:k]]
    return [tuple(point) for point in top_k]





