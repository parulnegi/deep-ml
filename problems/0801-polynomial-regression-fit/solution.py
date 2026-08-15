import numpy as np

def fit_polynomial(x, y, degree):
    """
    Fit a polynomial of the given degree to (x, y) by least squares.

    Args:
        x: list/array of input values, length n
        y: list/array of target values, length n
        degree: non-negative integer, the polynomial degree

    Returns:
        List of coefficients [c_0, c_1, ..., c_degree] in increasing power order.
    """
    x= np.array(x).reshape(-1,1)
    m,n=x.shape
    x= x ** np.arange(1, degree+1)
    x= np.hstack((np.ones((m,1)),x))
    y=np.array(y)
    coff= np.zeros((degree+1,1))
    first= np.linalg.inv( x.T @ x)
    sec= x.T @ y
    coff= first @ sec

    return coff.tolist()


