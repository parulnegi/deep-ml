import numpy as np

def svd_2x2_singular_values(A: np.ndarray) -> tuple:
    """
    Compute SVD of a 2x2 matrix using one Jacobi rotation.
    
    Args:
        A: A 2x2 numpy array
    
    Returns:
        Tuple (U, S, Vt) where A ≈ U @ diag(S) @ Vt
        - U: 2x2 orthogonal matrix
        - S: length-2 array of singular values
        - Vt: 2x2 orthogonal matrix (transpose of V)
    """
    ata=np.matmul(A.T,A)
    a=ata[0][0]
    b=ata[0][1]
    d=ata[1][1]
    if a==d:
        theta= np.pi /4
    else:
        theta= 0.5 * np.arctan2(2*b, a-d)

    R=np.array([[np.cos(theta), -1*np.sin(theta)],
                [np.sin(theta),np.cos(theta)]])

    s= R.T @ ata @ R
    sign_val= np.sqrt(np.diag(s))
    sig_inv=np.diag([1/val  if val>0 else 0 for val in sign_val ])
    
    
    U= A @ R @ sig_inv

    return (U, sign_val, R.T)



    
  