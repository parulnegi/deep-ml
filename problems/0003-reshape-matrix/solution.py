import numpy as np

def reshape_matrix(a: list[list[int|float]], new_shape: tuple[int, int]) -> list[list[int|float]]:
	#Write your code here and return a python list after reshaping by using numpy's tolist() method
    m,n=len(a),len(a[0])
    m_,n_=new_shape[0],new_shape[1]
    reshaped_matrix=[]
    if m*n== m_*n_:
        reshaped_matrix=np.reshape(a, new_shape)
        return reshaped_matrix.tolist()
    else:
        return reshaped_matrix