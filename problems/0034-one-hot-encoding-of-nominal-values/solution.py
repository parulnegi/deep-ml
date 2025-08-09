import numpy as np

def to_categorical(x, n_col=None):
	new_arr=np.array(x)
    new_arr=list(set(new_arr))
    new_arr.sort()
    m=len(x)
    n_col=new_arr[-1]
    one_hot=np.zeros((m,n_col+1))
	for i in range(m):
        one_hot[i][x[i]]=1
    return one_hot

