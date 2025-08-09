import numpy as np

def batch_iterator(X, y=None, batch_size=64):
    batch=[]
    m,n=X.shape
    if y is not None:
        for i in range(0,m,batch_size):
            if i+batch_size<m:
                xi,yi=X[i:i+batch_size,: ],y[i:i+batch_size]
                batch.append([[xi],yi])
            else:
                xi,yi=X[i: , : ],y[i:]
                batch.append([[xi],yi])
    else:
        for i in range(0,m,batch_size):
            if i+batch_size<m:
                xi=X[i:i+batch_size,:]
                batch.append([xi])
            else:
                xi=X[i: ,:]
                batch.append([xi])

    
    return batch
