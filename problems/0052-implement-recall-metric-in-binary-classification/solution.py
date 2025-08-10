import numpy as np
def recall(y_true, y_pred):

    actual=np.sum(y_true==1)
    pred=0
    for i in range(len(y_true)):
        if y_true[i]==1 and y_pred[i]==1:
            pred+=1
    
    return np.round((pred/actual),3)

    
