import numpy as np
def precision(y_true, y_pred):
    actual_true=0
    for i in range(len(y_true)):
        if y_true[i]==1 and y_pred[i]==1:
            actual_true+=1

    predicted_true=np.sum(y_pred==1)
    return actual_true/predicted_true

