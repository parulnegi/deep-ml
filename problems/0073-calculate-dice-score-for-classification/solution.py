
import numpy as np

def dice_score(y_true, y_pred):
	tp=np.sum((y_true==1)&(y_pred==1))
    fn=np.sum((y_true==1)&(y_pred==0))
    fp=np.sum((y_true==0)&(y_pred==1))


    
    res= 2*tp/(2*tp+fp+fn) if (tp+fp+fn)!=0 else 0.0

	return round(res, 3)
