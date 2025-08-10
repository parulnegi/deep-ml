import numpy as np

def f_score(y_true, y_pred, beta):
	"""
	Calculate F-Score for a binary classification task.

	:param y_true: Numpy array of true labels
	:param y_pred: Numpy array of predicted labels
	:param beta: The weight of precision in the harmonic mean
	:return: F-Score rounded to three decimal places
	"""
	actual_true=np.sum(y_true==1)
    pred_true=np.sum(y_pred==1)
    tp=0
    for i in range(len(y_true)):
        if y_true[i]==1 and y_pred[i]==1:
            tp+=1
    
    precision=tp/pred_true
    recall=tp/actual_true

    f=((1+(beta**2))*precision*recall)/((beta**2 *precision)+recall)

    return np.round(f,3)

