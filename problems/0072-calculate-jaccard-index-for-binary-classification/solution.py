
import numpy as np

def jaccard_index(y_true, y_pred):
	a=0
	b=0
	ab=0
	for i in range(len(y_true)):
		if y_true[i]==1:
			a+=1
		if y_pred[i]==1:
			b+=1
		if y_true[i]==1 and y_pred[i]==1:
			ab+=1
		
	dem=a+b-ab
	result=0
	if dem==0:
		return result
	else:
		result=ab/dem
		return round(result, 3)
