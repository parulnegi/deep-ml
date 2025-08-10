
import numpy as np
from collections import Counter
def gini_impurity(y):
	"""
	Calculate Gini Impurity for a list of class labels.

	:param y: List of class labels
	:return: Gini Impurity rounded to three decimal places
	"""
	labels=Counter(y)
    prob=0
    for k,v in labels.items():
        prob+=(v/len(y))**2
    return round((1-prob),3)

	