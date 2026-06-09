import math
import numpy as np

def single_neuron_model(features: list[list[float]], labels: list[int], weights: list[float], bias: float) -> (list[float], float):
	
	features= np.array(features)
	labels=np.array(labels)
	weights=np.array(weights)
	n=len(features)

	probabilities=[]

	for itr in range(n):
		x = (features[itr].T @ weights) + bias
		z= 1/ (1+ math.exp(-x))
		probabilities.append(z)
	

	mse= (sum((probabilities - labels)**2)) / n
		


	return probabilities, mse