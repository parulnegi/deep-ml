import numpy as np
import math
def train_neuron(features: np.ndarray, labels: np.ndarray, initial_weights: np.ndarray, initial_bias: float, learning_rate: float, epochs: int) -> (np.ndarray, float, list[float]):

	mse_values=[]
	for _ in range(epochs):
		prediction = modelPredict(features, initial_weights, initial_bias)
		mse= mseloss(prediction,labels)
		mse_values.append(mse)
		initial_weights, initial_bias = backprop(prediction,labels,features,learning_rate,initial_weights, initial_bias)

	return np.round(initial_weights,4), np.round(initial_bias,4), np.round(mse_values,4)

def backprop(prediction,labels,features,learning_rate,initial_weights,initial_bias):
	new_weight=[0]*len(initial_weights)
	new_bias=0.0
	for weight in range(len(new_weight)):
		grad=gradient(labels, prediction , features[:, weight])
		new_weight[weight]= initial_weights[weight] - learning_rate* grad
	
	feature_array= np.ones(len(labels))
	grad_len= gradient(labels , prediction , feature_array )
	new_bias= initial_bias - learning_rate*grad_len

	return new_weight, new_bias

def gradient (labels, predicted, feature):
	diff= labels - predicted
	sigdev = -(predicted * ( 1- predicted)) * feature

	return sum(diff * sigdev) * 2 / len( labels)

def sigmoid(x):
	 return 1/ ( 1+ math.exp(-x))

def modelPredict(features, weights, bias):
	probabilites=[]
	for ft in features:
		x= (ft.T @ weights) + bias
		z= sigmoid(x)
		probabilites.append(z)

	return np.array(probabilites)

def mseloss(predicted, truelabel):
	return (sum((truelabel - predicted)**2))/ len(truelabel)




