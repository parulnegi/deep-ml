import numpy as np

def simple_conv2d(input_matrix: np.ndarray, kernel: np.ndarray, padding: int, stride: int):
	input_height, input_width = input_matrix.shape
	kernel_height, kernel_width = kernel.shape
	padded_input_matrix=np.pad(input_matrix, pad_width=padding, constant_values=0)
	m= (input_height-kernel_height+ (2* padding))//stride + 1
	n= (input_width-kernel_width+ (2*padding))//stride + 1
	output_matrix= np.zeros((m,n))

	for i in range(m):
		for j in range(n):
			output_matrix[i][j]=conv(padded_input_matrix[i*stride:i*stride+kernel_height, j*stride:j*stride+kernel_width], kernel)
	return output_matrix


def conv(arr1,arr2):
	n=len(arr1)
	answer=0
	for i in range(len(arr1)):
		for j in range(len(arr1[0])):
			answer+= arr1[i][j]*arr2[i][j]
	return answer


    
	return output_matrix
