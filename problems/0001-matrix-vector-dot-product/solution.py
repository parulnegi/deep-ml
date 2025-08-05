import numpy as np
def matrix_dot_vector(a: list[list[int|float]], b: list[int|float]) -> list[int|float]:
	# Return a list where each element is the dot product of a row of 'a' with 'b'.
	# If the number of columns in 'a' does not match the length of 'b', return -1.
	
    m,n=len(a), len(a[0])
    answer=[]
    if n==len(b):
        for i in range(m):
            val= np.dot(a[i],b)
            answer.append(val)
        return answer
    else:
        return -1
