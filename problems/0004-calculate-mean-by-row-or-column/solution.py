def calculate_matrix_mean(matrix: list[list[float]], mode: str) -> list[float]:
    mean=[]
    m,n=len(matrix),len(matrix[0])
    if mode =="row":
        for i in range(m):
            s=0
            for j in range(n):
                s+=matrix[i][j]
            s=s/n
            mean.append(s)
    else:
        for i in range(n):
            s=0
            for j in range(m):
                s+=matrix[j][i]
            s=s/m
            mean.append(s)



	return mean