
def transpose_matrix(a: list[list[int|float]]) -> list[list[int|float]]:
    m,n=len(a),len(a[0])
    b=[[0]*m for _ in range(n)]

    for i in range(m):
        for j in range(n):
            b[j][i]=a[i][j]

    
	return b