def calculate_covariance_matrix(vectors: list[list[float]]) -> list[list[float]]:
    size=len(vectors)
    answer=[[0]*size for _ in range(size)]
    mean=[]
    
    for i in range(size):
        s=sum(vectors[i])/len(vectors[i])
        mean.append(s)

    for i in range(size):
        for j in range(size):
            val=0
            for k in range(size):
                val+=(vectors[i][k]-mean[i])*(vectors[j][k]-mean[j])
            answer[i][j]=val/(size-1)
            val=0
    return answer
