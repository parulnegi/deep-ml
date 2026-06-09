import math

def softmax(scores: list[float]) -> list[float]:
    sumval=0
    answer=[]
    maxval=max(scores)
    for val in scores:
        sumval+=math.exp(val - maxval)
    
    for val in scores:
        x = math.exp(val - maxval)
        answer.append(x/sumval)

    return answer