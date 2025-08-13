
from collections import Counter

def confusion_matrix(data):
    tp=0
    fp=0
    tn=0
    fn=0
    for tl,pl in data:
        if tl==1 and pl==1:
            tp+=1
        elif tl==1 and pl==0:
            fn+=1
        elif tl==0 and pl==1:
            fp+=1
        else:
            tn+=1

	return [[tp,fn],[fp,tn]]
