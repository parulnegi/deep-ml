import numpy as np

def bhattacharyya_distance(p: list[float], q: list[float]) -> float:
    if not p and not q :
        return 0.0
    if len(p)!=len(q):
        return 0.0
    p=np.array(p)
    q=np.array(q)
    bc=np.sum((p*q)**0.5)
    bd=-(np.log(bc))
    return round(bd,4)