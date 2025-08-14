import numpy as np

def convert_range(values: np.ndarray, c: float, d: float) -> np.ndarray:
    """
    Shift and scale values from their original range [min, max] to a target [c, d] range.
    """
    # if values.ndim==1:
    a,b=np.min(values),np.max(values)              
    x= c + ((d-c)/(b-a)) * (values-a)        
    # else:
    #     x=[]
    #     for row in values:
    #         a,b=np.min(row),np.max(row)
    #         val=c + ((d-c)/(b-a)) * (row-a)
    #         x.append(val)
    #     x=np.array(x)

    return x
