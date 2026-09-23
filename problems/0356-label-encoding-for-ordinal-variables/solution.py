import numpy as np
def label_encode_ordinal(values: list, order: list) -> list:
    """
    Encode ordinal categorical values to integers based on specified order.
    
    Args:
        values: List of categorical values to encode
        order: List specifying the order of categories from lowest (0) to highest
    
    Returns:
        List of integers representing the encoded values
    """
    values = np.array(values)
    order = np.array(order)
    answer=[]
    for i in range(len(values)):
        val =-1
        if values[i] in order:
            val = np.where(order== values[i])[0][0]
        answer.append(val)

    return answer
