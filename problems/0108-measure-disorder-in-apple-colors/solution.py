from collections import Counter
def disorder(apples: list) -> float:
	"""
	Compute the disorder in a basket of apples.
	"""
	hashmap=Counter(apples)
    maxdiff=float("-inf")
	for k,v in hashmap.items():
        maxdiff=max(maxdiff,v/len(apples))
    
    return 1-maxdiff
