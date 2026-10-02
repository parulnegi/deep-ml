import numpy as np

def compute_tf_idf(corpus, query):
	"""
	Compute TF-IDF scores for a query against a corpus of documents.
    
	:param corpus: List of documents, where each document is a list of words
	:param query: List of words in the query
	:return: List of lists containing TF-IDF scores for the query words in each document
	"""
	if len(corpus) ==0  or len(query) == 0:
		return []

	scores  = []
	dfs ={}
	for i, word in enumerate(query):
		fs = sum(1 for doc in corpus if word in doc)
		idf = np.log( (len(corpus) + 1) / (fs + 1)) + 1
		dfs[word] = idf

	for doc in corpus:
		l = len(doc)
		score = [0]*len(query)
		if l>0:
			for i, w in enumerate(query):
				count = doc.count(w) / l
				score[i] = round(count*dfs[w], 5)
		
		scores.append(score)
	return scores

