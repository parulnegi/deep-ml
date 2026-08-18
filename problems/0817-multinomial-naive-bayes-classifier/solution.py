import numpy as np

def multinomial_naive_bayes(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """
    Implements Multinomial Naive Bayes classifier.

    Args:
        X_train: Training count features (shape: N_train x D)
        y_train: Training labels (shape: N_train)
        X_test: Test count features (shape: N_test x D)
        alpha: Laplace smoothing parameter

    Returns:
        Predicted class labels for X_test (shape: N_test)
    """
    classes, counts= np.unique(y_train, return_counts = True)
    prior = {cls : np.log(counts[i]/ len(y_train)) for i,cls in enumerate(classes)}

    likelihood={}
    for cls in classes:
        sum_vals= np.sum(X_train[y_train==cls], axis=0)
        prob = (sum_vals + alpha)/ (np.sum(sum_vals) + alpha * X_train.shape[1])
        likelihood[cls]= np.log(prob)

    results= []
    for cls in classes:
        posteries = X_test @ likelihood[cls].T
        results.append( posteries + prior[cls])
    
    results = np.array(results).T

    index=np.argmax(results, axis=1)

    return classes[index]


    



    

    





