import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    n,k = Q.shape
    qk = (Q @ K.T)/ np.sqrt(k)
    values = np.exp(qk  - (np.max(qk, axis = 1, keepdims = True)))
    dem = np.sum(np.exp(qk - (np.max(qk, axis = 1, keepdims =True))), axis =1, keepdims=True)
    softmax = values / dem 
    attn_output = softmax @ V
    
    return attn_output
