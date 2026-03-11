import numpy as np

from self_attention.scaled_dot_product_attention import scaled_dot_product_attention

def self_attention(X, WQ, WK, WV):

    queries = np.matmul(X, WQ)

    keys = np.matmul(X, WK)

    values = np.matmul(X, WV)

    self_attention_output, attention_weights = scaled_dot_product_attention(
        queries, keys, values
    )

    return self_attention_output, attention_weights