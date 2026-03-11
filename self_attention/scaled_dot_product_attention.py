import numpy as np

from self_attention.softmax import row_softmax

def scaled_dot_product_attention(queries, keys, values):

    attention_logits = np.matmul(queries, keys.T)

    key_vector_dimension = keys.shape[1]

    scaled_attention_logits = attention_logits / np.sqrt(key_vector_dimension)

    attention_weights = row_softmax(scaled_attention_logits)

    self_attention_output = np.matmul(attention_weights, values)

    return self_attention_output, attention_weights