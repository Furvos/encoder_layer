import numpy as np

def row_softmax(logits_matrix, axis=-1):
    stabilized_logits = logits_matrix - np.max(logits_matrix, axis=axis, keepdims=True)
    exponentials = np.exp(stabilized_logits)
    return exponentials / np.sum(exponentials, axis=axis, keepdims=True)
