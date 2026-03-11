import numpy as np

def layer_norm(X, gamma, beta, epsilon=1e-6):

    mean = np.mean(X, axis=-1, keepdims=True)

    variance = np.var(X, axis=-1, keepdims=True)

    normalized_X = (X - mean) / np.sqrt(variance + epsilon)

    output = gamma * normalized_X + beta

    return output