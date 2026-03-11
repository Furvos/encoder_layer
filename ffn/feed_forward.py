import numpy as np

def relu(X):
    return np.maximum(0, X)

def feed_forward(X, W1, b1, W2, b2):

    hidden = np.matmul(X, W1) + b1

    hidden_activated = relu(hidden)

    output = np.matmul(hidden_activated, W2) + b2

    return output