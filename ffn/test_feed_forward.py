import numpy as np
from feed_forward import feed_forward

tokens = 3
d_model = 4
d_ff = 8

X = np.random.randn(tokens, d_model)

W1 = np.random.randn(d_model, d_ff)
b1 = np.zeros(d_ff)

W2 = np.random.randn(d_ff, d_model)
b2 = np.zeros(d_model)

output = feed_forward(X, W1, b1, W2, b2)

print("Input shape:", X.shape)
print("Output shape:", output.shape)