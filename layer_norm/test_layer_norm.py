import numpy as np

from layer_norm import layer_norm

X = np.array([
    [1.0, 2.0, 3.0],
    [2.0, 4.0, 6.0]
])

d_model = X.shape[-1]

gamma = np.ones((d_model,))
beta = np.zeros((d_model,))

output = layer_norm(X, gamma, beta)

print(output)

print("\nMean per token:")
print(np.mean(output, axis=-1))

print("\nVariance per token:")
print(np.var(output, axis=-1))