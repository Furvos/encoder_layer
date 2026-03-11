import numpy as np
from self_attention import self_attention

X = np.array([
    [1,0,1],
    [0,2,0],
    [1,1,1]
])

WQ = np.random.randn(3,2)
WK = np.random.randn(3,2)
WV = np.random.randn(3,2)

output, weights = self_attention(X, WQ, WK, WV)

print("Attention Weights = ")
print(weights)

print("\nSelf Attention Output = ")
print(output)