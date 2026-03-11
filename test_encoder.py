import numpy as np
from encoder import encoder_layer

tokens = 10
d_model = 64
d_ff = 256

# Entrada simulada (embeddings)
X = np.random.randn(tokens, d_model)

# Matrizes de atenção
WQ = np.random.randn(d_model, d_model)
WK = np.random.randn(d_model, d_model)
WV = np.random.randn(d_model, d_model)

# Feed Forward
W1 = np.random.randn(d_model, d_ff)
b1 = np.zeros(d_ff)

W2 = np.random.randn(d_ff, d_model)
b2 = np.zeros(d_model)

# LayerNorm parameters
gamma1 = np.ones(d_model)
beta1 = np.zeros(d_model)

gamma2 = np.ones(d_model)
beta2 = np.zeros(d_model)

output, attention_weights = encoder_layer(
    X,
    WQ, WK, WV,
    W1, b1, W2, b2,
    gamma1, beta1,
    gamma2, beta2
)

print("Input shape:", X.shape)
print("Output shape:", output.shape)
print("Attention weights shape:", attention_weights.shape)