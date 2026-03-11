import numpy as np

from self_attention.self_attention import self_attention
from layer_norm.layer_norm import layer_norm
from ffn.feed_forward import feed_forward

def encoder_layer(
    X,
    WQ, WK, WV,
    W1, b1, W2, b2,
    gamma1, beta1,
    gamma2, beta2
):

    # Self Attention
    attention_output, attention_weights = self_attention(X, WQ, WK, WV)

    # Residual
    residual_1 = X + attention_output

    # LayerNorm
    norm_1 = layer_norm(residual_1, gamma1, beta1)

    # Feed Forward
    ffn_output = feed_forward(norm_1, W1, b1, W2, b2)

    # Residual 
    residual_2 = norm_1 + ffn_output

    # LayerNorm
    encoder_output = layer_norm(residual_2, gamma2, beta2)

    return encoder_output, attention_weights