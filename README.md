# Transformer Encoder From Scratch (NumPy)

Este projeto implementa **os principais componentes de um Transformer Encoder** a partir do zero utilizando **Python + NumPy**, inspirado no paper:

**Vaswani et al., 2017 — "Attention Is All You Need"**

O objetivo do projeto é **didático**: entender passo a passo como funcionam as partes internas de um Transformer.

---

# Estrutura do Projeto

IA/
│
├── transformer/
│
│ ├── encoder.py
│
│ ├── self_attention/
│ │ ├── softmax.py
│ │ ├── scaled_dot_product_attention.py
│ │ ├── self_attention.py
│ │ └── test_self_attention.py
│
│ ├── layer_norm/
│ │ ├── layer_norm.py
│ │ └── test_layer_norm.py
│
│ ├── ffn/
│ │ ├── feed_forward.py
│ │ └── test_feed_forward.py
│
├── test_encoder.py
│
├── requirements.txt
└── README.md


Cada diretório contém **uma parte do Transformer** e **um teste isolado** para validar seu funcionamento.

---

# Dependências

O projeto usa apenas **NumPy**.

Instalação:

```bash
pip install -r requirements.txt
```
        ou
```bash
pip install numpy
```

**Como Executar os Testes**
Sempre execute os scripts a partir da raiz do projeto


**1 — Teste do Self Attention**

Arquivo:
transformer/self_attention/test_self_attention.py

Esse script testa:
softmax,
scaled dot product attention,
self attention

Executar:
```bash
python transformer/self_attention/test_self_attention.py
```

O script utiliza matrizes de entrada definidas no próprio arquivo para verificar o funcionamento do mecanismo de atenção.

**2 — Teste do Layer Normalization**

Arquivo:
transformer/layer_norm/test_layer_norm.py

Esse script testa:
layer norm

Executar:
```bash
python transformer/layer_norm/test_layer_norm.py
```

Esse teste verifica o comportamento da Layer Normalization aplicada por token.

**3 — Teste da Feed Forward Network**

Arquivo:
transformer/ffn/test_feed_forward.py

Executar:
```bash
python transformer/ffn/test_feed_forward.py
```

Essa camada implementa:
FFN(x) = max(0, xW1 + b1)W2 + b2

**4 — Teste do Encoder Completo**

Arquivo:
test_encoder.py

Esse script executa um Encoder Layer completo, combinando:

Self Attention,
Residual Connection,
LayerNorm,
Feed Forward,
Residual Connection,
LayerNorm

Configuração utilizada no teste:
tokens = 10
d_model = 64
d_ff = 256

Executar:
```bash
python test_encoder.py
```

Saída esperada:
Input shape: (10, 64)
Output shape: (10, 64)
Attention weights shape: (10, 10)

**Fluxo do Encoder Layer**
Input
 │
 ▼
Self Attention
 │
 ▼
Add (Residual)
 │
 ▼
LayerNorm
 │
 ▼
Feed Forward
 │
 ▼
Add (Residual)
 │
 ▼
LayerNorm
 │
 ▼
Output

Objetivo do Projeto

Este projeto demonstra como construir um Transformer Encoder manualmente, sem frameworks de deep learning, permitindo compreender:

mecanismo de atenção,
softmax aplicado à atenção,
normalização por camada,
redes feed-forward,
conexões residuais,
fluxo interno do encoder

**Referência**

Vaswani et al. (2017)
Attention Is All You Need

https://arxiv.org/abs/1706.03762