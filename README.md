# Transformer From Scratch (NumPy)

Este projeto implementa **os principais componentes de um Transformer** a partir do zero utilizando **Python + NumPy**, inspirado no paper:

**Vaswani et al., 2017 — "Attention Is All You Need"**

O objetivo do projeto é **didático**: compreender passo a passo como funcionam **Encoder, Decoder e o mecanismo de geração auto-regressiva** de modelos de linguagem.

O código evita frameworks como PyTorch ou TensorFlow para que **toda a matemática do Transformer seja visível diretamente em NumPy**.

---

# Estrutura do Projeto

```text
transformer/
│
├── encoder/
│   │
│   ├── encoder.py
│   │
│   ├── self_attention/
│   │   ├── softmax.py
│   │   ├── scaled_dot_product_attention.py
│   │   ├── self_attention.py
│   │   └── test_self_attention.py
│   │
│   ├── layer_norm/
│   │   ├── layer_norm.py
│   │   └── test_layer_norm.py
│   │
│   ├── ffn/
│   │   ├── feed_forward.py
│   │   └── test_feed_forward.py
│
├── decoder/
│   │
│   ├── causal_mask/
│   │   ├── causal_mask.py
│   │   └── test_causal_mask.py
│   │
│   ├── cross_attention/
│   │   ├── cross_attention.py
│   │   └── test_cross_attention.py
│   │
│   └── inference/
│       ├── generate.py
│       └── test_generate.py
│
├── test_encoder.py
│
├── requirements.txt
└── README.md
```

Cada diretório contém **um componente do Transformer** e **um script de teste independente** para validar sua implementação.

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

---

# Como Executar os Testes

Execute sempre os scripts **a partir da raiz do projeto**.

Recomendado usar execução como módulo:

```bash
python -m caminho.do.modulo
```

Isso garante que os **imports do pacote `transformer` funcionem corretamente**.

---

# Testes do Encoder

## 1 — Self Attention

Arquivo:

```
transformer/encoder/self_attention/test_self_attention.py
```

Esse script testa:

* Softmax
* Scaled Dot Product Attention
* Self Attention

Executar:

```bash
python -m transformer.encoder.self_attention.test_self_attention
```

O script utiliza **matrizes fictícias** para verificar o funcionamento do mecanismo de atenção.

---

## 2 — Layer Normalization

Arquivo:

```
transformer/encoder/layer_norm/test_layer_norm.py
```

Esse script testa:

* Layer Normalization

Executar:

```bash
python -m transformer.encoder.layer_norm.test_layer_norm
```

O teste verifica o comportamento da **normalização por token**.

---

## 3 — Feed Forward Network

Arquivo:

```
transformer/encoder/ffn/test_feed_forward.py
```

Executar:

```bash
python -m transformer.encoder.ffn.test_feed_forward
```

Essa camada implementa:

```
FFN(x) = max(0, xW1 + b1)W2 + b2
```

---

## 4 — Encoder Completo

Arquivo:

```
test_encoder.py
```

Esse script executa um **Encoder Layer completo**, combinando:

* Self Attention
* Residual Connection
* LayerNorm
* Feed Forward
* Residual Connection
* LayerNorm

Executar:

```bash
python test_encoder.py
```

Configuração do teste:

```
tokens = 10
d_model = 64
d_ff = 256
```

Saída esperada:

```
Input shape: (10, 64)
Output shape: (10, 64)
Attention weights shape: (10, 10)
```

---

# Testes do Decoder

O Decoder adiciona três componentes importantes ao Transformer:

* **Causal Masking**
* **Cross Attention**
* **Geração Auto-Regressiva**

---

# 5 — Causal Mask (Look-Ahead Mask)

Arquivo:

```
transformer/decoder/causal_mask/test_causal_mask.py
```

Esse teste implementa o **mascaramento causal**, que impede o modelo de olhar tokens futuros durante o treinamento.

A máscara tem formato:

```
[seq_len, seq_len]
```

Exemplo:

```
0      -inf   -inf
0       0     -inf
0       0      0
```

Executar:

```bash
python -m transformer.decoder.causal_mask.test_causal_mask
```

Esse teste demonstra que, após o **softmax**, as probabilidades de tokens futuros se tornam **0.0**.

---

# 6 — Cross Attention (Encoder → Decoder)

Arquivo:

```
transformer/decoder/cross_attention/test_cross_attention.py
```

Esse teste implementa a **atenção entre Encoder e Decoder**.

Diferente do Self Attention:

```
Query  → Decoder
Key    → Encoder
Value  → Encoder
```

Executar:

```bash
python -m transformer.decoder.cross_attention.test_cross_attention
```

Saída esperada:

```
Encoder output shape: (1, 10, 512)
Decoder state shape: (1, 4, 512)
Cross Attention output shape: (1, 4, 512)
Attention weights shape: (1, 4, 10)
```

Isso significa que **cada token do decoder olha para toda a sequência do encoder**.

---

# 7 — Simulação de Geração de Texto (Auto-Regressive Loop)

Arquivo:

```
transformer/decoder/inference/test_generate.py
```

Esse teste simula **como modelos de linguagem geram texto**.

Processo:

```
<START> → token → token → token → ... → <EOS>
```

A cada passo:

1. O decoder recebe os tokens já gerados
2. Calcula probabilidades do vocabulário
3. Seleciona o próximo token (argmax)
4. Adiciona ao contexto
5. Repete até gerar `<EOS>`

Executar:

```bash
python -m transformer.decoder.inference.test_generate
```

Saída exemplo:

```
Sequência: ['<START>', 'token_928']
Sequência: ['<START>', 'token_928', 'token_442']
Sequência: ['<START>', 'token_928', 'token_442', 'token_125']
...
Fim da geração
```

Esse processo reproduz o **loop fundamental usado por LLMs para gerar texto**.

---

# Fluxo do Encoder Layer

```
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
```

---

# Fluxo do Decoder Layer

```
Input
 │
 ▼
Masked Self Attention
 │
 ▼
Add + Norm
 │
 ▼
Cross Attention
 │
 ▼
Add + Norm
 │
 ▼
Feed Forward
 │
 ▼
Add + Norm
 │
 ▼
Output
```

---

# Objetivo do Projeto

Este projeto demonstra como construir **os blocos fundamentais de um Transformer** manualmente, permitindo compreender:

* mecanismo de atenção
* scaled dot-product attention
* softmax aplicado à atenção
* normalização por camada
* redes feed-forward
* conexões residuais
* máscara causal do decoder
* cross attention entre encoder e decoder
* geração auto-regressiva de tokens

---

# Observação

Parte desta documentação foi gerada com IA generativa (GPT-5).
O aprendizado técnico - entendimento de arquitetura, conexão dos módulos e uso de numpy - foi feito utilizando os modelos GPT-5 e Claude Sonnet 4.6.

# Referência

Vaswani et al. (2017)
**Attention Is All You Need**

https://arxiv.org/abs/1706.03762
