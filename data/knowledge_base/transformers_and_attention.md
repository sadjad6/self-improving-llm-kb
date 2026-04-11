# Transformers and Attention Mechanisms

#transformers #attention #nlp

## The Attention Mechanism

Attention allows models to focus on relevant parts of the input when producing output. It was introduced in the paper "Attention Is All You Need" (Vaswani et al., 2017).

### Self-Attention

Self-attention computes relationships between all positions in a sequence:

1. Each token is projected into three vectors: **Query (Q)**, **Key (K)**, **Value (V)**
2. Attention scores are computed as: Attention(Q, K, V) = softmax(QK^T / √d_k) V
3. The scaling factor √d_k prevents gradients from becoming too small

### Multi-Head Attention

Instead of a single attention function, multi-head attention:
- Runs multiple attention operations in parallel (heads)
- Each head learns different relationship patterns
- Outputs are concatenated and projected

## Transformer Architecture

### Encoder

The encoder processes the input sequence:
- Multi-head self-attention layer
- Feed-forward neural network
- Layer normalization and residual connections
- Used in models like [[BERT]]

### Decoder

The decoder generates the output sequence:
- Masked multi-head self-attention (prevents looking at future tokens)
- Cross-attention with encoder output
- Feed-forward neural network
- Used in autoregressive generation like [[GPT]]

### Positional Encoding

Since transformers have no inherent notion of sequence order, positional encodings are added to input embeddings. Methods include:
- **Sinusoidal encoding**: Fixed mathematical functions
- **Learned embeddings**: Trainable position vectors
- **RoPE** (Rotary Position Embedding): Modern approach used in LLaMA

## Key Transformer Models

### BERT (Bidirectional Encoder Representations from Transformers)
- Encoder-only architecture
- Pre-trained with masked language modeling (MLM)
- Excels at understanding tasks: classification, NER, QA

### GPT (Generative Pre-trained Transformer)
- Decoder-only architecture
- Autoregressive language modeling
- Excels at generation tasks
- [[GPT-4]] is the latest major version

### T5 (Text-to-Text Transfer Transformer)
- Encoder-decoder architecture
- Frames all NLP tasks as text-to-text problems

## Modern Advances

### Context Engineering
As described by [[Andrej Karpathy]], context engineering is the art of:
- Filling the context window with the right information at the right time
- Managing what the model sees to maximize performance
- Building systems that dynamically construct optimal prompts

### Scaling Laws
Research shows predictable relationships between:
- Model size (parameters)
- Dataset size (tokens)
- Compute budget (FLOPs)
- Model performance

## Related Topics

- [[Neural Networks]]
- [[Natural Language Processing]]
- [[Retrieval Augmented Generation]]
- [[Machine Learning Basics]]

