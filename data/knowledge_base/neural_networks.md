# Neural Networks

#deep-learning #neural-networks

## Overview

Neural networks are computing systems inspired by biological neural networks in the brain. They consist of interconnected nodes (neurons) organized in layers that process information.

A neural network typically has:
- **Input layer**: Receives the raw data
- **Hidden layers**: Process and transform the data
- **Output layer**: Produces the final prediction

## How Neural Networks Learn

### Forward Propagation

Data flows from the input layer through hidden layers to the output layer. Each neuron applies:
1. A weighted sum of its inputs
2. An **activation function** to introduce non-linearity

Common activation functions:
- **ReLU** (Rectified Linear Unit): f(x) = max(0, x) — most commonly used
- **Sigmoid**: f(x) = 1/(1+e^(-x)) — used for binary outputs
- **Tanh**: f(x) = (e^x - e^(-x))/(e^x + e^(-x)) — zero-centered
- **Softmax**: Used for multi-class classification outputs

### Backpropagation

The network learns by:
1. Computing the **loss** (error) between predictions and true labels
2. Calculating **gradients** using the chain rule
3. Updating weights using an **optimizer** (e.g., SGD, Adam)

This process repeats for many **epochs** until the model converges.

## Types of Neural Networks

### Feedforward Neural Networks (FNN)
The simplest type where data flows in one direction. Used for tabular data and basic classification.

### Convolutional Neural Networks (CNN)
Specialized for grid-like data (images). Key components:
- **Convolutional layers**: Extract spatial features
- **Pooling layers**: Reduce dimensionality
- **Applications**: Image classification, object detection, [[Computer Vision]]

### Recurrent Neural Networks (RNN)
Designed for sequential data. They have memory of previous inputs.
- **LSTM** (Long Short-Term Memory): Solves the vanishing gradient problem
- **GRU** (Gated Recurrent Unit): Simplified version of LSTM
- **Applications**: [[Natural Language Processing]], time series

### Transformers
Modern architecture that uses **self-attention** mechanisms:
- No recurrence needed — processes sequences in parallel
- Foundation of models like [[GPT]], [[BERT]], and modern LLMs
- **Applications**: NLP, computer vision, multimodal AI

## Hyperparameters

Key hyperparameters to tune:
- **Learning rate**: How big the weight updates are
- **Batch size**: Number of samples per gradient update
- **Number of layers/neurons**: Network architecture
- **Dropout rate**: Regularization technique (randomly zeroing neurons)
- **Weight initialization**: How initial weights are set

## Related Topics

- [[Machine Learning Basics]]
- [[Deep Learning]]
- [[Transfer Learning]]
- [[Model Evaluation]]

