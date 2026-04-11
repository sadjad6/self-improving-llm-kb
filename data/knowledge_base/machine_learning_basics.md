# Machine Learning Basics

#machine-learning #fundamentals

## What is Machine Learning?

Machine learning is a subset of [[Artificial Intelligence]] that focuses on building systems that learn from data. Instead of being explicitly programmed, these systems identify patterns in data and make decisions with minimal human intervention.

There are three main types of machine learning:

1. **Supervised Learning**: The algorithm learns from labeled training data. Examples include classification and regression tasks.
2. **Unsupervised Learning**: The algorithm finds hidden patterns in unlabeled data. Examples include clustering and dimensionality reduction.
3. **Reinforcement Learning**: The agent learns by interacting with an environment and receiving rewards or penalties.

## Key Concepts

### Features and Labels

In supervised learning, **features** (also called inputs or predictors) are the variables used to make predictions. **Labels** (also called targets or outputs) are the values the model tries to predict.

### Training and Testing

A dataset is typically split into:
- **Training set**: Used to train the model (usually 70-80% of data)
- **Validation set**: Used to tune hyperparameters (usually 10-15%)
- **Test set**: Used to evaluate final performance (usually 10-15%)

### Overfitting and Underfitting

- **Overfitting**: The model performs well on training data but poorly on unseen data. It has learned noise rather than the underlying pattern.
- **Underfitting**: The model is too simple to capture the underlying pattern in the data.

The goal is to find the right balance — the **bias-variance tradeoff**.

## Common Algorithms

| Algorithm | Type | Use Case |
|-----------|------|----------|
| Linear Regression | Supervised | Predicting continuous values |
| Logistic Regression | Supervised | Binary classification |
| Decision Trees | Supervised | Classification and regression |
| K-Means | Unsupervised | Clustering |
| PCA | Unsupervised | Dimensionality reduction |

## Related Topics

- [[Neural Networks]]
- [[Deep Learning]]
- [[Feature Engineering]]
- [[Model Evaluation]]

