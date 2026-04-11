# Model Evaluation

#evaluation #metrics #machine-learning

## Why Evaluation Matters

Proper evaluation ensures that [[Machine Learning Basics|machine learning]] models generalize well to unseen data and meet business requirements. Without rigorous evaluation, models may appear to perform well but fail in production.

## Classification Metrics

### Accuracy
- Proportion of correct predictions
- Misleading for imbalanced datasets

### Precision and Recall
- **Precision**: Of all positive predictions, how many are correct? TP/(TP+FP)
- **Recall**: Of all actual positives, how many were found? TP/(TP+FN)
- **F1 Score**: Harmonic mean of precision and recall: 2·(P·R)/(P+R)

### ROC-AUC
- Plots True Positive Rate vs False Positive Rate
- AUC of 1.0 = perfect classifier, 0.5 = random

## Regression Metrics

- **MSE** (Mean Squared Error): Average of squared differences
- **RMSE** (Root MSE): Square root of MSE, same units as target
- **MAE** (Mean Absolute Error): Average of absolute differences
- **R² Score**: Proportion of variance explained by the model

## Retrieval Metrics

### Recall@K
Proportion of relevant documents found in the top K results:
- Recall@K = |relevant ∩ retrieved@K| / |relevant|

### Mean Reciprocal Rank (MRR)
Average of reciprocal ranks of the first relevant result:
- MRR = (1/|Q|) Σ (1/rank_i)

### Normalized Discounted Cumulative Gain (nDCG)
Measures ranking quality considering position:
- Higher-ranked relevant results contribute more to the score

## Cross-Validation

### K-Fold Cross-Validation
1. Split data into K equal folds
2. Train on K-1 folds, validate on remaining fold
3. Repeat K times, average results

### Stratified K-Fold
- Maintains class distribution in each fold
- Important for imbalanced datasets

## LLM Evaluation

### LLM-as-Judge
Using a powerful LLM to evaluate outputs:
- Score answers on relevance, faithfulness, completeness
- Provide rubrics for consistent scoring
- Cost-effective alternative to human evaluation

### Heuristic Evaluation
- Answer length and completeness checks
- Citation and grounding verification
- Hallucination detection via context comparison

## Related Topics

- [[Machine Learning Basics]]
- [[Retrieval Augmented Generation]]
- [[Neural Networks]]

