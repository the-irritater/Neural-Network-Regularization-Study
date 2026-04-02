# Comparative Study of Neural Network Regularization Techniques

## For Regression and Spam Classification

---

| Field | Details |
|---|---|
| **Author** | Sanman Kadam |
| **Domain** | Deep Learning, Regularization, NLP |
| **Language** | Python 3.13 |
| **Framework** | TensorFlow / Keras |
| **Date** | April 2026 |

---

## Project Overview

This project implements a comparative study of regularization techniques in neural networks across two distinct tasks:

1. **Regression**: A synthetic cubic dataset with injected outliers, evaluating how each technique handles noisy, corrupted data.
2. **Classification**: Spam detection using TF-IDF vectorized SMS messages, assessing regularization impact on text classification.

---

## Problem Statement

Deep neural networks possess a high degree of representational capacity, enabling them to model complex nonlinear relationships. However, this capacity makes them prone to **overfitting** -- memorizing noise, outliers, and idiosyncrasies in training data rather than learning true underlying patterns.

In real-world applications, training data is often contaminated with measurement noise, labeling errors, and outliers. Without appropriate constraints on model complexity, neural networks will fit these artifacts, resulting in unstable predictions and poor generalization.

**Regularization** techniques address this by introducing constraints or penalties that discourage excessive model complexity. This study investigates:

- **Weight penalties** (L1, L2) that modify the loss function to penalize large weights
- **Stochastic methods** (Dropout) that randomly disable neurons during training
- **Normalization methods** (Batch Normalization) that stabilize internal activations
- **Data-level methods** (Shuffling) that reduce ordering bias in training batches

The central question: **How do different regularization techniques compare in their ability to reduce overfitting and improve generalization across fundamentally different learning tasks?**

---

## Aim

To study and compare the effect of different regularization techniques (L1, L2, Dropout, Batch Normalization, and Data Shuffling) in neural networks for regression and binary classification problems.

---

## Objectives

- Understand overfitting in deep neural networks
- Apply L1, L2, Dropout, Batch Normalization, and Data Shuffling regularization
- Compare Mean Squared Error across techniques (regression)
- Compare accuracy and loss across techniques (classification)
- Analyze why regularization behaves differently in regression vs. text classification

---

## Techniques Implemented

| Technique | Mechanism | Task |
|---|---|---|
| L1 Regularization (Lasso) | Adds absolute value of weights to loss | Regression and Classification |
| L2 Regularization (Ridge) | Adds squared value of weights to loss | Regression and Classification |
| Dropout | Randomly deactivates neurons during training | Regression and Classification |
| Batch Normalization | Normalizes layer inputs to zero mean / unit variance | Regression and Classification |
| Data Shuffling | Randomizes sample order each epoch | Regression |

---

## Repository Structure

```
Neural-Network-Regularization-Study/
|-- Data/
|   |-- spam.csv                                  # SMS spam dataset
|-- Images/                                       # Generated plots
|-- neural_network_regularizers.py                # Main Python script
|-- Neural_Network_Regularization_Study.ipynb     # Jupyter notebook
|-- requirements.txt                              # Dependencies
|-- README.md                                     # This file
```

---

## Technologies Used

- Python 3.13
- TensorFlow / Keras
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- NLTK

---

## Key Observations

### Regression (Synthetic Cubic Data with Outliers)

- The baseline model without regularization exhibits the highest MSE due to overfitting
- L1 regularization achieves the best MSE by promoting weight sparsity
- L2 regularization provides smooth weight shrinkage and stable predictions
- Dropout reduces overfitting through stochastic neuron deactivation
- Batch Normalization offers mild regularization as a side-effect
- Data Shuffling provides minimal improvement over baseline

### Classification (Spam Detection)

- Regularization techniques do not significantly improve accuracy
- TF-IDF features provide inherent sparsity, reducing the benefit of L1
- The classification boundary is approximately linear, limiting overfitting
- Baseline model already achieves strong performance due to feature separability

---

## How to Run

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run the Script

```bash
python neural_network_regularizers.py
```

### Run the Notebook

```bash
jupyter notebook Neural_Network_Regularization_Study.ipynb
```

---

## Conclusion

The effectiveness of regularization is strongly task-dependent:

- For **noisy regression** problems with outliers, regularization (especially L1 and L2) is essential for reducing overfitting and improving generalization.
- For **text classification** with well-separated TF-IDF features, regularization provides marginal gains because the baseline model does not severely overfit.

This study demonstrates the importance of selecting appropriate regularization strategies based on the data characteristics and task complexity.

---

| Field | Details |
|---|---|
| **Author** | Sanman Kadam |
| **Project** | Comparative Study of Neural Network Regularization Techniques |
| **Status** | Complete |
