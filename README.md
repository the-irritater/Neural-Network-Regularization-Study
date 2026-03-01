# Comparative Study of Neural Network Regularization Techniques

## Project Overview
This project compares different regularization techniques in neural networks for:

1. Regression (Synthetic Cubic Dataset with Outliers)
2. Spam Classification (TF-IDF based Text Classification)

## Objectives
- Understand overfitting in deep neural networks
- Apply L1, L2, Dropout, Batch Normalization, and Data Shuffling
- Compare Mean Squared Error (Regression)
- Compare Accuracy and Loss (Classification)
- Analyze how regularization behaves differently across problem types

---

## Techniques Compared
- No Regularization (Baseline)
- L1 Regularization
- L2 Regularization
- Dropout
- Batch Normalization
- Data Shuffling

---

##  Key Findings

###  Regression
- Baseline model overfits noisy outliers
- L1 and L2 significantly reduce Mean Squared Error
- Dropout improves robustness
- Batch Normalization stabilizes training

### Spam Classification
- Baseline accuracy ≈ 97–98%
- Regularization techniques show minimal improvement
- TF-IDF features are already sparse and well-separated

---

## Conclusion

Regularization has a strong impact in regression problems with noise and outliers.  
However, in TF-IDF based spam classification, regularization provides limited improvement because the feature space is already sparse and structured.

---

## Technologies Used
- Python
- TensorFlow / Keras
- Scikit-learn
- NLTK
- Matplotlib
- Pandas
- NumPy

---

## How to Run

```bash
pip install -r requirements.txt
python neural_network_regularizers.py
```

---

## Author
Sanman Kadam :- https://www.linkedin.com/in/sanman-kadam-7a4990374/

Focus: Machine Learning & Deep Learning