"""
Comparative Study of Neural Network Regularization Techniques
for Regression and Spam Classification

Author  : Sanman Kadam
Project : Neural Network Regularization Study
Domain  : Deep Learning, Regularization, NLP

Description
-----------
This script implements and compares regularization techniques (L1, L2,
Dropout, Batch Normalization, Data Shuffling) on two tasks:
  1. Regression: Synthetic cubic dataset with injected outliers
  2. Classification: Spam detection using TF-IDF features
"""

# ---------------------------------------------------------------------------
# 1. Imports
# ---------------------------------------------------------------------------
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import nltk
nltk.download("stopwords", quiet=True)
from nltk.corpus import stopwords

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)


# ---------------------------------------------------------------------------
# 2. Utility Functions
# ---------------------------------------------------------------------------
def generate_data(seed=43, std=0.1, samples=500):
    """Generate synthetic cubic data with Gaussian noise."""
    np.random.seed(seed)
    X = np.linspace(-1, 1, samples)
    f = X**3 + 2 * X**2 - X
    y = f + np.random.randn(samples) * std
    return X, y


def plot_regression(X, y, f, y_pred, title):
    """Plot regression results: data, true function, and predictions."""
    plt.figure(figsize=(10, 5))
    plt.plot(X, y, "rx", alpha=0.5, label="Data Samples")
    plt.plot(X, f, "b-", linewidth=2, label="True Function")
    plt.plot(X, y_pred, "g-", linewidth=2, label="Predicted")
    plt.title(title, fontsize=14)
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def plot_classification_metrics(history):
    """Plot training and validation accuracy and loss curves."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.history["accuracy"], label="Train Accuracy", linewidth=2)
    axes[0].plot(history.history["val_accuracy"], label="Val Accuracy", linewidth=2)
    axes[0].set_title("Accuracy", fontsize=13)
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(history.history["loss"], label="Train Loss", linewidth=2)
    axes[1].plot(history.history["val_loss"], label="Val Loss", linewidth=2)
    axes[1].set_title("Loss", fontsize=13)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


# ===================================================================
# PART A: REGRESSION -- Synthetic Cubic Dataset with Outliers
# ===================================================================

# ---------------------------------------------------------------------------
# 3. Data Generation and Outlier Injection
# ---------------------------------------------------------------------------
X, y = generate_data()
f = X**3 + 2 * X**2 - X

plt.figure(figsize=(10, 5))
plt.plot(X, y, "rx", alpha=0.5, label="Data Samples")
plt.plot(X, f, "b-", linewidth=2, label="True Function")
plt.title("Synthetic Cubic Data with Gaussian Noise", fontsize=14)
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# Inject outliers
y[20:30] = 0
y[100:110] = 2
y[180:190] = 4
y[260:270] = -2
y[340:350] = -3
y[420:430] = 4

plt.figure(figsize=(10, 5))
plt.plot(X, y, "rx", alpha=0.5, label="Data Samples (with Outliers)")
plt.plot(X, f, "b-", linewidth=2, label="True Function")
plt.title("Cubic Data with Injected Outliers", fontsize=14)
plt.xlabel("X")
plt.ylabel("y")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------------
# 4. Baseline Model (No Regularization)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("BASELINE MODEL -- No Regularization")
print("=" * 60)

model = Sequential([
    Dense(1000, activation="relu", input_shape=(1,)),
    Dense(120, activation="relu"),
    Dense(120, activation="relu"),
    Dense(1),
])
model.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
model.fit(X, y, epochs=20, batch_size=100)

y_pred = model.predict(X)
plot_regression(X, y, f, y_pred, "Baseline Model -- No Regularization")

no_reg = np.mean((y - y_pred) ** 2)
print(f"Mean Squared Error (No Regularization): {no_reg:.4f}\n")

# ---------------------------------------------------------------------------
# 5. L1 Regularization (Lasso)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("L1 REGULARIZATION (Lasso)")
print("=" * 60)

model_l1 = Sequential([
    Dense(1000, activation="relu", input_shape=(1,),
          kernel_regularizer=keras.regularizers.l1(l1=0.01)),
    Dense(120, activation="relu",
          kernel_regularizer=keras.regularizers.l1(l1=0.001)),
    Dense(120, activation="relu"),
    Dense(1),
])
model_l1.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
model_l1.fit(X, y, epochs=20, batch_size=100)

y_pred = model_l1.predict(X)
plot_regression(X, y, f, y_pred, "L1 Regularization (Lasso)")

l1 = np.mean((y - y_pred) ** 2)
print(f"Mean Squared Error (L1): {l1:.4f}\n")

# ---------------------------------------------------------------------------
# 6. L2 Regularization (Ridge)
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("L2 REGULARIZATION (Ridge)")
print("=" * 60)

model_l2 = Sequential([
    Dense(1000, activation="relu", input_shape=(1,),
          kernel_regularizer=keras.regularizers.l2(l2=0.0001)),
    Dense(120, activation="relu",
          kernel_regularizer=keras.regularizers.l2(l2=0.0001)),
    Dense(120, activation="relu",
          kernel_regularizer=keras.regularizers.l2(l2=0.0001)),
    Dense(1),
])
model_l2.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
model_l2.fit(X, y, validation_split=0.2, epochs=20, batch_size=40)

y_pred = model_l2.predict(X)
plot_regression(X, y, f, y_pred, "L2 Regularization (Ridge)")

l2 = np.mean((y - y_pred) ** 2)
print(f"Mean Squared Error (L2): {l2:.4f}\n")

# ---------------------------------------------------------------------------
# 7. Dropout Regularization
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("DROPOUT REGULARIZATION (rate=0.1)")
print("=" * 60)

model_dp = Sequential([
    Dense(1000, activation="relu", input_shape=(1,)),
    Dropout(0.1),
    Dense(120, activation="relu"),
    Dropout(0.1),
    Dense(120, activation="relu"),
    Dropout(0.1),
    Dense(1),
])
model_dp.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
model_dp.fit(X, y, validation_split=0.2, epochs=20, batch_size=40)

y_pred = model_dp.predict(X)
plot_regression(X, y, f, y_pred, "Dropout Regularization (rate=0.1)")

dp = np.mean((y - y_pred) ** 2)
print(f"Mean Squared Error (Dropout): {dp:.4f}\n")

# ---------------------------------------------------------------------------
# 8. Batch Normalization
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("BATCH NORMALIZATION")
print("=" * 60)

model_bn = Sequential([
    Dense(1000, activation="relu", input_shape=(1,)),
    BatchNormalization(),
    Dense(120, activation="relu"),
    Dense(120, activation="relu"),
    Dense(1),
])
model_bn.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
model_bn.fit(X, y, validation_split=0.2, epochs=20, batch_size=40)

y_pred = model_bn.predict(X)
plot_regression(X, y, f, y_pred, "Batch Normalization")

bn = np.mean((y - y_pred) ** 2)
print(f"Mean Squared Error (BatchNorm): {bn:.4f}\n")

# ---------------------------------------------------------------------------
# 9. Data Shuffling
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("DATA SHUFFLING")
print("=" * 60)

model_sh = Sequential([
    Dense(1000, activation="relu", input_shape=(1,)),
    Dense(120, activation="relu"),
    Dense(120, activation="relu"),
    Dense(1),
])
model_sh.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
model_sh.fit(X, y, validation_split=0.2, epochs=20, batch_size=40, shuffle=True)

y_pred = model_sh.predict(X)
plot_regression(X, y, f, y_pred, "Data Shuffling")

sh = np.mean((y - y_pred) ** 2)
print(f"Mean Squared Error (Shuffled): {sh:.4f}\n")

# ---------------------------------------------------------------------------
# 10. Regression Comparison
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("REGRESSION RESULTS SUMMARY")
print("=" * 60)

names = ["No Reg.", "L1", "L2", "Dropout", "BatchNorm", "Shuffling"]
errors = [no_reg, l1, l2, dp, bn, sh]

plt.figure(figsize=(12, 6))
colors = ["#e74c3c", "#3498db", "#2ecc71", "#9b59b6", "#f39c12", "#1abc9c"]
bars = plt.bar(names, errors, width=0.6, color=colors, edgecolor="black", linewidth=0.8)

for bar, err in zip(bars, errors):
    plt.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.03,
        f"{err:.4f}",
        ha="center", va="bottom", fontsize=11, fontweight="bold",
    )

plt.title("MSE Comparison Across Regularization Techniques", fontsize=14)
plt.ylabel("Mean Squared Error", fontsize=12)
plt.xlabel("Regularization Technique", fontsize=12)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()
plt.show()

results_df = pd.DataFrame({"Technique": names, "MSE": [f"{e:.4f}" for e in errors]})
print("\n", results_df.to_string(index=False))


# ===================================================================
# PART B: CLASSIFICATION -- Spam Detection with TF-IDF
# ===================================================================

# ---------------------------------------------------------------------------
# 11. Data Loading and Preprocessing
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SPAM CLASSIFICATION SETUP")
print("=" * 60)

data = pd.read_csv("Data/spam.csv", encoding="latin-1")
data = data[["v1", "v2"]]
data.columns = ["label", "text"]
data["label"] = data["label"].map({"ham": 0, "spam": 1})

sw = stopwords.words("english")
vectorizer = TfidfVectorizer(stop_words=sw)

X_cls = vectorizer.fit_transform(data["text"]).toarray()
y_cls = data["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    X_cls, y_cls, test_size=0.2, random_state=42
)
input_dim = X_cls.shape[1]

print(f"Feature Dimensions: {input_dim}")
print(f"Training Samples:   {X_train.shape[0]}")
print(f"Testing Samples:    {X_test.shape[0]}")


# ---------------------------------------------------------------------------
# 12. Classification Model Builder
# ---------------------------------------------------------------------------
def build_and_evaluate(reg=None, epochs=10):
    """Build, train, and evaluate a classification model."""
    model = Sequential()
    model.add(Dense(512, activation="relu", input_shape=(input_dim,)))

    if reg == "L1":
        model.add(Dense(256, activation="relu",
                        kernel_regularizer=regularizers.l1(0.001)))
        model.add(Dense(64, activation="relu",
                        kernel_regularizer=regularizers.l1(0.001)))
    elif reg == "L2":
        model.add(Dense(256, activation="relu",
                        kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dense(64, activation="relu",
                        kernel_regularizer=regularizers.l2(0.001)))
    elif reg == "Dropout":
        model.add(Dropout(0.3))
        model.add(Dense(256, activation="relu"))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation="relu"))
    elif reg == "BatchNorm":
        model.add(BatchNormalization())
        model.add(Dense(256, activation="relu"))
        model.add(BatchNormalization())
        model.add(Dense(64, activation="relu"))
    else:
        model.add(Dense(256, activation="relu"))
        model.add(Dense(64, activation="relu"))

    model.add(Dense(1, activation="sigmoid"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

    history = model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=epochs, batch_size=64, verbose=1,
    )

    plot_classification_metrics(history)

    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")

    return model, history


# ---------------------------------------------------------------------------
# 13. Classification Experiments
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("CLASSIFICATION -- Base Model")
print("=" * 60)
build_and_evaluate()

print("\n" + "=" * 60)
print("CLASSIFICATION -- L1 Regularization")
print("=" * 60)
build_and_evaluate(reg="L1")

print("\n" + "=" * 60)
print("CLASSIFICATION -- L2 Regularization")
print("=" * 60)
build_and_evaluate(reg="L2")

print("\n" + "=" * 60)
print("CLASSIFICATION -- Dropout")
print("=" * 60)
build_and_evaluate(reg="Dropout")

print("\n" + "=" * 60)
print("CLASSIFICATION -- Batch Normalization")
print("=" * 60)
build_and_evaluate(reg="BatchNorm")


# ---------------------------------------------------------------------------
# 14. Final Summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("STUDY COMPLETE")
print("=" * 60)
print("Author  : Sanman Kadam")
print("Project : Neural Network Regularization Study")
print("=" * 60)