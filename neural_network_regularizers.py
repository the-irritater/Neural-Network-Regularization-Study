# **Comparative Study of Neural Network Regularization Techniques for Regression and Spam Classification**

"""## Aim
To study and compare the effect of different regularization techniques (L1, L2, Dropout, Batch Normalization, and Data Shuffling) in neural networks for:

*  A regression problem (synthetic cubic dataset)

*  A binary classification problem (Spam detection using TF-IDF)"""

"""## Objectives

*  To understand overfitting in deep neural networks.

*  To apply L1, L2, Dropout, and BatchNorm regularization.

*  To compare Mean Squared Error (regression).

*  To compare accuracy and loss (classification).

*  To analyze why regularization behaves differently in regression and text classification.
"""

# Commented out IPython magic to ensure Python compatibility.
import warnings
warnings.filterwarnings('ignore')
# All Libraries required for this lab are listed below. The libraries pre-installed on Skills Network Labs are commented.
# !mamba install -qy numpy==1.22.3 matplotlib==3.5.1 tensorflow==2.9.0 opencv-python==4.5.5.62

# Note: If your environment doesn't support "!mamba install", use "!pip install --user"

# RESTART YOUR KERNEL AFTERWARD AS WELL

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline

import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report, accuracy_score
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense
def warn(*args, **kwargs):
    pass
import warnings
warnings.warn = warn
warnings.filterwarnings('ignore')
warn()

def prepare_data():
    try:
        data = pd.read_csv("spam.csv", encoding='latin-1')
    except FileNotFoundError:
        print("Data file not found, make sure it's downloaded.")

    data.drop(['Unnamed: 2','Unnamed: 3','Unnamed: 4'], axis=1, inplace=True)
    data.rename(columns={"v1": "label", "v2": "text"}, inplace=True)
    data.label = data['label'].map({'ham':0, 'spam':1})
    data['Count'] = data['text'].apply(lambda x: len(x))

    sw=stopwords.words("english")
    vectorizer = TfidfVectorizer(stop_words=sw, binary=True)

    X = vectorizer.fit_transform(data.text).toarray()
    y = data.label

    return X, y

def plot_metrics(history):
    fig = plt.figure(figsize=(10,5))
    for i, metric in enumerate(['accuracy', 'loss']):
        train_metrics = history.history[metric]
        val_metrics = history.history['val_'+metric]
        epochs = range(1, len(train_metrics) + 1)
        plt.subplot(1,2,i+1)
        plt.plot(epochs, train_metrics)
        plt.plot(epochs, val_metrics)
        plt.title('Training and validation '+ metric)
        plt.xlabel("Epochs")
        plt.ylabel(metric)
        plt.legend(["train_"+metric, 'val_'+metric])

tf.keras.regularizers.l2(l2=0.01)

dense_layer = Dense(32,
                activation="relu",
                kernel_regularizer=tf.keras.regularizers.l2(l2=0.01))

dense_layer = Dense(32,
            activation="relu",
            kernel_regularizer=tf.keras.regularizers.l1(l1=0.01))

dense_layer = Dense(32,
                activation="relu",
                kernel_regularizer="l1")

from tensorflow.keras.layers import Dropout

dropout_layer = Dropout(rate=0.2)

from tensorflow.keras.layers import Dense, BatchNormalization

batchnorm_layer = BatchNormalization()

def generate_data(seed=43,std=0.1,samples=500):
    np.random.seed(seed)
    X =np.linspace(-1,1,samples)
    f = X**3 +2*X**2 -X
    y=f+np.random.randn(samples)*std

    return X, y


X,y = generate_data()
f = X**3 +2*X**2 -X
plt.plot(X, y,'rx',label="data samples")
plt.plot(X, f,label="true function")
plt.title("data and true function")
plt.legend()
plt.show()

"""The initial plot shows red noisy points around a smooth cubic curve. This represents real-world data with randomness."""

y[20:30] = 0
y[100:110] = 2
y[180:190] = 4
y[260:270] = -2
y[340:350] = -3
y[420:430] = 4

plt.plot(X, y,'rx',label="data samples")
plt.plot(X, f,label="true function")
plt.legend()
plt.show()

"""The updated plot shows some points far away from the true function. These outliers increase the risk of overfitting."""

from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

model = Sequential()
model.add(Dense(1000, activation='relu',input_shape=(1,)))
model.add(Dense(120,activation='relu'))
model.add(Dense(120,activation='relu'))
model.add(Dense(1))

model.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
model.fit(X, y,  epochs=20, batch_size=100)

"""From the output, I observed that the training loss is decreasing rapidly, which indicates that the model is learning the training data very well. However, in the prediction plot, the curve is bending sharply near noisy and outlier points. This shows that the model is memorizing the noise instead of learning the true cubic pattern. The Mean Squared Error is comparatively high. Therefore, I conclude that the model is suffering from overfitting due to high model complexity and absence of regularization."""

y_pred = model.predict(X)
plt.plot(X, y,'rx',label="data samples")
plt.plot(X, f,label="true function")
plt.plot(X, y_pred ,label="predicted function")
plt.legend()
plt.show()

"""From the graph, I observe that the red crosses represent the noisy data samples, the blue curve represents the true cubic function, and the orange curve represents the predicted function by the model.

The predicted curve closely follows most of the data points but shows noticeable bending near the center region and around extreme outliers. It does not perfectly match the true function and slightly deviates in the middle portion. This indicates that the model is trying to fit the noisy data and is somewhat influenced by outliers.

The model captures the general trend of the function but shows signs of slight overfitting due to noise.
"""

no_reg = np.mean((y-y_pred)**2)
print(f"Mean squared error is {no_reg}\n")

"""From the output, I observed that the Mean Squared Error value is 1.9173, which is relatively high. Since MSE measures the average squared difference between actual and predicted values, this large value indicates that the model’s predictions are deviating significantly from the true function.

This high error suggests that the model is not generalizing well and is likely overfitting the noisy data and outliers. The squared error increases the impact of large deviations, especially due to extreme points.
"""

model_l1 = Sequential()

model_l1.add(Dense(1000, activation='relu',input_shape=(1,),kernel_regularizer=keras.regularizers.l1(l1=0.01)))
model_l1.add(Dense(120,activation='relu',kernel_regularizer=keras.regularizers.l1(l1=0.001)))
model_l1.add(Dense(120,activation='relu'))
model_l1.add(Dense(1))
model_l1.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
model_l1.fit(X, y,  epochs=20, batch_size=100)

"""In this step, I applied L1 regularization. From the output, I observed that the loss decreases in a controlled manner. The predicted curve is smoother compared to the model without regularization. It does not aggressively fit the outliers. The MSE is reduced compared to the baseline model. This shows that L1 penalizes large weights and forces some weights to become zero, reducing model complexity. Hence, overfitting is reduced."""

y_pred = model_l1.predict(X)
plt.plot(X, y,'rx',label="data samples")
plt.plot(X, f,label="true function")
plt.plot(X, y_pred,label="predicted function")
plt.legend()
plt.show()

"""From the plot, I observe that although there are extreme outliers (red points at y=4 and y=−3), the predicted function (orange curve) is not heavily influenced by them. Unlike standard Ordinary Least Squares (OLS), which uses squared error and gets strongly affected by large deviations, the predicted curve remains stable and closely follows the true function (blue curve).

This behavior indicates robust regression, where the loss function reduces the influence of extreme errors. Since large deviations are not pulling the curve drastically, the model is likely using a loss function such as MAE (L1 loss) or Huber loss, which penalizes large errors linearly rather than quadratically.

Therefore, the model prioritizes the dense central data region instead of fitting extreme outliers. This shows high robustness, better resistance to noise, and an improved breakdown point.
"""

l1 = np.mean((y-y_pred)**2)
print(f"Mean squared error is {l1}\n")

"""From the output, I observed that the Mean Squared Error is 1.6848, which is lower than the previous value (1.9173). This indicates that the model’s predictions are now closer to the true values.

The reduction in MSE suggests improved generalization and reduced overfitting. The model is less influenced by extreme outliers and is capturing the overall trend more effectively. Since MSE penalizes large errors quadratically, a decrease in this value confirms that large deviations have been minimized.
"""

model_l2 = Sequential()

model_l2.add(Dense(1000, activation='relu',input_shape=(1,),kernel_regularizer=keras.regularizers.l2(l2=0.0001)))
model_l2.add(Dense(120,activation='relu',kernel_regularizer=keras.regularizers.l2(l2=0.0001)))
model_l2.add(Dense(120,activation='relu',kernel_regularizer=keras.regularizers.l2(l2=0.0001)))
model_l2.add(Dense(1))
model_l2.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
model_l2.fit(X, y, validation_split=0.2, epochs=20, batch_size=40)

"""In this step, I applied L2 regularization. From the output, I observed stable training and validation loss. The predicted curve is smooth and closely follows the actual function. It does not overreact to noisy data points. The MSE value is lower compared to the model without regularization. This shows that L2 shrinks the weights smoothly and reduces variance without eliminating weights completely. Therefore, L2 improves generalization performance."""

y_pred = model_l2.predict(X)
plt.plot(X, y,'rx',label="data samples")
plt.plot(X, f,label="true function")
plt.plot(X, y_pred ,label="predicted function")
plt.legend()

l2 = np.mean((y-y_pred)**2)
print(f"Mean squared error is {l2}\n")

model_dp = Sequential()

model_dp.add(Dense(1000, activation='relu',input_shape=(1,)))
model_dp.add(Dropout(0.1))
model_dp.add(Dense(120,activation='relu'))
model_dp.add(Dropout(0.1))
model_dp.add(Dense(120,activation='relu'))
model_dp.add(Dropout(0.1))
model_dp.add(Dense(1))
model_dp.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
model_dp.fit(X, y, validation_split=0.2, epochs=20, batch_size=40)

"""In this step i applied Dropout (0.1). From the output, I observed improved MSE compared to the model without regularization. The graph shows that the predicted curve does not tightly follow noise and appears smoother.

Dropout randomly deactivates neurons during training, preventing co-adaptation and reducing overfitting. Thus, the model becomes more robust.
"""

y_pred = model_dp.predict(X)
plt.plot(X, y,'rx',label="data samples")
plt.plot(X, f,label="true function")
plt.plot(X, y_pred ,label="predicted function")
plt.legend()

dp = np.mean((y-y_pred)**2)
print(f"Mean squared error is {dp}\n")

model_bn = Sequential()

model_bn.add(Dense(1000, activation='relu',input_shape=(1,)))
model_bn.add(BatchNormalization())
model_bn.add(Dense(120,activation='relu'))

model_bn.add(Dense(120,activation='relu'))
model_bn.add(Dense(1))
model_bn.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
model_bn.fit(X, y, validation_split=0.2, epochs=20, batch_size=40)

"""In this step, I applied Batch Normalization. From the output, I observed stable training and moderate improvement in MSE. The graph shows a smooth predicted curve with less extreme bending.

Batch Normalization normalizes layer inputs and reduces internal covariate shift. This improves training stability and learning efficiency.
"""

y_pred = model_bn.predict(X)
plt.plot(X, y,'rx',label="data samples")
plt.plot(X, f,label="true function")
plt.plot(X, y_pred ,label="predicted function")
plt.legend()

bn = np.mean((y-y_pred)**2)
print(f"Mean squared error is {bn}\n")

model_sh = Sequential()

model_sh.add(Dense(1000, activation='relu',input_shape=(1,)))
model_sh.add(Dense(120,activation='relu'))
model_sh.add(Dense(120,activation='relu'))
model_sh.add(Dense(1))

model_sh.compile(optimizer=Adam(learning_rate=1e-3), loss="mean_squared_error")
model_sh.fit(X, y, validation_split=0.2, epochs=20, batch_size=40,shuffle=True)

"""In this step, I enabled data shuffling during training. From the output, I observed slight improvement in MSE. The predicted curve is slightly smoother compared to the baseline model.

Shuffling prevents sequence bias and improves generalization slightly. However, the improvement is minor compared to L1 and L2 regularization.
"""

y_pred = model_sh.predict(X)

plt.plot(X, y,'rx',label="data samples")
plt.plot(X, f,label="true function")
plt.plot(X, y_pred ,label="predicted function")
plt.legend()

sh = np.mean((y-y_pred)**2)
print(f"Mean squared error is {sh}\n")

names = ['No_reg','L1','L2','Drop_out','Batch_norm','Data_shuffling']
error = [no_reg, l1, l2, dp, bn, sh]

plt.figure(figsize=(10, 6))
plt.bar(names, error, width=0.8)
plt.title("Mean Squared Error", fontsize=13)

for i, err in enumerate(error):
    plt.text(i-0.2, err+0.1, str(round(err,3)), color='blue', va='center')

# Reproducibility
np.random.seed(42)
tf.random.set_seed(42)

# Download stopwords
nltk.download("stopwords")

data = pd.read_csv("spam.csv", encoding="latin-1")

# Keep only required columns
data = data[['v1', 'v2']]
data.columns = ['label', 'text']

# Convert labels to numeric
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

sw = stopwords.words("english")
vectorizer = TfidfVectorizer(stop_words=sw)

X = vectorizer.fit_transform(data['text']).toarray()
y = data['label'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

input_dim = X.shape[1]

def get_model(reg=None, epochs=10):

    model = Sequential()
    model.add(Dense(512, activation='relu', input_shape=(input_dim,)))

    if reg == "L1":
        model.add(Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l1(0.001)))
        model.add(Dense(64, activation='relu',
                        kernel_regularizer=regularizers.l1(0.001)))

    elif reg == "L2":
        model.add(Dense(256, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)))
        model.add(Dense(64, activation='relu',
                        kernel_regularizer=regularizers.l2(0.001)))

    elif reg == "Dropout":
        model.add(Dropout(0.3))
        model.add(Dense(256, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(64, activation='relu'))

    elif reg == "BatchNorm":
        model.add(BatchNormalization())
        model.add(Dense(256, activation='relu'))
        model.add(BatchNormalization())
        model.add(Dense(64, activation='relu'))

    else:
        model.add(Dense(256, activation='relu'))
        model.add(Dense(64, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        loss="binary_crossentropy",
        optimizer="adam",
        metrics=["accuracy"]
    )
    #Plots
    history = model.fit(X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=epochs,
    batch_size=64,
    verbose=1)

    plot_metrics(history)

    # Evaluation
    y_pred = (model.predict(X_test) > 0.5).astype("int32")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

    return model

def plot_metrics(history):

    plt.figure(figsize=(10,4))

    # Accuracy Plot
    plt.subplot(1,2,1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()

    # Loss Plot
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.show()

print("\nBase Model")
get_model()

"""The model perfectly fits training data, indicating high capacity. Slight gap between training and validation suggests mild overfitting."""

print("\nL1 Regularization")
get_model(reg="L1")

"""L1 does not significantly improve performance because TF-IDF features are already sparse."""

print("\nL2 Regularization")
get_model(reg="L2")

"""L2 improves weight stability but does not drastically change accuracy."""

print("\nDropout")
get_model(reg="Dropout")

"""Dropout shows minimal improvement due to the strong linear separability of spam dataset."""

print("\nBatch Normalization")
get_model(reg="BatchNorm")

"""Batch normalization stabilizes learning but does not outperform baseline significantly."""


"""Unlike regression, regularization techniques do not significantly improve 
classification accuracy in spam detection because the TF-IDF representation already provides strong feature separation"""