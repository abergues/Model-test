# Try to import Time distributed layer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Fixing format of the data
def enc_label(label):
    code = 0
    if label == "right-turn":
        code = 1
    if label == "side":
        code = 2
    if label == "cuban-basic":
        code = 3
    if label == "suzie-q":
        code = 4
    return code

# Define hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 10

MAX_SEQ_LENGTH = 30   # number of frames
NUM_FEATURES = 50     # number of join coordinates
no_sample = 10

# Import the data
PATH_DATA = "../Model-test/Data.csv"
data = pd.read_csv(PATH_DATA)

# Exploratory analysis number of clases

# Data preprocessing, get the input X and the label y
ind_start = data[data['status'] == "S"].index.tolist()
ind_end = data[data['status'] == "E"].index.tolist()

# Take an interval between consecutive "S", one example
X = []
y = []

for i in range(no_sample):
    X.append(data.iloc[ind_start[i]: ind_end[i], 3:-2])
    y.append(data.loc[ind_start[i], 'label'])

# select 10 frames from the interval TODO should be uniform
ind_samp = []

for i in range(no_sample):
    aux = np.random.randint(ind_start[i], ind_end[i], MAX_SEQ_LENGTH)
    aux.sort()
    ind_samp.append(aux)
# TODO: make sure number are not repeated. Better to uniform sample the interval

# Changing format of the data to be compatible with Tensor Flow
X_train = [x.loc[ind_samp[ind], :].to_numpy() for (ind, x) in enumerate(X)]
X_train = np.array(X_train)
X_train = X_train.reshape(no_sample, MAX_SEQ_LENGTH, NUM_FEATURES).astype("float32")
# TODO: decide of the X values need to be normalized
y_train = [enc_label(x) for x in y]
y_train = np.array(y_train).astype("float32")

# Build the model
model = models.Sequential()
model.add(layers.InputLayer(input_shape=(MAX_SEQ_LENGTH, NUM_FEATURES)))
model.add(layers.LSTM(64))
model.add(layers.Dense(5, activation="softmax"))
model.summary()

# Compile the model
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Check the trainning accuracy
history = model.fit(
    X_train,
    y_train,
    epochs=10,
    validation_data=(X_train, y_train)
)

# Prediction example
pred = model.predict( np.expand_dims(X_train[0], axis=0) )
print(f"Pred {pred}: Real Label {y_train[0]}")