# import libraries for data creation and process
import numpy as np
from random import randint
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler

# import libraries for model creation
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.python.ops.gen_batch_ops import batch
from tensorflow.python.ops.gen_nn_ops import softmax

# SEQUENTIAL MODEL
# X = input data -- np array, tf tensor, dictionary mapping input names to array/tensor, tf.data dataset,
#                   generator or keras.utils.Sequence
# y = target data -- consistent with X, if X is dataset, generator or keras.utils.Sequence y shouldn't be specified
#                    because target are obtained from X

train_samples = []
train_labels = []  # label0 = no side effects, 1 = side effects

# FAKE DATA
# an experimental drug was tested on individuals from age 13 to age 100
# the trial had 2100 partecipants, half under 65, half above
# around 95% of patience 65 or older experienced side effects
# around 95% of patience below 65 or didn't experienced side effects


# "outliers" group
for i in range(50):
    # the 5% younger individuals who did experience side effect
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(1)

    # the 5% older individual who did not experience side effects
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(0)

# "bulk" of the group
for i in range(1000):
    # the 95% younger individuals who did not experience side effect
    random_younger = randint(13, 64)
    train_samples.append(random_younger)
    train_labels.append(0)

    # the 95% older individual who did experience side effects
    random_older = randint(65, 100)
    train_samples.append(random_older)
    train_labels.append(1)

# process data to be in the format needed for the fit function
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)

# shuffling the data
train_samples, train_labels = shuffle(train_samples, train_labels)

# normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
# we reshape the data because the fit_transform function doesn't accept 1d array
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))

# BUILDING THE MODEL
model = Sequential([
    # first dense layer is actually the second layer, we don't explicit define
    # the input layer because the input data create the layer itself. We need
    # to pass the input_shape in the first hidden layer to correctly create the
    # input layer
    # Dense Layer are fully connected layers
    Dense(units=16, input_shape=(1,), activation='relu'),
    # we don't need to specify the input_shape anymore
    Dense(units=32, activation='relu'),
    # output layer, 1 unit for each class, activation with softmax
    Dense(units=2, activation='softmax')
])

model.summary()

# TRAIN THE MODEL
# compile function to initialize the parameters of the training
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy']
              )
# fit function to actually train the model
model.fit(x=scaled_train_samples,
          y=train_labels,
          batch_size=10,
          epochs=30,
          shuffle=True,
          verbose=2
          )
