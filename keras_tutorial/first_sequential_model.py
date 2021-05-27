# import libraries for data creation and process
import itertools
from json import load
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

# import for confusion matrix
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

# import for saving and loading model
import os.path
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model

# SEQUENTIAL MODEL
# X = input data -- np array, tf tensor, dictionary mapping input names to array/tensor, tf.data dataset,
#                   generator or keras.utils.Sequence
# y = target data -- consistent with X, if X is dataset, generator or keras.utils.Sequence y shouldn't be specified
#                    because target are obtained from X

train_samples = []
train_labels = []  # label0 = no side effects, 1 = side effects
test_samples = []
test_labels = []

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

# "outliers" group
for i in range(50):
    # the 5% younger individuals who did experience side effect
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(1)

    # the 5% older individual who did not experience side effects
    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(0)

# "bulk" of the group
for i in range(1000):
    # the 95% younger individuals who did not experience side effect
    random_younger = randint(13, 64)
    test_samples.append(random_younger)
    test_labels.append(0)

    # the 95% older individual who did experience side effects
    random_older = randint(65, 100)
    test_samples.append(random_older)
    test_labels.append(1)

# process data to be in the format needed for the fit function
train_labels = np.array(train_labels)
train_samples = np.array(train_samples)
test_labels = np.array(test_labels)
test_samples = np.array(test_samples)

# shuffling the data
train_samples, train_labels = shuffle(train_samples, train_labels)
test_samples, test_labels = shuffle(test_samples, test_labels)

# normalize the data
scaler = MinMaxScaler(feature_range=(0, 1))
# we reshape the data because the fit_transform function doesn't accept 1d array
scaled_train_samples = scaler.fit_transform(train_samples.reshape(-1, 1))
scaled_test_samples = scaler.fit_transform(test_samples.reshape(-1, 1))

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
          validation_split=0.1,  # defining validation in training
          batch_size=10,
          epochs=30,
          shuffle=True,  # the shuffle happens after the splitting, so if we didn't
          # shuffled before we don't have the validation set shuffled
          verbose=2
          )

# PREDICTION
# we get the probability for each label
predictions = model.predict(x=scaled_test_samples,
                            batch_size=10,
                            verbose=0)

# argmax to get the highest index of the prediciton probability
rounded_predictions = np.argmax(predictions, axis=-1)

# CONFUSION MATRIX

cm = confusion_matrix(y_true=test_labels, y_pred=rounded_predictions)

cm_plot_labels = ["no side effects", "side effects"]


def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title="Confusion Matrix",
                          cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    else:
        print("Confusion Matrix, without normalization")

    print(cm)

    thresh = cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.show()


plot_confusion_matrix(cm=cm, classes=cm_plot_labels)

# SAVE THE MODEL
# there are multiple ways to save a mdoel.
# first one: .save() -- check to see if file exists, if not save it
if os.path.isfile("models/medial_trial_model.h5") is False:
    model.save("models/medical_trial_model.h5")

# this function saves:
# the architecture of the model,
# the weights,
# the training configuration (loss, optimizer),
# the state of the optimizer

# load the model
new_model = load_model("models/medical_trial_model.h5")

# second one: .to_json() -- save only the architecture (also to_yaml)
json_string = model.to_json()

# load the architecture (also from_yaml)
model_architecture = model_from_json(json_string)

# third one: .save_weights() --  save only the weights
if os.path.isfile("models/medical_trial_model_weights.h5") is False:
    model.save("models/medical_trial_model_weights.h5")

# when we save only the weights, we need to create a new model with the
# same architecture and then load the weights

model2 = Sequential([
    Dense(units=16, input_shape=(1,), activation='relu'),
    Dense(units=32, activation='relu'),
    Dense(units=2, activation='softmax')
])

model2.load_weights("models/medical_trial_model_weights.h5")
