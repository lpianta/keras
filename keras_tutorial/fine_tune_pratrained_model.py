# import libraries
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import glob
import matplotlib.pyplot as plt
import warnings

# ignore future warnings
warnings.simplefilter(action="ignore", category=FutureWarning)

# plot confusion matrix function


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


# DOWNLOAD MODEL
vgg16_model = tf.keras.applications.vgg16.VGG16()

# get model summary
vgg16_model.summary()

# CREATE NEW SEQUENTIAL MODEL in which we add all the layer of the VGG16 model
# except for the last layer. We do this to work with sequential mdoel without
# using the functional api
model = Sequential()
for layer in vgg16_model.layers[:-1]:
    model.add(layer)

# set all the layers to non trainable
for layer in model.layers:
    layer.trainable = False

# ADD NEW OUTPUT LAYER with just 2 classes
model.add(Dense(units=2, activation="softmax"))

# check the summary of the mdoel
model.summary()

# IMPORT DATA
# define paths
train_path = "keras_tutorial/datasets/dogs_vs_cats/train"
valid_path = "keras_tutorial/datasets/dogs_vs_cats/valid"
test_path = "keras_tutorial/datasets/dogs_vs_cats/test"

# create batches of images converted to keras generator
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224, 224), classes=["cat", "dog"], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224, 224), classes=["cat", "dog"], batch_size=10)
# we don't shuffle test batches to evaluate the model later
# check https://deeplizard.com/learn/video/LhEMXbjGV_4 for unlabeled test case
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224, 224), classes=["cat", "dog"], batch_size=10, shuffle=False)

# COMPILE MODEL
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="categorical_crossentropy", metrics=["accuracy"])

# TRAIN THE MODEL
# model.fit(x=train_batches, validation_data=valid_batches, epochs=5, verbose=2)

# INFERENCE
predictions = model.predict(x=test_batches, verbose=0)

# CONFUSION MATRIX
cm = confusion_matrix(y_true=test_batches.classes,
                      y_pred=np.argmax(predictions, axis=-1))
# check index of class in test to correctly define cm_plot_labels
test_batches.class_indices
cm_plot_labels = ["cat", "dog"]
plot_confusion_matrix(cm=cm, classes=cm_plot_labels)
