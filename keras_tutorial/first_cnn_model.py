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

# DATA PREPARATION
# organize data into train, valid and test
os.chdir("keras_tutorial/datasets/dogs_vs_cats")
if os.path.isdir("train/dog") is False:
    os.makedirs("train/dog")
    os.makedirs("train/cat")
    os.makedirs("valid/dog")
    os.makedirs("valid/cat")
    os.makedirs("test/dog")
    os.makedirs("test/cat")

    for c in random.sample(glob.glob("train/cat*"), 500):
        shutil.move(c, "train/cat")
    for c in random.sample(glob.glob("train/dog*"), 500):
        shutil.move(c, "train/dog")
    for c in random.sample(glob.glob("train/cat*"), 100):
        shutil.move(c, "valid/cat")
    for c in random.sample(glob.glob("train/dog*"), 100):
        shutil.move(c, "valid/dog")
    for c in random.sample(glob.glob("train/cat*"), 50):
        shutil.move(c, "test/cat")
    for c in random.sample(glob.glob("train/dog*"), 50):
        shutil.move(c, "test/dog")

os.chdir("../../")

# define paths
train_path = "datasets/dogs_vs_cats/train"
valid_path = "datasets/dogs_vs_cats/valid"
test_path = "datasets/dogs_vs_cats/test"

# create batches of images converted to keras generator
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=train_path, target_size=(224, 224), classes=["cat", "dog"], batch_size=10)
valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=valid_path, target_size=(224, 224), classes=["cat", "dog"], batch_size=10)
# we don't shuffle test batches to evaluate the model later
# check https://deeplizard.com/learn/video/LhEMXbjGV_4 for unlabeled test case
test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
    .flow_from_directory(directory=test_path, target_size=(224, 224), classes=["cat", "dog"], batch_size=10, shuffle=False)

# assert number of images in batches
assert train_batches.n == 1000
assert valid_batches.n == 200
assert test_batches.n == 100
assert train_batches.num_classes == valid_batches.num_classes == test_batches.num_classes == 2

# take one batch from the train set
imgs, labels = next(train_batches)

# function to plot the images


def plot_images(images_arr):
    fig, axes = plt.subplots(1, 10, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()


# plot_images(imgs)     not gonna plot and print every time
# print(labels)

# BUILD THE MODEL
model = Sequential([
    # 32 for filters is arbitrary, kernel_size is common, padding="same" means 0 padding
    Conv2D(filters=32, kernel_size=(3, 3), activation="relu",
           padding="same", input_shape=(224, 224, 3)),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Conv2D(filters=64, kernel_size=(3, 3), activation="relu", padding="same"),
    MaxPool2D(pool_size=(2, 2), strides=2),
    Flatten(),
    Dense(units=2, activation="softmax")
])

model.summary()

# COMPILE AND TRAIN
# since we have only 2 classes, we can also use binary_crossentropy as loss function
# in that case we change the output layer to 1 unit and the activation function to sigmoid
model.compile(optimizer=Adam(learning_rate=0.0001),
              loss="categorical_crossentropy", metrics=["accuracy"])

# train the model
# commented out to avoid training every time
# model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)

# INFERENCES
# get a batch from test data
test_imgs, test_labels = next(test_batches)

# get prediction
predictions = model.predict(x=test_batches, verbose=0)

# CONFUSION MATRIX


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


cm = confusion_matrix(y_true=test_batches.classes,
                      y_pred=np.argmax(predictions, axis=-1))

cm_plot_labels = ["cat", "dog"]
plot_confusion_matrix(cm=cm, classes=cm_plot_labels)
