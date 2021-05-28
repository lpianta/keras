# import libraries
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Dense, Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.applications import imagenet_utils
from sklearn.metrics import confusion_matrix
import itertools
import os
import shutil
import random
import matplotlib.pyplot as plt
from tensorflow.python.keras.backend import count_params

# DOWNLOAD MOBILENET
mobile = tf.keras.applications.mobilenet.MobileNet()

# preprocess image for predictions


def preprocess_img(file):
    img_path = "data/mobilenet_samples/"
    img = image.load_img(img_path + file, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array_expanded = np.expand_dims(img_array, axis=0)
    return tf.keras.applications.mobilenet.preprocess_input(img_array_expanded)


# preprocessed_img = preprocess_img("1.jpg")
# predictions = mobile.predict(preprocessed_img)
# results = imagenet_utils.decode_predictions(predictions)
# print(results)

# organize data on disk
os.chdir("datasets/sign_language_digits")
if os.path.isdir('train/0/') is False:
    os.mkdir("train")
    os.mkdir("valid")
    os.mkdir("test")

    for i in range(0, 10):
        shutil.move(f"{i}", "train")
        os.mkdir(f"valid/{i}")
        os.mkdir(f"test/{i}")

        valid_samples = random.sample(os.listdir(f"train/{i}"), 30)
        for j in valid_samples:
            shutil.move(f"train/{i}/{j}", f"valid/{i}")

        test_samples = random.sample(os.listdir(f"train/{i}"), 5)
        for k in test_samples:
            shutil.move(f"train/{i}/{k}", f"test/{i}")
os.chdir("../..")

# define path
train_path = "datasets/sign_language_digits/train"
valid_path = "datasets/sign_language_digits/valid"
test_path = "datasets/sign_language_digits/test"

# create batches
train_batches = image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)\
    .flow_from_directory(directory=train_path, target_size=(224, 224), batch_size=10)
valid_batches = image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)\
    .flow_from_directory(directory=valid_path, target_size=(224, 224), batch_size=10)
test_batches = image.ImageDataGenerator(preprocessing_function=tf.keras.applications.mobilenet.preprocess_input)\
    .flow_from_directory(directory=test_path, target_size=(224, 224), batch_size=10, shuffle=False)

# MODIFY MODEL
mobile.summary()
# define which layer we want to keep
x = mobile.layers[-6].output
# define custom output layer
# to this output layer, pass all the layers defined in x
output = Dense(units=10, activation="softmax")(x)

# create the actual model from the layer we defined
model = Model(inputs=mobile.input, outputs=output)

# freeze layers
for layer in model.layers[:-23]:
    layer.trainable = False
model.summary()

# compile the model
model.compile(optimizer=Adam(lr=0.0001),
              loss="categorical_crossentropy", metrics=["accuracy"])

# TRAIN the model
model.fit(x=train_batches, validation_data=valid_batches, epochs=10, verbose=2)
