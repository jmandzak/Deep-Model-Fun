import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers, losses, metrics, optimizers
import cv2
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import datetime
import matplotlib.pyplot as plt

# the first task, which uses an ANN
def task1(train_img, train_class, val_img, val_class, target):
    num_outputs = 0

    if target == 'g':
        num_outputs = 2
        y_train = pd.get_dummies(train_class.gender, prefix='gender')
        y_test = pd.get_dummies(val_class.gender, prefix='gender')
    elif target == 'r':
        num_outputs = 7
        y_train = pd.get_dummies(train_class.race, prefix='race')
        y_test = pd.get_dummies(val_class.race, prefix='race')

    # reshape images to make them 1d
    train_img = train_img.reshape(86744, 1024)
    val_img = val_img.reshape(10954, 1024)
    
    # TODO: figure out which tasks to implement based on user input
    model = Sequential()
    model.add(layers.Dense(1024, input_shape=(1024,), activation='tanh'))
    model.add(layers.Dense(512, activation='sigmoid'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(num_outputs, activation='softmax'))

    lr = 0.001
    batch_size = 100
    # optimizer = optimizers.SGD(learning_rate=lr)
    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # create tensorboard
    #creating unique name for tensorboard directory
    log_dir = "logs/class/" + datetime.datetime.now().strftime(f"%Y/%m/%d-%H:%M:%S-batch_size={batch_size}-lr={lr}")
    #Tensforboard callback function
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(train_img, y_train, batch_size=batch_size, epochs=100, validation_data=(val_img, y_test), callbacks=[tensorboard_callback])

    print(model.evaluate(val_img, y_test))



def main():
    # do command line arguments here

    # read in all the data we need to
    
    # first read in the train and validation classification results
    train_labels = pd.read_csv('fairface_label_train.csv')
    valid_labels = pd.read_csv('fairface_label_val.csv')

    # now grab the photos
    train_images = []
    min = 255
    max = 0
    for filename in train_labels['file']:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if np.max(image) > max:
            max = np.max(image)
        if np.min(image) < min:
            min = np.min(image)
        train_images.append(image)

    val_images = []
    for filename in valid_labels['file']:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        val_images.append(image)
    
    # convert to numpy arrays
    train_images = np.asarray(train_images)
    val_images = np.asarray(val_images)

    # do min max scaling
    train_images = (train_images.astype('float32') - min) / (max - min)
    val_images = (val_images.astype('float32') - min) / (max - min)

    # TODO: replace 1 with command line args later
    target = 'g'
    if(1):
        task1(train_images, train_labels, val_images, valid_labels, target)



if __name__ == '__main__':
    main()