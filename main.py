import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
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
    elif target == 'a':
        num_outputs = 9
        y_train = pd.get_dummies(train_class.age, prefix='age')
        y_test = pd.get_dummies(val_class.age, prefix='age')

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

    model.fit(train_img, y_train, batch_size=batch_size, epochs=50, validation_data=(val_img, y_test), callbacks=[tensorboard_callback])

    print(model.evaluate(val_img, y_test))


# train a CNN that has two branches at the end to determine two different attributes
def task4(train_img, train_class, val_img, val_class, target):

    # for now let's just assume the two tasks are always gender and race
    y_train_gender = pd.get_dummies(train_class.gender)
    y_train_race = pd.get_dummies(train_class.race)

    y_test_gender = pd.get_dummies(val_class.gender)
    y_test_race = pd.get_dummies(val_class.race)

    # make each layer
    input = layers.Input(shape=(32,32,1))
    conv1 = layers.Conv2D(filters=40, kernel_size=(5,5), padding='valid', strides=1, activation='relu', name='conv1')(input)
    max1 = layers.MaxPool2D(pool_size=(2,2), name='max1')(conv1)
    flatten = layers.Flatten(name='flatten')(max1)
    fc1 = layers.Dense(100, activation='relu', name='fc1')(flatten)
    fc2 = layers.Dense(100, activation='relu', name='fc2')(flatten)
    out1 = layers.Dense(2, activation='softmax', name='out1')(fc1) # assuming task1 is gender
    out2 = layers.Dense(7, activation='softmax', name='out2')(fc2) # assuming task2 is race

    # now make the model
    model = Model(inputs=input, outputs=[out1, out2], name='model')

    # define the different losses + other hyperparams
    losses = {
	    "out1": "categorical_crossentropy",
	    "out2": "categorical_crossentropy",
    }
    batch_size = 100
    lr = 0.001
    optimizer = optimizers.Adam(learning_rate=lr)

    model.compile(optimizer=optimizer, loss=losses, metrics=['accuracy'])

    # create tensorboard
    #creating unique name for tensorboard directory
    log_dir = "logs/class/" + datetime.datetime.now().strftime(f"%Y-%m-%d-%H:%M:%S-batch_size={batch_size}-lr={lr}")
    #Tensforboard callback function
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(train_img, {'out1': y_train_gender, 'out2': y_train_race}, batch_size=batch_size, epochs=50, validation_data=(val_img, {'out1': y_test_gender, 'out2': y_test_race}), callbacks=[tensorboard_callback])

    print(model.evaluate(val_img, {'out1': y_test_gender, 'out2': y_test_race}))

def main():
    # do command line arguments here
    if len(sys.argv) < 3:
        print("args: task[1/2/3/4/5] attribute[g/a/r]")
        return
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
    target = sys.argv[2]
    task = sys.argv[1]
    if(task == '1'):
        task1(train_images, train_labels, val_images, valid_labels, target)

    elif(task == '4'):
        task4(train_images, train_labels, val_images, valid_labels, target)



if __name__ == '__main__':
    main()