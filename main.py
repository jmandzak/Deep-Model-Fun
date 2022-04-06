import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import TensorBoard
from tensorflow.keras import layers, losses, metrics, optimizers
import cv2
import numpy as np
from sklearn.preprocessing import LabelBinarizer, OneHotEncoder
from sklearn.metrics import classification_report, confusion_matrix
import datetime

# the first task, which uses an ANN
def task1(train_img, train_class, val_img, val_class):
    
    # reshape images to make them 1d
    train_img = train_img.reshape(86744, 1024)
    val_img = val_img.reshape(10954, 1024)
    
    # TODO: figure out which tasks to implement based on user input
    model = Sequential()
    model.add(layers.Dense(1024, input_shape=(1024,), activation='tanh'))
    model.add(layers.Dense(512, activation='sigmoid'))
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))

    lr = 0.1
    batch_size = 100
    model.compile(optimizer=optimizers.Adam(learning_rate=lr), loss=losses.CategoricalCrossentropy(), metrics=['accuracy'])#metrics.CategoricalAccuracy()])

    print(model.summary())

    # y_train = LabelBinarizer().fit_transform(train_class.gender)
    # y_test = LabelBinarizer().fit_transform(val_class.gender)
    # print(y_train)
    # print(y_test)

    y_train = pd.get_dummies(train_class.gender, prefix='gender')
    y_test = pd.get_dummies(val_class.gender, prefix='gender')
    # y_train = OneHotEncoder().fit_transform(train_class)

    print(model.evaluate(val_img, y_test))
    # print(f'\n\ntest_loss = {test_loss}\ntest_accuracy = {test_accuracy}')
    # y_pred = model.predict(val_img) > 0.5
    # y_pred = np.squeeze(y_pred) * 1
    #print(y_pred)

    # print(classification_report(y_test, y_pred))
    # print(confusion_matrix(y_test, y_pred))

    # create tensorboard
    #creating unique name for tensorboard directory
    log_dir = "logs/class/" + datetime.datetime.now().strftime(f"%Y/%m/%d-%H:%M:%S-batch_size={batch_size}-lr={lr}")
    #Tensforboard callback function
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    print('starting')
    model.fit(train_img, y_train, batch_size=batch_size, epochs=1000, validation_data=(val_img, y_test), callbacks=[tensorboard_callback])
    print('done')

    print(model.evaluate(val_img, y_test))
    # test_loss, test_accuracy = model.evaluate(val_img, y_test)
    # print(f'\n\ntest_loss = {test_loss}\ntest_accuracy = {test_accuracy}')



def main():
    # do command line arguments here

    # read in all the data we need to
    
    # first read in the train and validation classification results
    train_labels = pd.read_csv('fairface_label_train.csv')
    valid_labels = pd.read_csv('fairface_label_val.csv')

    # now grab the photos
    train_images = []
    path = os.getcwd() + '/train/'
    train_files_list = os.listdir(path)
    max = 0
    min = 255

    for file in train_files_list:
        image = cv2.imread(str(path + file), cv2.IMREAD_GRAYSCALE)
        if np.max(image) > max:
            max = np.max(image)
        if np.min(image) < min:
            min = np.min(image)
        train_images.append(image)

    val_images = []
    path = os.getcwd() + '/val/'
    val_files_list = os.listdir(path)

    for file in val_files_list:
        image = cv2.imread(str(path + file), cv2.IMREAD_GRAYSCALE)
        val_images.append(image)

    # convert to numpy arrays
    train_images = np.asarray(train_images)
    val_images = np.asarray(val_images)

    # do min max scaling
    train_images = (train_images.astype('float32') - min) / (max - min)
    val_images = (val_images.astype('float32') - min) / (max - min)

    # now grab the classifications
    train_y_df = pd.read_csv('fairface_label_train.csv')
    val_y_df = pd.read_csv('fairface_label_val.csv')

    # TODO: replace 1 with command line args later
    if(1):
        task1(train_images, train_y_df, val_images, val_y_df)


if __name__ == '__main__':
    main()