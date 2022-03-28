import sys
import os
import pandas as pd
import tensorflow as tf
import cv2


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

    for file in train_files_list:
        image = cv2.imread(str(path + file), cv2.IMREAD_GRAYSCALE)
        train_images.append(image)

    val_images = []
    path = os.getcwd() + '/val/'
    val_files_list = os.listdir(path)

    for file in val_files_list:
        image = cv2.imread(str(path + file), cv2.IMREAD_GRAYSCALE)
        val_images.append(image)

if __name__ == '__main__':
    main()