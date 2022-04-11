import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras import layers, losses, optimizers, backend
import cv2
import numpy as np
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import datetime
import matplotlib.pyplot as plt


# we'll need some dicts to convert one-hot-encoding back to normal
gender = {
    0: 'Female',
    1: 'Male'
}
race = {
    0: 'Black',
    1: 'East Asian',
    2: 'Indian',
    3: 'Latino_Hispanic',
    4: 'Middle Eastern',
    5: 'Southeast Asian',
    6: 'White'
}
age = {
    0: '0-2',
    1: '10-19',
    2: '20-29',
    3: '3-9',
    4: '30-39',
    5: '40-49',
    6: '50-59',
    7: '60-69',
    8: 'more than 70'
}

# helper function for the VAE pulled from the lecture notes
def sampling(args):
    """Reparameterization trick by sampling from an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """
    #Extract mean and log of variance
    z_mean, z_log_var = args
    #get batch size and length of vector (size of latent space)
    batch = backend.shape(z_mean)[0]
    dim = backend.int_shape(z_mean)[1]
    
    # by default, random_normal has mean = 0 and std = 1.0
    epsilon = backend.random_normal(shape=(batch, dim))
    #Return sampled number (need to raise var to correct power)
    return z_mean + backend.exp(z_log_var) * epsilon

# the first task, which uses an ANN
def task1(train_img, train_class, val_img, val_class, target):
    num_outputs = 0
    y_true = 0
    translation = 0

    if target == 'g':
        num_outputs = 2
        y_train = pd.get_dummies(train_class.gender, prefix='gender')
        y_test = pd.get_dummies(val_class.gender, prefix='gender')
        y_true = val_class.gender
        translation = gender
    elif target == 'r':
        num_outputs = 7
        y_train = pd.get_dummies(train_class.race, prefix='race')
        y_test = pd.get_dummies(val_class.race, prefix='race')
        y_true = val_class.race
        translation = race
    elif target == 'a':
        num_outputs = 9
        y_train = pd.get_dummies(train_class.age, prefix='age')
        y_test = pd.get_dummies(val_class.age, prefix='age')
        y_true = val_class.age
        translation = age

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
    log_dir = "logs/class/" + datetime.datetime.now().strftime(f"%Y/%m/%d-%H-%M-%S-batch_size={batch_size}-lr={lr}")
    #Tensforboard callback function
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    model.fit(train_img, y_train, batch_size=batch_size, epochs=5, validation_data=(val_img, y_test), callbacks=[tensorboard_callback])

    loss, acc = model.evaluate(val_img, y_test)
    y_pred = model.predict(val_img)
    y_pred = np.argmax(y_pred, axis=1)
    new_y_pred = []
    for y in y_pred:
        new_y_pred.append(translation[y])
    y_pred = pd.DataFrame(new_y_pred, columns=['col'])
    
    print(f'Final Accuracy: {round(acc, 2)}')
    print(confusion_matrix(y_true, y_pred.col))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred.col, cmap='Blues')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('fig.jpg')

def task2(train_img, train_class, val_img, val_class, target):
    if target == 'g':
        num_outputs = 2
        y_train = pd.get_dummies(train_class.gender, prefix='gender')
        y_test = pd.get_dummies(val_class.gender, prefix='gender')
        y_true = val_class.gender
        translation = gender
    elif target == 'r':
        num_outputs = 7
        y_train = pd.get_dummies(train_class.race, prefix='race')
        y_test = pd.get_dummies(val_class.race, prefix='race')
        y_true = val_class.race
        translation = race
    elif target == 'a':
        num_outputs = 9
        y_train = pd.get_dummies(train_class.age, prefix='age')
        y_test = pd.get_dummies(val_class.age, prefix='age')
        y_true = val_class.age
        translation = age

    
    train_img = train_img.reshape(86744, 32, 32, 1)
    val_img = val_img.reshape(10954, 32, 32, 1)

    print("Building Model")
    model = Sequential()
    model.add(layers.Conv2D(40, (5, 5), input_shape=((32, 32, 1)), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Flatten())
    model.add(layers.Dense(100, activation='relu'))
    model.add(layers.Dense(num_outputs, activation='softmax'))

    print("setting params")
    lr = .001
    batch_size = 100
    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # create tensorboard
    #creating unique name for tensorboard directory
    log_dir = "logs/class/" + datetime.datetime.now().strftime(f"%Y/%m/%d-%H-%M-%S-batch_size={batch_size}-lr={lr}")
    #Tensforboard callback function
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # train_img =np.expand_dims(train_img, axis=(0,3))
    # val_img =np.expand_dims(val_img, axis=(0,3))

    print("Running model")
    model.fit(train_img, y_train, batch_size=batch_size, epochs=5, validation_data=(val_img, y_test), callbacks=[tensorboard_callback])

    

    print("Collecting data")
    loss, acc = model.evaluate(val_img, y_test)
    y_pred = model.predict(val_img)
    y_pred = np.argmax(y_pred, axis=1)
    new_y_pred = []
    for y in y_pred:
        new_y_pred.append(translation[y])
    y_pred = pd.DataFrame(new_y_pred, columns=['col'])

    print(f'Final Accuracy: {round(acc, 2)}')
    print(confusion_matrix(y_true, y_pred.col))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred.col, cmap='Blues')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('fig.jpg')

def task3(train_img, train_class, val_img, val_class, target):
    if target == 'g':
        num_outputs = 2
        y_train = pd.get_dummies(train_class.gender, prefix='gender')
        y_test = pd.get_dummies(val_class.gender, prefix='gender')
        y_true = val_class.gender
        translation = gender
    elif target == 'r':
        num_outputs = 7
        y_train = pd.get_dummies(train_class.race, prefix='race')
        y_test = pd.get_dummies(val_class.race, prefix='race')
        y_true = val_class.race
        translation = race
    elif target == 'a':
        num_outputs = 9
        y_train = pd.get_dummies(train_class.age, prefix='age')
        y_test = pd.get_dummies(val_class.age, prefix='age')
        y_true = val_class.age
        translation = age

    train_img = train_img.reshape(86744, 32, 32, 1)
    val_img = val_img.reshape(10954, 32, 32, 1)

    print("Building Model")
    model = Sequential()
    model.add(layers.Conv2D(20, (5, 5), input_shape=((32, 32, 1)), activation='relu'))
    model.add(layers.MaxPooling2D(2, 2))
    model.add(layers.Conv2D(40, (3, 3), activation='relu'))
    # model.add(layers.MaxPooling2D(2, 2))
    # model.add(layers.Conv2D(400, (2, 2), activation='relu'))
    model.add(layers.Flatten())
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(num_outputs, activation='softmax'))

    print("setting params")
    lr = .001
    batch_size = 100
    optimizer = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

    # create tensorboard
    #creating unique name for tensorboard directory
    log_dir = "logs/class/" + datetime.datetime.now().strftime(f"%Y/%m/%d-%H-%M-%S-batch_size={batch_size}-lr={lr}")
    #Tensforboard callback function
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    # train_img =np.expand_dims(train_img, axis=(0,3))
    # val_img =np.expand_dims(val_img, axis=(0,3))

    print("Running model")
    model.fit(train_img, y_train, batch_size=batch_size, epochs=5, validation_data=(val_img, y_test), callbacks=[tensorboard_callback])

    

    print("Collecting data")
    loss, acc = model.evaluate(val_img, y_test)
    y_pred = model.predict(val_img)
    y_pred = np.argmax(y_pred, axis=1)
    new_y_pred = []
    for y in y_pred:
        new_y_pred.append(translation[y])
    y_pred = pd.DataFrame(new_y_pred, columns=['col'])

    print(f'Final Accuracy: {round(acc, 2)}')
    print(confusion_matrix(y_true, y_pred.col))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred.col, cmap='Blues')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('fig.jpg')

# train a CNN that has two branches at the end to determine two different attributes
def task4(train_img, train_class, val_img, val_class, target):

    # let's just assume the two tasks are always gender and race
    gender_translation = gender
    race_translation = race

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
    loss, gender_loss, race_loss, gender_acc, race_acc = model.evaluate(val_img, {'out1': y_test_gender, 'out2': y_test_race})

    y_pred = model.predict(val_img)
    print(y_pred[1].shape)
    gender_pred = np.argmax(y_pred[0], axis=1)
    race_pred = np.argmax(y_pred[1], axis=1)
    new_gender_pred = []
    new_race_pred = []
    for g, r in zip(gender_pred, race_pred):
        new_gender_pred.append(gender_translation[g])
        new_race_pred.append(race_translation[r])
    
    gender_pred = pd.DataFrame(new_gender_pred, columns=['gender'])
    race_pred = pd.DataFrame(new_race_pred, columns=['race'])

    print(f'Final Accuracy for gender: {round(gender_acc, 2)}')
    print(f'Final Accuracy for race: {round(race_acc, 2)}\n\n')
    print('Confusion Matrix for gender: (Image can be found in fig_gender.jpg)')
    print(confusion_matrix(val_class.gender, gender_pred.gender))
    print('\nConfusion matrix for race: (Image can be found in fig_race.jpg)')
    print(confusion_matrix(val_class.race, race_pred.race))
    ConfusionMatrixDisplay.from_predictions(val_class.gender, gender_pred.gender, cmap='Blues')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('fig_gender.jpg')
    ConfusionMatrixDisplay.from_predictions(val_class.race, race_pred.race, cmap='Blues')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.savefig('fig_race.jpg')


# this task implements a VAE
def task5(train_img, val_img):
    latent_dim = 5
    batch_size = 100
    lr = 0.001
    optimizer = optimizers.Adam(learning_rate=lr)
    
    # build encoder model
    input = layers.Input(shape=(32, 32, 1))
    conv1 = layers.Conv2D(filters=20, kernel_size=(6,6), padding='valid', strides=1, activation='relu', name='conv1')(input)
    conv2 = layers.Conv2D(filters=40, kernel_size=(5,5), padding='valid', strides=1, activation='relu', name='conv2')(conv1)
    flatten = layers.Flatten()(conv2)
    
    z_mean = layers.Dense(latent_dim, name='z_mean')(flatten)
    z_log_var = layers.Dense(latent_dim, name='z_log_var')(flatten)

    # use reparameterization trick to push the sampling out as input (taken from lecture notes)
    z = layers.Lambda(sampling, name='z')([z_mean, z_log_var])
    
    encoder = Model(input, [z_mean, z_log_var, z], name='encoder_output')
    print(encoder.summary())

    latent_input = layers.Input(shape=(latent_dim,), name='z_sampling')
    fc1 = layers.Dense(21160, activation='relu', name='decoder_dense')(latent_input)
    reshape = layers.Reshape((23, 23, 40))(fc1)
    deconv1 = layers.Conv2DTranspose(20, kernel_size=(5,5), padding='valid', strides=1, activation='relu', name='deconv1')(reshape)
    deconv2 = layers.Conv2DTranspose(1, kernel_size=(6,6), padding='valid', strides=1, activation='relu', name='deconv2')(deconv1)

    decoder = Model(latent_input, deconv2, name='decoder_output')
    print(decoder.summary())

    # create the model and loss function, again pulled from lecture notes
    # instantiate VAE model
    outputs = decoder(encoder(input)[2])
    vae = Model(input, outputs, name='vae_mlp')

    #setting loss
    # inspiration for input dim ^2 and flatten comes from https://jaketae.github.io/study/vae/
    # which I also believe is where the lecture notes came from or at least look very similar to
    reconstruction_loss = 32 * 32 * losses.mse(backend.flatten(input), backend.flatten(outputs))
    reconstruction_loss *=1
    kl_loss = backend.exp(z_log_var) + backend.square(z_mean) - z_log_var - 1
    kl_loss = backend.sum(kl_loss, axis=-1)
    kl_loss *= 0.001
    vae_loss = backend.mean(reconstruction_loss + kl_loss)
    vae.add_loss(vae_loss)
    vae.compile(optimizer=optimizer)
    
    # create tensorboard
    #creating unique name for tensorboard directory
    log_dir = "logs/class/task5/" + datetime.datetime.now().strftime(f"%Y-%m-%d-%H:%M:%S-batch_size={batch_size}-lr={lr}")
    #Tensforboard callback function
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    vae.fit(train_img, epochs=10, batch_size=batch_size, validation_data=(val_img, None), callbacks=[tensorboard_callback])

    # print out some randomly generated images
    arr = np.random.rand(10, 5) * 12.5 - 5
    y_pred = decoder.predict(arr)
    i = 0
    for a in y_pred:
        img = np.reshape(a, (32, 32)) * 255
        f = f'face{i}.jpg'
        print(img)
        cv2.imwrite(f, img)
        i += 1


def main():
    # do command line arguments here
    if len(sys.argv) < 3:
        print("args: task[1/2/3/4/5] attribute[g/a/r]")
        return
    # read in all the data we need to
    
    print("Reading csvs")
    # first read in the train and validation classification results
    train_labels = pd.read_csv('fairface_label_train.csv')
    valid_labels = pd.read_csv('fairface_label_val.csv')

    # now grab the photos
    train_images = []
    min = 255
    max = 0
    print("Reading files")
    for filename in train_labels['file']:
        image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
        if np.max(image) > max:
            max = np.max(image)
        if np.min(image) < min:
            min = np.min(image)
        train_images.append(image)

    print("files read")

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

    elif(task == '2'):
        task2(train_images, train_labels, val_images, valid_labels, target)

    elif(task == '3'):
        task3(train_images, train_labels, val_images, valid_labels, target)

    elif(task == '4'):
        task4(train_images, train_labels, val_images, valid_labels, target)

    elif(task == '5'):
        task5(train_images, val_images)

if __name__ == '__main__':
    main()