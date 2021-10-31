# This program:
# extracts spectograms out of audio
# trains and saves a CNN model

# for this to work, requires audio files inside corresponding enumarete folders
# for use with urban dataset use urban_set_dataset.py first

# for graphical representation of the model:
#- pip install pydot
# install graphviz (https://graphviz.gitlab.io/download/)
# else I think you can just comment out line 145

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #this line removes all GPU usage warnings
from os import listdir
from os.path import isfile, join
import time
import numpy as np
from numpy.core.fromnumeric import size
import gc
import librosa
import librosa.display
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import matplotlib.pyplot as plt
from path import Path
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.models import Sequential

# most of it found in
# https://medium.com/gradientcrescent/urban-sound-classification-using-convolutional-neural-networks-with-keras-theory-and-486e92785df4
# and
# https://keras.io/examples/vision/image_classification_from_scratch/

# Change the path in order to work
path_audio = "C:/Users/mike_/OneDrive/Desktop/KampitakisCode/CNN_quick_test/audio/"
path_spectogr = "C:/Users/mike_/OneDrive/Desktop/KampitakisCode/CNN_quick_test/spector/"
dir_list = os.listdir(path_audio + 'train/') # list of class folders for training

def create_spectrogram(filename,folder_name,name):
    plt.interactive(False)
    clip, sample_rate = librosa.load(filename, sr=None)
    fig = plt.figure(figsize=[0.72,0.72])
    ax = fig.add_subplot(111)
    ax.axes.get_xaxis().set_visible(False)
    ax.axes.get_yaxis().set_visible(False)
    ax.set_frame_on(False)
    S = librosa.feature.melspectrogram(y=clip, sr=sample_rate)
    librosa.display.specshow(librosa.power_to_db(S, ref=np.max))
    filename  = path_spectogr + 'train/' + folder_name + '/' + name + '.jpg'
    if not os.path.exists(path_spectogr + 'train/' + folder_name+'/'):
        os.mkdir(path_spectogr + 'train/' + folder_name+'/')
    plt.savefig(filename, dpi=400, bbox_inches='tight',pad_inches=0)
    plt.close()
    fig.clf()
    plt.close(fig)
    plt.close('all')
    del filename,name,clip,sample_rate,fig,ax,S


def extract_features():
    for folder_name in dir_list:
        print('extracting on folder '+folder_name)
        wav_list = [f for f in listdir(path_audio + 'train/' + folder_name + '/') if isfile(join(path_audio + 'train/' + folder_name + '/', f))] #= os.walk()[3]
        for file in wav_list[:80]: # I made less than 100 of them because of memory errors.
            #Define the filename as is, "name" refers to the JPG, and is split off into the number itself. 
            audio_file_name, name = file,file.split('/')[-1].split('.')[0]
            try:
                create_spectrogram(path_audio + 'train/' + folder_name + '/' + audio_file_name,folder_name,name)
            except:
                print('failed to create spectogram on file', folder_name + '/' + audio_file_name)
        del wav_list
        time.sleep(3) # because the memory fulls fast
        gc.collect()
    gc.collect()        


# do this once in order to extract spectograms
#extract_features() # uncomment it


image_size = (225, 225)
batch_size = 32

# makes a train and a validation set with 80% and 20% of data respectively for fitting later to the model
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path_spectogr + 'train/',
    validation_split=0.2,
    subset="training",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)
val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    path_spectogr + 'train/',
    validation_split=0.2,
    subset="validation",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

# I don't know why 32 :P
train_ds = train_ds.prefetch(buffer_size=32)
val_ds = val_ds.prefetch(buffer_size=32)

# model construction.
# The network architecture consists of 6 convolutional layers with increasing
# filter density in order to best extract the features of each image with each 
# successive layer. The pooling and dropout layers serve to increase 
# computational efficiency and to prevent overfitting, respectively
model = Sequential()
model.add(Rescaling(1./255))
model.add(Conv2D(32, (3, 3), padding='same',
                 input_shape=(64,64,3)))
model.add(BatchNormalization()) # I added this extra
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization()) # I added this extra
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(64, (3, 3), padding='same'))
model.add(BatchNormalization()) # I added this extra
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(BatchNormalization()) # I added this extra
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.5))
model.add(Conv2D(128, (3, 3), padding='same'))
model.add(BatchNormalization()) # I added this extra
model.add(Activation('relu'))
model.add(Conv2D(128, (3, 3)))
model.add(BatchNormalization()) # I added this extra
model.add(Activation('relu'))
#model.add(MaxPooling2D(pool_size=(2, 2))) # I removed this and add Average
model.add(AveragePooling2D((2,2), strides=2, padding="same")) # I added this
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(512))
model.add(BatchNormalization()) # I added this extra
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(len(dir_list), activation='softmax')) # specify the amount of output classes here

model.compile(
    keras.optimizers.Adam(1e-3),
    loss="sparse_categorical_crossentropy", # because I use integers instead of class names
    #loss="categorical_crossentropy",       # use this for use with named classes
    metrics=["accuracy"])

# this just present information on screen
keras.utils.plot_model(model, show_shapes=True)

# This function can:
# Write TensorBoard logs after every batch of training to monitor metrics
# Periodically save the model to disk
# Do early stopping
# Get a view on internal states and statistics of the model during training
callbacks = [
    keras.callbacks.ModelCheckpoint("SPA_Model_{epoch}.h5"),
]

# fit data to the model
model.fit(
    train_ds,
    epochs=4, # 50 or more for the final model
    callbacks=callbacks,
    validation_data=val_ds
)

# print nodes information about model
model.summary()

print('model evaluation')
# and ecaluation process
model.evaluate(val_ds)

