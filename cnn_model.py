"""
This file implements all model procedures
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #this line removes all GPU usage warnings
import warnings
warnings.filterwarnings("ignore")
from os import listdir
from os.path import isfile, join
import numpy as np
import pandas as pd
from numpy.core.fromnumeric import shape, size
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import matplotlib.pyplot as plts
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Resizing
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.models import Sequential
from keras.layers import Conv2D

def find_unique_classe(full_list):
    """ finds the unique classes out of a class list

        for use with preprocessing output before fitting into model

        Args:
            full_list (array): the 'classes' output of preprocesing

        Returns:
            list of unique classes
    """
    unique_list = [] 
    for class_name in full_list:
        # check if exists in unique_list or not
        if class_name not in unique_list:
            unique_list.append(class_name)
    return(unique_list)

def image_size_check(images, labels, im_norm_size=(513, 860)):
    """ check that all objecs on list are im_norm_size
        
        discard non standard images from both the 'image' and 'labels' lists    

        Args:
            images (array): a list of numpy arrays with spectogram data
            classes (array): the corresponding class name of each spectogram
            in_norm_size (array): image size expected

        Returns:
            image_array (array): new list of numpy arrays with only expecter spectogram data 
            classes_array (array): new list of corresponding class name of each spectogram
    """
    image_array = np.array(images)
    classes_array = np.array(labels)
    #find all zero sizes
    zero_indexes = []
    for im_index, image in enumerate(image_array):
        if image.shape!=im_norm_size:
            zero_indexes.append(im_index)
    image_array = np.delete(image_array, zero_indexes)
    classes_array = np.delete(classes_array, zero_indexes)
    return image_array.tolist(), classes_array.tolist()

class cnn_model:
    """ initializes a CNN classifier model

        Args:
            data (array): a list of numpy arrays with spectogram data
            classes (array): the corresponding class name of each spectogram

        Returns:
            no value
    """
    def __init__(self, data, classes_all, im_norm_size=(513, 860)):
        """ model initialization
            
            works for only 1 channel picture (single picture)
            
            Args:
                data (list): a list with preprocess data pictures with shape (numper of images, hight, width)
                classes_all (list): coresponding labels of data list.
                im_norm_size (array): expected picture size (hight, width)

            Returns:
                no value
        """
        print('before --> ', np.shape(data))
        # check size
        data, self.labels = image_size_check(data, classes_all, im_norm_size)
        
        self.data = tf.expand_dims(data, axis=-1) # add channel information to array, τηε 4thn dimemntion that is needed
        
        print('during --> ', np.shape(self.data))
        self.data = np.array(self.data) 
        self.labels = np.array(self.labels)
        
        self.input_shape=(513, 860, 1)
        
        self.classes = find_unique_classe(classes_all) #returns a list with all main classes
        self.ClassesNum = len(self.classes)

        print('after --> ', np.shape(self.data))

    def make_model(self):
        """ sets the NN layers, compiles the model based on optimization function

            Args:
                self

            Returns:
                no value
        """
        self.model = Sequential()
        self.model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=self.input_shape))
        #self.model.add(Conv2D(32, (3, 3), padding='same'))
        self.model.add(BatchNormalization()) # I added this extra
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(BatchNormalization()) # I added this extra
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.25))
        self.model.add(Conv2D(64, (3, 3), padding='same'))
        self.model.add(BatchNormalization()) # I added this extra
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(64, (3, 3)))
        self.model.add(BatchNormalization()) # I added this extra
        self.model.add(Activation('relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2)))
        self.model.add(Dropout(0.5))
        self.model.add(Conv2D(128, (3, 3), padding='same'))
        self.model.add(BatchNormalization()) # I added this extra
        self.model.add(Activation('relu'))
        self.model.add(Conv2D(128, (3, 3)))
        self.model.add(BatchNormalization()) # I added this extra
        self.model.add(Activation('relu'))
        #model.add(MaxPooling2D(pool_size=(2, 2))) # I removed this and add Average
        self.model.add(AveragePooling2D((2,2), strides=2, padding="same")) # I added this
        self.model.add(Dropout(0.5))
        self.model.add(Flatten())
        self.model.add(Dense(512))
        self.model.add(BatchNormalization()) # I added this extra
        self.model.add(Activation('relu'))
        self.model.add(Dropout(0.5))
        self.model.add(Dense(self.ClassesNum, activation='softmax')) # specify the amount of output classes here

        print('compiling model', end="\r")
        # compile the modelselected optimization function (Adam)
        self.model.compile(
            keras.optimizers.Adam(1e-3), # optimization function (Adam) and learn rate
            loss="categorical_crossentropy", # depents on the type of classification
            metrics=["accuracy"])
        print('compiling model - done')
    
    def train_model(self, model_name = "new_model", epoch = 50, validation_split=0.25, batch_size=32):
        """ handles the training process

        Splits the data for train and evaluation processes. Prepares the data for input in CNN and fits them.
        it stores a model in every epoch itteration

            Args:
                model_name (char): output name of the model
                epoch (int): number of iterations of retraining process
                validation_split (float): persent of validation data with range 0. to 1
                batch_size (int): training examples utilized in one iteration

            Returns:
                no value
        """
        self.epoch = epoch
        self.validation_split=validation_split # persent of validation data
        self.batch_size=batch_size
        
        # split data into train and validation data
        # X data are numpy data
        # Y data are coresponding labels
        (self.trainX, self.testX, self.trainY, self.testY) = train_test_split(self.data, self.labels,
	        test_size=validation_split, stratify=self.labels, random_state=42)
        
        # prepares data from numpy arrays into readable form for the model
        datagen_train = ImageDataGenerator()
        datagen_val = ImageDataGenerator()
        
        # stores the model 
        callbacks = [
            keras.callbacks.ModelCheckpoint(model_name+"_{epoch}.h5"),
        ]

        # fits the data into model and starts the training process
        self.model.fit(
                x = datagen_train.flow(
                self.trainX, 
                self.trainY, 
                batch_size=batch_size
                ),
            epochs=self.epoch,
            steps_per_epoch=len(self.trainX) // batch_size,
            callbacks=callbacks,
            validation_data=datagen_val.flow(self.testX, self.testY),
            validation_steps=len(self.testX) // batch_size
        )

        print('model evaluation')
        self.model.evaluate(datagen_val.flow(self.testX, self.testY))
