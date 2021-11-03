"""
This file implements all model procedures
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #this line removes all GPU usage warnings
from os import listdir
from os.path import isfile, join
import numpy as np
from numpy.core.fromnumeric import size
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import matplotlib.pyplot as plts
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D
from keras.models import Sequential

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

class cnn_model:
    """ initializes a CNN classifier model

        Args:
            data (array): a list of numpy arrays with spectogram data
            classes (array): the corresponding class name of each spectogram

        Returns:
            no value
    """
    def __init__(self, data, classes):
        self.data = np.array(data)
        self.classes_all = np.array(classes)
        self.classes = find_unique_classe(classes)
        self.ClassesNum = len(self.classes)
        image_data = np.array(data[50]) # should work with zero later
        self.image_size = image_data.shape
        print("image size -->", self.image_size)

        print('what I feed into rescaling', self.image_size[0])
    
    def make_model(self):
        self.model = Sequential()
        #self.model.add(Rescaling(1./self.image_size[0])) TODO
        self.model.add(Conv2D(32, (3, 3), padding='same',
                         input_shape=(64,64,3)))
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
    
    def train_model(self, model_name = "new_model", epoch = 50):
        self.epoch = epoch
        (self.trainX, self.testX, self.trainY, self.testY) = train_test_split(self.data, self.classes_all,
	        test_size=0.25, stratify=self.classes_all, random_state=42)
        #self.dataset_train = tf.data.Dataset.from_tensor_slices(self.data) #might don't work
        #self.dataset = self.dataset.map(lambda x, y: x) TODO
        # initialize the training data augmentation object
        trainAug = ImageDataGenerator(
        	rotation_range=30,
        	zoom_range=0.15,
        	width_shift_range=0.2,
        	height_shift_range=0.2,
        	shear_range=0.15,
        	horizontal_flip=True,
        	fill_mode="nearest")
        # initialize the validation/testing data augmentation object (which
        # we'll be adding mean subtraction to)
        valAug = ImageDataGenerator()
    
        callbacks = [
            keras.callbacks.ModelCheckpoint("{}_{epoch}.h5"),
        ]
    
        # fit data to the model
        self.model.fit(
            trainAug.flow(self.trainX, self.trainY, batch_size=32),
            epochs=self.epoch,
            callbacks=callbacks,
            validation_data=valAug.flow(self.testX, self.testY)
        )
    
        print('model evaluation')
        # and ecaluation process
        self.model.evaluate(valAug.flow(self.testX, self.testY))

    def model_summary(self):
        self.model.summary()
