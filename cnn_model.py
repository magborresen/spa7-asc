"""
This file implements all model procedures
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' #this line removes all GPU usage warnings
from contextlib import suppress
import warnings
warnings.filterwarnings("ignore")
from os import listdir
from os.path import isfile, join
import numpy as np
from tqdm import tqdm
from numpy.core.fromnumeric import size
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import matplotlib.pyplot as plts
from keras.layers import Dense, Activation, Flatten, Dropout, BatchNormalization, Resizing
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

def image_size_check(images, labels, im_norm_size=(513, 860)):
    """ check that all objecs on list are im_norm_size

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
    return image_array, classes_array

class cnn_model:
    """ initializes a CNN classifier model

        Args:
            data (array): a list of numpy arrays with spectogram data
            classes (array): the corresponding class name of each spectogram

        Returns:
            no value
    """
    def __init__(self, data, classes_all, im_norm_size=(513, 860)):
        self.image_size = 224
        # check size
        self.data, self.labels = image_size_check(data, classes_all, im_norm_size)
        self.data = np.array(self.data)
        self.labels = np.array(self.labels)
        self.classes = find_unique_classe(classes_all)
        self.ClassesNum = len(self.classes)
        
        # prepare data
        
        # transforming class names into integers
        #lb = LabelEncoder()
        #self.labels = lb.fit_transform(self.labels)
        #self.labels = to_categorical(self.labels, self.ClassesNum)

        # transforming rgb scale into gray (range 0 to 1)
        #self.data = self.data/255.0
        self.data = np.resize(self.data, (self.data.shape[0], self.image_size, self.image_size)) # resizing the image doesn't work
    
    def make_model(self):
        """ prepare and compile the model

        Args:
            self
        
        Returns:
            self
        """
        self.model = Sequential()
        #self.model.add(Rescaling(1./255))
        self.model.add(Resizing(self.image_size, self.image_size))
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
        print('compiling model', end="\r")
        self.model.compile(
            keras.optimizers.Adam(1e-3),
            loss="categorical_crossentropy",
            metrics=["accuracy"])
        print('compiling model - done')
    
    def train_model(self, model_name = "new_model", epoch = 50, validation_split=0.25, batch_size=32):
        self.epoch = epoch
        (self.trainX, self.testX, self.trainY, self.testY) = train_test_split(self.data, self.labels,
	        test_size=validation_split, stratify=self.labels, random_state=42)
        #self.dataset_train = tf.data.Dataset.from_tensor_slices(self.data) #might don't work
        #self.dataset = self.dataset.map(lambda x, y: x) TODO
        # initialize the training data augmentation object
        datagen_train = ImageDataGenerator(
            #validation_split = validation_split, 
            rescale=1. / 255,
            zca_whitening=True,
            data_format="channels_first"
            )
        
        datagen_val = ImageDataGenerator(
            rescale=1. / 255,
            data_format="channels_first"
            )
    
        callbacks = [
            keras.callbacks.ModelCheckpoint(model_name+"_{epoch}.h5"),
        ]
        
        # (std, mean, and principal components if ZCA whitening is applied)
        datagen_train.fit(self.trainX)

        # fit data to the model
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
        # and ecaluation process
        self.model.evaluate(valAug.flow(self.testX, self.testY))

    def model_summary(self):
        """ summary the models nodes

        Args:
            self
        
        Returns:
            prints the characteristics
        """
        self.model.summary()
