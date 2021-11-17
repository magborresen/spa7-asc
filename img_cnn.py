import warnings
warnings.filterwarnings("ignore")
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix



class CNN:
    """
    Class represents the image processing CNN

    Attributes:

    """
    # def __init__(self, train_batches, valid_batches, test_batches):
    #     self.train_batches = train_batches
    #     self.valid_batches = valid_batches
    #     self.test_batches = test_batches

    def Create_batches(self, train_path, valid_path, test_path, class_names):
        """
        Creates training batches for the image processing CNN.
        File structure for 2 classes:
        CNN_data
            train
                class1
                    image1
                    ...
                class2
                    image1
                    ...
            validation
                class1
                    image1
                    ...
                class2
                    image1
                    ...
            test
                class1
                    image1
                    ...
                class2
                    image1
                    ...

        :param train_path: str, the directory path to the training data
        :param valid_path: str, the directory path to the validation data
        :param test_path: str, the directory path to the test data
        :param class_names: str array, the classification path names; should be matched with the folder names.
        :return: None
        """

        train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
            .flow_from_directory(directory=train_path, target_size=(513, 860), classes=class_names, batch_size=10)
        valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
            .flow_from_directory(directory=valid_path, target_size=(513, 860), classes=class_names, batch_size=5)
        test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
            .flow_from_directory(directory=test_path, target_size=(513, 860), classes=class_names, batch_size=2,
                                 shuffle=False)
        print(train_batches)
        return train_batches, valid_batches, test_batches

    def Create_CNN_model(self, train_batches, valid_batches, test_batches, model_name):
        """

        :return:
        """
        model = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(513, 860, 3)),
            MaxPool2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPool2D(pool_size=(2, 2), strides=2),
            Flatten(),
            Dense(units=5, activation='softmax')
        ])
        model.summary()
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x=train_batches, validation_data=valid_batches, epochs=2, verbose=2)
        model.save(model_name)
        return model

    def Test_CNN(self, model, test_batches):
        predictions = model.predict(x=test_batches, verbose=0)
        np.round(predictions)
        print(predictions)

        cm = confusion_matrix(y_true=test_batches.classes, y_pred=np.argmax(predictions, axis=-1))
        print(cm)




