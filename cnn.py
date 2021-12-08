import warnings
warnings.filterwarnings("ignore")
import logging
import argparse
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from keras.models import Model
from tensorflow.keras.layers import Activation, Dense, Flatten, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import CSVLogger
from tensorflow.keras.metrics import categorical_crossentropy
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score

_LOG = logging.getLogger(__name__)

class CNN:
    """
    Class represents the image processing CNN

    Attributes:

    """
    def __init__(self, train_path=None, valid_path=None, test_path=None, chunk_size=10):
        self._dirname = os.path.dirname(__file__)
        if train_path == None:
            self._train_path = os.path.join(self._dirname, "training")
            self._vali_path = os.path.join(self._dirname, "validation")
            self._test_path = os.path.join(self._dirname, "test")
        else:
            self._train_path = os.path.join(self._dirname, train_path)
            self._vali_path = os.path.join(self._dirname, valid_path)
            self._test_path = os.path.join(self._dirname, test_path)
        self._class_names = ['inside',
                            'inside_vehicle',
                            'office',
                            'outside',
                            'semi_outside']

        self._train_batches = None
        self._valid_batches = None
        self._test_batches = None
        self._input_shape = (513, 860) if chunk_size == 10 else (513, 85)

        self.create_batches(self._train_path, self._vali_path, self._test_path)

    def create_batches(self, train_path, valid_path, test_path):
        """
        Creates training batches for the image processing CNN.

        :param train_path: str, the directory path to the training data
        :param valid_path: str, the directory path to the validation data
        :param test_path: str, the directory path to the test data
        :param class_names: str array, the classification path names; should be matched with the folder names.
        :return: None
        """

        train_path = self._train_path
        valid_path = self._vali_path
        test_path = self._test_path

        _LOG.info("Creating batches...")
        self._train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
            .flow_from_directory(directory=train_path, target_size=self._input_shape, classes=self._class_names, batch_size=10)
        self._valid_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
            .flow_from_directory(directory=valid_path, target_size=self._input_shape, classes=self._class_names, batch_size=5)
        self._test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
            .flow_from_directory(directory=test_path, target_size=self._input_shape, classes=self._class_names, batch_size=2,
                                 shuffle=False)
        #print(train_batches)
        #return train_batches, valid_batches, test_batches
        _LOG.info("Batches created...")
        return True

    def create_CNN_model(self, model_name, epochs):
        """

        :return:
        """
        filename = f"{model_name}_{epochs}_epochs" + f"_log" + f".csv"
        history_logger = CSVLogger(os.path.join(self._dirname, "model_history", filename), separator=",", append=True)
        model = Sequential([
            Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(*self._input_shape, 3)),
            MaxPool2D(pool_size=(2, 2), strides=2),
            Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'),
            MaxPool2D(pool_size=(2, 2), strides=2),
            Flatten(),
            Dense(units=5, activation='softmax')
        ])
        model.summary()
        model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x=self._train_batches, validation_data=self._valid_batches, epochs=epochs, callbacks=[history_logger], verbose=1)
        model_path = os.path.join(self._dirname, "models", f"{model_name}_{epochs}_epochs")
        model.save(model_path)
        return model

    def test_CNN(self, model_name):
        model_path = os.path.join(self._dirname, "models", model_name)
        model = load_model(model_path)
        predictions = model.predict(x=self._test_batches, verbose=0)
        np.round(predictions)
        cm = confusion_matrix(y_true=self._test_batches.classes, y_pred=np.argmax(predictions, axis=-1), normalize='true', labels=[0, 1, 2, 3, 4])
        accu = accuracy_score(self._test_batches.classes, np.argmax(predictions, axis=-1))
        print(cm)
        _LOG.info(f"Model accuracy on test data {accu}")
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self._class_names)
        disp.plot(cmap=plt.cm.Blues)
        disp_dist = os.path.join(self._dirname, model_name)
        plt.tight_layout()
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")
        plt.savefig(disp_dist)

    def plot_filters(self, model_path):
        model = load_model(model_path)
        # retrieve weights from the second hidden layer
        filters, biases = model.layers[0].get_weights()
        # normalize filter values to 0-1 so we can visualize them
        f_min, f_max = filters.min(), filters.max()
        filters = (filters - f_min) / (f_max - f_min)
        # plot first few filters
        n_filters, ix = 16, 1
        for i in range(n_filters):
            # get the filter
            f = filters[:, :, :, i]
            # plot each channel separately
            # specify subplot and turn of axis
            ax = plt.subplot(n_filters, 4, ix)
            ax.set_xticks([])
            ax.set_yticks([])
            # plot filter channel in grayscale
            plt.imshow(f)
            ix += 1
        # show the figure
        plt.show()

    def plot_feature_maps(self, model_path):
        model = load_model(model_path)
        model = Model(inputs=model.inputs, outputs=model.layers[0].output)
        model.summary()
        img = load_img("inside10000.png")
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        feature_maps = model.predict(img)
        square = 8
        square2 = 4
        ix = 1
        plt.figure(figsize=(8, 20))
        for _ in range(square):
            for _ in range(square2):
                ax = plt.subplot(square, square, ix)
                ax.set_xticks([])
                ax.set_yticks([])
                plt.imshow(feature_maps[0, :, :, ix-1])
                ix += 1

        plt.tight_layout()
        plt.savefig("output.png")


def script_invocation():
    """Script invocation."""
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(levelname)-8s - %(message)s',
                        datefmt='%H:%M:%S')
    _LOG.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(description="Train or test the CNN")

    parser.add_argument("-f", "--batch_folders",
                        help="Create batches using the given folder names for training, validation and testing",
                        nargs=3, type=str)
    parser.add_argument("-n", "--model_name",
                        help="Name of the model to store to / read from",
                        type=str)
    parser.add_argument("-c", "--create_model", help="Creates a CNN model with the name",
                        action="store_true")
    parser.add_argument("-t", "--test_model", help="Test a model with the given name",
                        action="store_true")
    parser.add_argument("-e", "--epochs", help="Number of epochs to train", type=int, default=2)
    parser.add_argument("-cs", "--chunk_size", help="Chunk size defining input shape", type=int, default=10)
    parser.add_argument("-pf", "--plot_filters", help="Plot the filters from the given model", type=str, default=None)
    parser.add_argument("-po", "--plot_output", help="Plot the feature map of the first layer", type=str, default=None)
    args = parser.parse_args()


    if args.create_model:
        _LOG.info("Creating CNN model")
        if args.batch_folders != None:
            model = CNN(args.batch_folders[0], args.batch_folders[1], args.batch_folders[2], args.chunk_size)
        else:
            model= CNN()
        model.create_CNN_model(args.model_name, args.epochs)

    if args.test_model:
        _LOG.info("Running test data through model")
        if args.batch_folders != None:
            model = CNN(args.batch_folders[0], args.batch_folders[1], args.batch_folders[2], args.chunk_size)
        else:
            model = CNN()
        model.test_CNN(args.model_name)

    if args.plot_filters is not None:
        model = CNN()
        model.plot_filters(args.plot_filters)

    if args.plot_output is not None:
        model = CNN()
        model.plot_feature_maps(args.plot_output)

if __name__ == "__main__":
    script_invocation()
        

