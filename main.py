import os
import warnings

from tensorflow import keras

from img_cnn import CNN
#from cnn_model import cnn_model

warnings.filterwarnings("ignore", '.*Chunk*.')


dirname = os.path.dirname(__file__)

training_img_dir = os.path.join(dirname, "training")

# Use dirname in prep object when running on the server. Otherwise, insert an absolute path to the training data folder
prep = preprocess(training_folder)


train_path = training_img_dir
valid_path = os.path.join(dirname, "vali")
test_path = os.path.join(dirname, "test")
train_classes = ['inside', 'inside_vehicle', 'office', 'outside', 'semi_outside']
cnn = CNN()
train_batches, valid_batches, test_batches = cnn.Create_batches(train_path, valid_path, test_path, train_classes)
model_name = 'clean_model'
train = True
if train:
    neural_model = cnn.Create_CNN_model(train_batches, valid_batches, test_batches, model_name)
else:
    neural_model = keras.models.load_model(model_name)

cnn.Test_CNN(neural_model, test_batches)
