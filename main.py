import os
import warnings
from preprocessing import preprocess
from img_cnn import CNN
#from cnn_model import cnn_model

warnings.filterwarnings("ignore", '.*Chunk*.')


dirname = os.path.dirname(__file__)
training_folder = os.path.join(dirname, "training_data/")

# Check if training folder/data already exists
training_img_dir = os.path.join(dirname, "training")
if not os.path.exists(training_img_dir):
    # Use dirname in prep object when running on the server. Otherwise, insert an absolute path to the training data folder
    prep = preprocess("C:/Users/kissg/Desktop/Training")
    # "C:/Users/mabo/Aalborg Universitet/P7 - SPA7 - Dokumenter/Project/SPA 7 770 database/"
    # "C:/Users/mabo/Aalborg Universitet/P7 - SPA7 - Dokumenter/Project/SPA 7 770 database/"
    # "C:/Users/mike_/OneDrive/Desktop/KampitakisCode/spa test/"
    prep.make_training_data()

# CNN model
#spa_cnn_model = cnn_model(data, classes, im_norm_size=(513, 860))
#test_data, test_labels = spa_cnn_model.make_test_data(10)
#spa_cnn_model.make_model()
#spa_cnn_model.model_summary() # uncoment to check model node structure
#spa_cnn_model.train_model(model_name = "SPA_model", epoch = 2)

#predictions = spa_cnn_model.predict(test_data)
#spa_cnn_model.comf_matrix(predictions, test_labels)



train_path = training_img_dir
valid_path = os.path.join(dirname, "vali")
test_path = os.path.join(dirname, "test")
train_classes = ['inside', 'inside_vehicle', 'office', 'outside', 'semi_outside']
cnn = CNN()
train_batches, valid_batches, test_batches = cnn.Create_batches(train_path, valid_path, test_path, train_classes)
neural_model = cnn.Create_CNN_model(train_batches, valid_batches, test_batches)
cnn.Test_CNN(neural_model, test_batches)
