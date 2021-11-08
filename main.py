import os
import warnings
from preprocessing import preprocess
from cnn_model import cnn_model

warnings.filterwarnings("ignore", '.*Chunk*.')


dirname = os.path.dirname(__file__)
training_folder = os.path.join(dirname, "training_data/")


# Use dirname in prep object when running on the server. Otherwise, insert an absolute path to the training data folder
prep = preprocess("C:/Users/mike_/Desktop/KampitakisCode/spa test/")
# "C:/Users/mabo/Aalborg Universitet/P7 - SPA7 - Dokumenter/Project/SPA 7 770 database/"


# "C:/Users/mabo/Aalborg Universitet/P7 - SPA7 - Dokumenter/Project/SPA 7 770 database/"
# "C:/Users/mike_/OneDrive/Desktop/KampitakisCode/spa test/"
data, classes = prep.make_training_data()

# CNN model
spa_cnn_model = cnn_model(data, classes, im_norm_size=(513, 860))
spa_cnn_model.make_model()
#spa_cnn_model.model_summary() uncoment to check model node structure
spa_cnn_model.train_model(model_name = "SPA_model", epoch = 2)
