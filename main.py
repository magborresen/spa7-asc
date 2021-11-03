import os
import warnings
from preprocessing import preprocess
from cnn_model import cnn_model

warnings.filterwarnings("ignore", '.*Chunk*.')


dirname = os.path.dirname(__file__)
training_folder = os.path.join(dirname, "training_data/")

# Use dirname in prep object when running on the server. Otherwise, insert an absolute path to the training data folder
prep = preprocess("C:/Users/mike_/Aalborg Universitet/P7 - SPA7 - Documents/Project/SPA 7 770 database")
# "C:/Users/mabo/Aalborg Universitet/P7 - SPA7 - Dokumenter/Project/SPA 7 770 database/"

data, classes = prep.make_training_data()

# CNN model
spa_cnn_model = cnn_model(data, classes)
spa_cnn_model.make_model()
spa_cnn_model.train_model(model_name = "SPA_model", epoch = 4)

print(len(classes))