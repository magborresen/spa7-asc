import os
import warnings
from preprocessing import preprocess

warnings.filterwarnings("ignore", '.*Chunk*.')


dirname = os.path.dirname(__file__)
training_folder = os.path.join(dirname, "training_data/")

# Use training_folder in prep object when running on the server. Otherwise, insert an absolute path to the training data folder
prep = preprocess("C:/Users/mabo/Aalborg Universitet/P7 - SPA7 - Dokumenter/Project/SPA 7 770 database")

data, classes = prep.make_training_data()

print(len(data), len(classes))
