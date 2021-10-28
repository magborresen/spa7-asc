import os
import warnings
from preprocessing import preprocess

warnings.filterwarnings("ignore", '.*Chunk*.')

dirname = os.path.dirname(__file__)

prep = preprocess("C:/Users/mabo/Aalborg Universitet/P7 - SPA7 - Dokumenter/Project/SPA 7 770 database/")

data, classes = prep.make_training_data()

print(data)
print(classes)