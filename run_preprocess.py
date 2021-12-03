import os
import warnings
from preprocessing import preprocess




#dirname = os.path.dirname(__file__)
dirname = "C:/Users/mike_/Desktop/KampitakisCode/spa test/"
training_folder = os.path.join(dirname, "training_data/")

# Check if training folder/data already exists
training_img_dir = os.path.join(dirname, "training")
if not os.path.exists(training_img_dir):
    # Use dirname in prep object when running on the server. Otherwise, insert an absolute path to the training data folder
    prep = preprocess("C:/Users/bjark/Documents/Uni/OneDrive/OneDrive - Aalborg Universitet/SPA7/Project/Training database")
    # "C:/Users/mabo/Aalborg Universitet/P7 - SPA7 - Dokumenter/Project/SPA 7 770 database/"
    # "C:/Users/mabo/Aalborg Universitet/P7 - SPA7 - Dokumenter/Project/SPA 7 770 database/"
    # "C:/Users/mike_/Desktop/KampitakisCode/spa test/training_data"

    prep.make_training_data(save_img=True, add_wind=True)
    #prep.make_env_noise()
