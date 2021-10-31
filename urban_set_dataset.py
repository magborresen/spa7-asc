# takes files from fold1, fold2 etc folders of UrbanSound8 and moves them in enumarated folders
# I made it so I can test model construction with similar to ours data

# before you run:
# take a number of audio files from random folders of dataset and put all together in one folern
# (for me is /audio/train)

import os
from os import listdir
from os.path import isfile, join
import shutil


test_path = "C:/Users/mike_/OneDrive/Desktop/KampitakisCode/CNN_quick_test/audio/train"

trainfiles = [f for f in listdir(test_path) if isfile(join(test_path, f))]

for idx, train_wav in enumerate(trainfiles):
    file_class = str(train_wav.split("-")[1])
    if not os.path.exists(test_path+'/'+file_class):
        os.mkdir(test_path+'/'+file_class+'/')
    original = test_path+'/'+train_wav
    target = test_path+'/'+file_class+'/'+train_wav
    shutil.move(original,target)
