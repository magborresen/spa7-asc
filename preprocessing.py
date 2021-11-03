import os
import glob
import re
import numpy as np
import math
import soundfile as sf
import librosa
import warnings
from tqdm import tqdm
from scipy.io import wavfile
from scipy import signal
from scipy.spatial.distance import pdist, squareform

warnings.filterwarnings("error")



class preprocess():
    def __init__(self, path):
        """ Intitialize the object with path to the training data

        Args:
            path (String): Path to training data folder

        Returns:
            no value
        """

        self.path = path
        self.classes = ['office', 'outside', 'semi_outside', 'inside', 'inside_vehicle']

    def make_training_data(self, method="spectrogram", chunk_size=10):
        """ Finds all training data and labels classes

        The functio finds all training data and labels the classes.
        It will also preprocess the data into the given transform

        Args:
            method (String): Preprocessing technique to use
            chunk_size (int): Chunk size to split each file into

        Returns:
            collected_data (list): List of preprocessed collected data
            class_labels (list): List of class labels
        """

        audio_files = glob.glob(os.path.join(self.path, '**/*.wav'), recursive=True)
        collected_data = []
        class_labels = []
        print("\nPreprocessing data into " + method + " data\n")
        pbar = tqdm(audio_files)
        for af in pbar:
            # Progress bar, just for show
            pbar.set_description("Processing %s" % af)

            # Find associated label files
            dirname = os.path.dirname(os.path.abspath(af))
            label_file = glob.glob(dirname + "/*.txt")
            
            # Get the class
            sample_class = [c for c in self.classes if re.search(r'\b' + c + r'\b', af)]

            data, sample_rate = sf.read(af)
            if len(label_file) > 0:
                data = self.rm_labeled_noise(data, sample_rate, label_file[0])

            # Create Chunks
            data_chunks = self.chunk_file(data, sample_rate, chunk_size)

            for chunk in data_chunks:
                
                try:
                    if method == "spectrogram":
                        transform = self.spectrogram(chunk, sample_rate)
                except UserWarning:
                    continue

                class_labels.append(sample_class[0])

                # Check if returned data is two channels
                if type(transform) == list:
                    collected_data.append(transform[0])
                    collected_data.append(transform[1])
                    class_labels.append(sample_class[0])
                else:
                    collected_data.append(transform)

        return collected_data, class_labels

    def rm_labeled_noise(self, data, sample_rate, label_file_path):
        cut_data_ch0 = np.array([])
        cut_data_ch1 = np.array([])

        start_time = 0

        with open(label_file_path, encoding="utf8") as f:
            lines = f.readlines()

        for line in lines:
            time_labels = line.split('\t')
            stop_time = int(float(time_labels[0]) * 44100)
            # Check if data is two channel
            if data.ndim > 1:
                cut_data_ch0 = np.append(cut_data_ch0, data[:,0][start_time:stop_time])
                cut_data_ch1 = np.append(cut_data_ch1, data[:,1][start_time:stop_time])
                cut_data = np.vstack((cut_data_ch0, cut_data_ch1)).T
            else:
                cut_data_ch0 = np.append(cut_data_ch0, data[start_time:stop_time])
                cut_data = cut_data_ch0
            start_time = int(float(time_labels[1]) * 44100)

        return cut_data

    def chunk_file(self, data, sample_rate, chunk_size):
        chunk_data = []
        start_sample = 0
        samples_per_chunk = chunk_size * sample_rate

        for sc in range(math.floor(len(data) / samples_per_chunk)):
            stop_sample = sc * chunk_size * sample_rate
            # Check if data is two channel
            if data.ndim > 1:
                chunk_ch0 = data[:,0][start_sample:stop_sample]
                chunk_ch1 = data[:,1][start_sample:stop_sample]
                chunk = np.vstack((chunk_ch0, chunk_ch1)).T
            else:
                chunk = data[start_sample:stop_sample]
            chunk_data.append(chunk)
            start_sample = stop_sample

        return chunk_data


    def spectrogram(self, data, sample_rate,  nperseg=1024, noverlap=512, window="hann"):
        """ Compute the spectrogram of a given signal

        This will compute the spectrogram for both channels of the signal.

        Args:
            data (array): Sampled data
            Fs (int): Sample rate
            nfft (int): fft bin size
            noverlap (int): numer of samples to overlap
            window (String): Window function to use

        Returns:
            Spectrogram of both channels as ndarray
        """
        # If 2 channels
        if data.ndim > 1:
            # Spectrogram for channel 0
            ch0f, ch0t, ch0Sxx = signal.spectrogram(data[:,0], sample_rate,
                                            nperseg=nperseg, noverlap=noverlap, window=window)

            ch1f, ch1t, ch1Sxx = signal.spectrogram(data[:,1], sample_rate,
                                            nperseg=nperseg, noverlap=noverlap, window=window)

            return [ch0Sxx, ch1Sxx]

        # If 1 channel
        ch0f, ch0t, ch0Sxx = signal.spectrogram(data, sample_rate,
                                            nperseg=nperseg, noverlap=noverlap, window=window)

        return ch0Sxx

    def mel_spectrogram(self, data, sample_rate, nfft=1024, startframe=0):
        """ Compute the mel spectrogram of a given signal

        This will compute the mel spectrogram for both channels of the signal.

        Args:
            data (array): Sampled data
            sample_rate (int): Sample rate
            nfft (int): fft bin size
            startframe (int): Which sample to start from
            endframe (int): Which sample to end with

        Returns:
            Mel spectrogram of both channels as ndarray
        """
        endframe=data.size - 1
        # Make sure that data type is 32b float
        if data.dtype != np.float32:
            data = data.astype("float32") / np.iinfo(data.dtype).max

        # If 2 channels
        if data.ndim > 1:
            # Spectrogram for channel 0
            ch0Sxx = librosa.power_to_db(librosa.feature.melspectrogram(
                y=data[:, 0][startframe:endframe],
                sr=sample_rate,
                n_fft=nfft,
                fmax=sample_rate / 2))

            # Spectrogram for channel 1
            ch1Sxx = librosa.power_to_db(librosa.feature.melspectrogram(
                y=data[:, 1][startframe:endframe],
                sr=sample_rate,
                n_fft=nfft,
                fmax=sample_rate / 2))
            return [ch0Sxx, ch1Sxx]

        # If 1 channel
        return librosa.power_to_db(librosa.feature.melspectrogram(
            y=data[startframe:endframe],
            sr=sample_rate,
            n_fft=nfft,
            fmax=sample_rate / 2))

    def hz2mel(self, f):
        """ Convert frequency scale from hertz to mel

        This will convert the frequency of a computed spectrogram from hertz to mel.

        Args:
            f (array): Frequency values

        Returns:
            mel frequency scale
        """
        return (1000/np.log(2))*np.log(1+(f/1000))

    def time_series(self):
        return None

    def recurrent(self, data, eps=0.00010, steps=10):
        """ Compute the recurrence plot data of a given signal

        Computes the eucledian distance between samples and compares them to a step size.
        Returns a distance matrix.

        For distance
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html

        For squareform
        https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.squareform.html

        Args:
            data (array): Sampled data
            eps (float): Error between previous trajectory locations
            steps (int): Dunno

        Returns:
            Z (np.ndarray): Recurrence plot data 
        """

        if data.shape[1] > 1:
            ch0data = data[:,0]
            ch1data = data[:,1]
            ch0dist = pdist(ch0data[:,None])
            ch0dist = np.floor(ch0dist/eps)
            ch0dist[dist > steps] = steps
            # Convert distance vector to matrix
            ch0Z = squareform(ch0dist)

            ch1dist = pdist(ch1data[:,None])
            ch1dist = np.floor(ch1dist/eps)
            ch1dist[dist > steps] = steps
            # Convert distance vector to matrix
            ch1Z = squareform(ch1dist)
            return [ch0Z, ch1Z]

        ch0data = data[:,0]
        ch0dist = pdist(ch0data[:,None])
        ch0dist = np.floor(ch0dist/eps)
        ch0dist[dist > steps] = steps
        # Convert distance vector to matrix
        ch0Z = squareform(ch0dist)

        return ch0Z
