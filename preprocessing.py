import os
import glob
import re
from scipy.io import wavfile
from scipy import signal


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

    def make_training_data(self, method="spectrogram"):
        """ Finds all training data and labels classes

        The functio finds all training data and labels the classes.
        It will also preprocess the data into the given transform

        Args:
            method (String): Preprocessing technique to use

        Returns:
            collected_data (list): List of preprocessed collected data
            class_labels (list): List of class labels
        """
        files = glob.glob(os.path.join(self.path, '/**/*.wav'), recursive=True)
        collected_data = []
        class_labels = []
        for f in files:
            # Get the class
            sample_class = [c for c in self.classes if re.search(r'\b' + c + r'\b', f)]
            sample_rate, data = wavfile.read(f)
            if method == "spectrogram":
                transform = self.spectrogram(data, sample_rate)

                collected_data.append(transform)
                class_labels.append(sample_class)

        return collected_data, class_labels

    def spectrogram(self, data, sample_rate,  nfft=1024, noverlap=512, window="hann"):
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
        if data.shape[1] > 1:
            # Spectrogram for channel 0
            ch0f, ch0t, ch0Sxx = signal.spectrogram(data[:,0], sample_rate,
                                            nfft=nfft, noverlap=noverlap, window=window)

            ch1f, ch1t, ch1Sxx = signal.spectrogram(data[:,1], sample_rate,
                                            nfft=nfft, noverlap=noverlap, window=window)

            return ch0Sxx, ch1Sxx
        # If 1 channel
        else:

            ch0f, ch0t, ch0Sxx = signal.spectrogram(data, sample_rate,
                                            nfft=nfft, noverlap=noverlap, window=window)

            return ch0Sxx

    def melSpec(self):
        return None

    def timeSeries(self):
        return None

    def recurrent(self):
        return None
