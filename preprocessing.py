from scipy.io import wavfile
from scipy import signal
from matplotlib import pyplot as plt
import numpy as np

class preprocess():
    def __init__(self, path):
        self.path = path
        self.samplerate, self.data = wavfile.read(self.path)
        self.times = np.arange(len(self.data))/float(self.samplerate)

    def spectrogram(self, nfft=1024, noverlap=512, window="hann"):

        # Spectrogram for channel 0
        ch0f, ch0t, ch0Sxx = signal.spectrogram(self.data[:,0], self.samplerate,
                                        nfft=nfft, noverlap=noverlap, window=window)

        ch1f, ch1t, ch1Sxx = signal.spectrogram(self.data[:,0], self.samplerate,
                                        nfft=nfft, noverlap=noverlap, window=window)

        return ch0Sxx, ch1Sxx

    def melSpec(self):
        return None

    def timeSeries(self):
        return None

    def recurrent(self):
        return None