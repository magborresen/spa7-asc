import os
import glob
import re
import numpy as np
import math
import soundfile as sf
import librosa
import warnings
import logging
import argparse
from matplotlib.image import imsave
from tqdm import tqdm
from scipy.io import wavfile
from scipy import signal
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore", '.*Chunk*.')
warnings.filterwarnings("error")
_LOG = logging.getLogger(__name__)

class preprocess():
    def __init__(self, path, chunk_size=10):
        """ Intitialize the object with path to the training data

        Args:
            path (String): Path to training data folder

        Returns:
            no value
        """

        self._path = path
        self._dirname = os.path.dirname(__file__)
        self._classes = ['office', 'outside', 'semi_outside', 'inside', 'inside_vehicle']
        self._audio_files = glob.glob(os.path.join(self._path, '**/*.wav'), recursive=True)
        self._sample_rate = None
        self._chunk_size = chunk_size
        self.train_data = None
        self.train_labels = None
        self.train_img = []
        self.vali_data = None
        self.vali_labels = None
        self.vali_img = []
        self.test_data = None
        self.test_labels = None
        self.test_img = []
        self.noise_data = None
        self.noise_labels = None
        self.noise_img = []

    def make_training_data(self, method="spectrogram", add_noise=None,
                            save_img=False, test_size=0.1, vali_size=0.1):

        """ Finds all training data and labels classes

        The function finds all training data and labels the classes.
        It will also preprocess the data into the given transform.
        The function will return arrays if save_img is False.
        Otherwise it will save the data as images

        Args:
            method (String): Preprocessing technique to use
            chunk_size (int): Chunk size to split each file into
            test_size (float): Size of test data given in percentage between 0 and 1
            vali_size (float): Size of validation data given in percentage between 0 and 1

        Returns:
            collected_data (list): List of preprocessed collected data
            class_labels (list): List of class labels
        """

        class_labels = []
        collected_data = []
        _LOG.info(f"Preprocessing data into {method} data")
        pbar = tqdm(self._audio_files)
        for af in pbar:
            # Progress bar, just for show
            processing_name = af.split("/")
            pbar.set_description("Processing %s" % processing_name[-1] + " in " + processing_name[-2])

            # Find associated label files
            dirname = os.path.dirname(os.path.abspath(af))
            label_file = glob.glob(dirname + "/*.txt")
            
            # Get the class
            sample_class = [c for c in self._classes if re.search(r'\b' + c + r'\b', af)]

            data, self._sample_rate = sf.read(af)
            if len(label_file) > 0:
                data = self.rm_labeled_noise(data, label_file[0])
                
            # Create Chunks
            data_chunks = self.chunk_file(data)

            # Create matching labels for the chunks and sort out empty chunks
            for i in range(len(data_chunks)):
                if len(data_chunks[i]) != 0:
                    class_labels.append(sample_class[0])
                    collected_data.append(data_chunks[i])



        # Split training and validation data
        self.train_data, self.vali_data, self.train_labels, self.vali_labels = train_test_split(collected_data, 
                                                                            class_labels,
                                                                            test_size=vali_size,
                                                                            random_state=42)


        # Split training and test data based on the previous split.
        self.train_data, self.test_data, self.train_labels, self.test_labels = train_test_split(self.train_data, 
                                                                            self.train_labels,
                                                                            test_size=test_size,
                                                                            random_state=42)

        # Convert test data using selected noise and image model
        for d in self.test_data:
            if add_noise == "awgn":
                y = self.awgn(d)
            else:
                y = d
            if method.lower() == "spectrogram":
                self.test_img.append(self.spectrogram(y))

        # Convert training data using the selected method
        for d in self.train_data:
            if method.lower() == "spectrogram":
                self.train_img.append(self.spectrogram(d))

        # Convert validation data using the selected method
        for d in self.vali_data:
            if method.lower() == "spectrogram":
                self.vali_img.append(self.spectrogram(d))

        # Save as images of selected
        if save_img:
            self.save_as_img(data_type="training")
            self.save_as_img(data_type="validation")
            self.save_as_img(data_type="test")

        return True

    def make_env_noise(self, method="spectrogram", save_img=False):

        collected_noise = []
        class_labels = []
        _LOG.info(f"Preprocessing data into {method} data")
        pbar = tqdm(self._audio_files)
        for af in pbar:
            # Progress bar, just for show
            processing_name = af.split("/")
            pbar.set_description("Processing %s" % processing_name[-1] + " in " + processing_name[-2])

            # Find associated label files
            dirname = os.path.dirname(os.path.abspath(af))
            label_file = glob.glob(dirname + "/*.txt")
            
            # Get the class
            sample_class = [c for c in self._classes if re.search(r'\b' + c + r'\b', af)]

            data, self._sample_rate = sf.read(af)
            if len(label_file) > 0:
                noise = self.get_labeled_noise(data, label_file[0])

                # Create Chunks
                noise_chunks = self.chunk_file(noise, self._sample_rate, self._chunk_size)

                # Convert data using the given method
                for chunk in noise_chunks:
                    class_labels.append(sample_class[0])
                    if method.lower() == "spectrogram":
                        transform = self.spectrogram(self.noise_data)
                        self.noise_img.append(transform)
                    collected_noise.append(chunk)

                self.noise_data = collected_noise
                self.noise_labels = class_labels

        # Save as images if selected. 
        if save_img:
            self.save_as_img(data_type="noise")

        return True


    def rm_labeled_noise(self, data, label_file_path):
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

    def get_labeled_noise(self, data, label_file_path):
        noise_data_ch0 = np.array([])
        noise_data_ch1 = np.array([])

        with open(label_file_path, encoding="utf8") as f:
            lines = f.readlines()

        for line in lines:
            time_labels = line.split('\t')
            start_time = int(float(time_labels[0]) * 44100)
            stop_time = int(float(time_labels[1]) * 44100)
            # Check if data is two channel
            if data.ndim > 1:
                noise_data_ch0 = np.append(noise_data_ch0, data[:,0][start_time:stop_time])
                noise_data_ch1 = np.append(noise_data_ch1, data[:,1][start_time:stop_time])
                noise_data = np.vstack((noise_data_ch0, noise_data_ch1)).T
            else:
                noise_data_ch0 = np.append(noise_data_ch0, data[start_time:stop_time])
                noise_data = noise_data_ch0

        return noise_data

    def chunk_file(self, data):
        chunk_data = []
        start_sample = 0
        samples_per_chunk = self._chunk_size * self._sample_rate

        for sc in range(math.floor(len(data) / samples_per_chunk)):
            stop_sample = sc * self._chunk_size * self._sample_rate
            # Check if data is two channel
            if data.ndim > 1:
                chunk_ch0 = data[:,0][start_sample:stop_sample]
                chunk_data.append(chunk_ch0)
                chunk_ch1 = data[:,1][start_sample:stop_sample]
                chunk_data.append(chunk_ch1)
            else:
                chunk = data[start_sample:stop_sample]
                chunk_data.append(chunk)
            start_sample = stop_sample

        return chunk_data

    def save_as_img(self, data_type=None):
        """ Saves the given data as images

        The function matches the class data to the training data
        and gives the saved png file as the class name plus some number

        Args:
            data_type (String): Which main folder to save the data to e.g. training

        Returns:
            None
        """
        data_type = data_type.lower()
        if data_type == "training":
            data = self.train_img
            classes = self.train_labels
        elif data_type == "validation":
            data = self.vali_img
            classes = self.vali_labels
        elif data_type == "test":
            data = self.test_img
            classes = self.test_labels
        elif data_type == "noise":
            data = self.noise_img
            classes = self.noise_labels
        else:
            _LOG.error("No datatype given, no data will be saved")
            return False

        _LOG.info(f"Saving {data_type} data as images")
        for i in tqdm(range(len(data))):
            filedir = os.path.join(self._dirname, data_type, classes[i])
            if not os.path.exists(filedir):
                os.makedirs(filedir)
            filename = os.path.join(self._dirname, data_type, classes[i], classes[i] + f"{i}.png")
            imsave(filename, data[i])

    def spectrogram(self, data, nperseg=1024, noverlap=512, window="hann"):
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

        f, t, Sxx = signal.spectrogram(data, self._sample_rate,
                                        nperseg=nperseg, noverlap=noverlap, window=window)

        try:
            return 10*np.log10(Sxx)
        except RuntimeWarning:
            pass

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
            ch0dist[ch0dist > steps] = steps
            # Convert distance vector to matrix
            ch0Z = squareform(ch0dist)

            ch1dist = pdist(ch1data[:,None])
            ch1dist = np.floor(ch1dist/eps)
            ch1dist[ch1dist > steps] = steps
            # Convert distance vector to matrix
            ch1Z = squareform(ch1dist)
            return [ch0Z, ch1Z]

        ch0data = data[:,0]
        ch0dist = pdist(ch0data[:,None])
        ch0dist = np.floor(ch0dist/eps)
        ch0dist[ch0dist > steps] = steps
        # Convert distance vector to matrix
        ch0Z = squareform(ch0dist)

        return ch0Z

    def awgn(self, x):
        """ Additive White Gaussian Noise

        This function adds white gaussian noise to the given signal x.

        Args:
            x (array): Input signal

        Returns:
            The signal with awgn
        """
        wgn = np.random.normal(loc=0.0, scale=1.0, size=x.shape[0])
        return np.add(x, wgn)

def script_invocation():
    """Script invocation."""
    logging.basicConfig(level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(levelname)-8s - %(message)s',
                        datefmt='%H:%M:%S')
    _LOG.setLevel(logging.DEBUG)

    parser = argparse.ArgumentParser(description="Preprocess and split data into training, validation and test")

    parser.add_argument('-mt', "--make_training", help="Output the training, test and validation data", action="store_true")
    parser.add_argument("-cs", "--chunk_size", nargs="?", help="Splits the data into the given chunk sizes", type=int, default=10)
    parser.add_argument("-n", "--add_noise", help="choose to add noise to the signal", action="store_true")
    parser.add_argument("-s", "--save_img", help="Save data as images", action="store_true")
    parser.add_argument("-ts" "--test_size", nargs="?", help="Split into test size (between 0 and 1)", type=float, default=0.1)
    parser.add_argument("-vs", "--vali_size", nargs="?", help="Split into validation size (between 0 and 1)", type=float, default=0.1)
    parser.add_argument("-m", "--method", help="Method to convert signals", type=str, default="spectrogram")
    parser.add_argument("-e", "--env_noise", help="Create enviromental noise test data", action="store_true")

    args = parser.parse_args()

    dirname = os.path.dirname(__file__)
    data_path = os.path.join(dirname, "training_data")
    prep = preprocess(data_path, args.chunk_size)

    if args.make_training:
        if args.save_img:
            prep.make_training_data(method=args.method, add_noise=args.add_noise, save_img=args.save_img, test_size=args.test_size, vali_size=args.vali_size)
        else:
            prep.make_training_data(method=args.method, add_noise=args.add_noise, test_size=args.test_size, vali_size=args.vali_size)

    if args.env_noise:
        prep.make_env_noise(method=args.method, save_img=args.save_img)


if __name__ == "__main__":
    script_invocation()