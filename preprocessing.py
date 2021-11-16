import os
import glob
import re
from random import random
from random import randint
import numpy as np
from numpy.random import default_rng
import math
import soundfile as sf
import librosa
import warnings
from matplotlib.image import imsave
from tqdm import tqdm
from scipy.io import wavfile
from scipy import signal
from scipy.spatial.distance import pdist, squareform
from sklearn.model_selection import train_test_split

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
        self.dirname = os.path.dirname(__file__)
        self.classes = ['office', 'outside', 'semi_outside', 'inside', 'inside_vehicle']

    def make_training_data(self, method="spectrogram", chunk_size=10, 
                            save_img=True, test_size=0.1, vali_size=0.1,
                            packet_loss=True):

        """ Finds all training data and labels classes

        The function finds all training data and labels the classes.
        It will also preprocess the data into the given transform.
        The function will return arrays if save_img is False.
        Otherwise it will save the data as images

        Args:
            method (String): Preprocessing technique to use
            chunk_size (int): Chunk size to split each file into
            save_img (bool): Whether to save the data as images or return them as arrays
            test_size (float): Size of test data given in percentage between 0 and 1
            vali_size (float): Size of validation data given in percentage between 0 and 1

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
            processing_name = af.split("/")
            pbar.set_description("Processing %s" % processing_name[-1] + " in " + processing_name[-2])

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
                        if packet_loss:
                            chunk = self.packet_loss_sim(chunk, sample_rate, loss_distr=0.05, loss_type='random')
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

        # Split training and validation data
        train_data, vali_data, train_labels, vali_labels = train_test_split(collected_data, 
                                                                            class_labels,
                                                                            test_size=vali_size,
                                                                            random_state=42)

        # Split training and test data based on the previous split.
        train_data, test_data, train_labels, test_labels = train_test_split(train_data, 
                                                                            train_labels,
                                                                            test_size=test_size,
                                                                            random_state=42)
        
        
        if save_img:
            self.save_as_img(train_data, train_labels, 'training')
            self.save_as_img(vali_data, vali_labels, 'vali')
            self.save_as_img(test_data, test_labels, 'test')
            return None

        else:
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

    def save_as_img(self, data, classes, data_type):
        """ Saves the given data as images

        The function matches the class data to the training data
        and gives the saved png file as the class name plus some number

        Args:
            data (array): Spectrogram image data 2D
            classes (array): Classes with index corresponding to the data
            data_type (String): Which main folder to save the data to e.g. training

        Returns:
            None
        """
        print(f"Saving {data_type} data as images")
        for i in tqdm(range(len(data))):
            filedir = os.path.join(self.dirname, data_type, classes[i])
            if not os.path.exists(filedir):
                os.makedirs(filedir)
            filename = os.path.join(self.dirname, data_type, classes[i], classes[i] + f"{i}.png")
            imsave(filename, data[i])



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

            try:
                return [10*np.log10(ch0Sxx), 10*np.log10(ch1Sxx)]
            except RuntimeWarning:
                try:
                    return 10*np.log10(ch0Sxx)
                except RuntimeWarning:
                    return 10*np.log10(ch1Sxx)

        # If 1 channel
        ch0f, ch0t, ch0Sxx = signal.spectrogram(data, sample_rate,
                                            nperseg=nperseg, noverlap=noverlap, window=window)

        return 10*np.log10(ch0Sxx)

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
    
    def packet_loss_sim(self, np_data, sample_rate, loss_type='random', 
                        loss_distr=0.05, packet_size=0.010):
        """ simulate packet loss on audio data

        adds noise verry close to zero values
        
        Args:
            np_data (array): batch size audio data
            sample_rate (int): audio file sample rate
            loss_type (str, optional): type of loss function. 
                    options: 'random', 'burst'. Defaults to 'random'.
            loss_distr (float, optional): percentage of audio loss of the file. Defaults to 0.05.
            packet_size (float, optional): size of transmited packet in sec. Defaults to 0.010.

        Returns:
            array: modified array
        """
        p_sample_size = packet_size*sample_rate
        packet_data = []
        start_sample = 0
        bernoulli_fun = default_rng() 
        # checks if there is indormation on the chunk file
        if np.size(np_data) != 0: 
            if np_data.ndim > 1:
                numPK = int(np.shape(np_data)[0]/p_sample_size)
            else:
                numPK = int(len(np_data)/p_sample_size)
            
            if loss_type=='random':
                # randomly lose packets
                
                bf = bernoulli_fun.binomial(size=numPK, n=1, p=1-loss_distr) 
                print(bf)
                for pk in range(numPK):
                    stop_sample = start_sample + int(p_sample_size)
                    packet = np_data[start_sample:stop_sample]

                    # multiply zero bernouli samples with noise
                    if bf[pk] == 0:
                        # multiply with near to zero noise
                        packet = packet * np.random.uniform(low=1e-06, high=99e-07, size=np.shape(packet))
                    
                    packet_data.append(packet)
                    start_sample = stop_sample
            
            if loss_type=='burst':
                # lose neighboring packets
                
                probability_thres = 0.9955
                
                totalPKLoss = int(numPK*loss_distr)
                numPK_counter = 0
                burstNumLoss = 0
                for pk in range(numPK):
                    stop_sample = start_sample + int(p_sample_size)
                    packet = np_data[start_sample:stop_sample]
                    
                    prob = random()
                    
                    # ignores the probab fuctor when loss packets left equal to remaining packets of the file
                    remaining_packets = numPK - pk
                    remaining_loss_packets = totalPKLoss - numPK_counter
                    remaining_bool = remaining_loss_packets == remaining_packets  # Bool
                    if (prob >= probability_thres and numPK_counter < totalPKLoss) or remaining_bool or burstNumLoss > 0:
                        if burstNumLoss == 0:
                            # ensures that this will happend once
                            burstNumLoss = randint(1, remaining_loss_packets)
                        else:
                            burstNumLoss = burstNumLoss - 1
                        # multiply with near to zero noise
                        packet = packet * np.random.uniform(low=1e-06, high=99e-07, size=np.shape(packet))

                        numPK_counter +=1
                    
                    packet_data.append(packet)
                    start_sample = stop_sample
                    
            
            np_data = np.concatenate(packet_data)
            
        return np_data
