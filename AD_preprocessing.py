# Python libraries:
import hashlib
import matplotlib.pyplot as plt
import numpy as np
import os
import random
import sys
from scipy import signal
from scipy.stats import normaltest
from sklearn.covariance import EmpiricalCovariance
# Sound management:
import librosa
import librosa.display
#from pingouin import multivariate_normality
from sklearn.decomposition import PCA

# Initialization:
random.seed(42)
np.random.seed(42)


class Preprocessing:
    
    """
    Initialises AD inference class

    PARAMETERS
    -------
        data (numpy array) - array of signals
        lables (numpy array) - array of labels
        data_type (str) - type of input data(audio, time series, image)
        
    """

    def __init__(self, data = None, labels = None, data_type = None):
        self.data = data
        self.labels = labels
        self.data_type = data_type


    
    def load_files_from_dir(self, dir_name):
        """
        Loading all the .wav files from the directory
        
        PARAMETERS
        -------
            dir_name (string) - directory name with the .wav files
            
        RETURNS
        -------
            signals (list) - list of sound signals
            srs (list) - list of sampling rates
        
        """
        print('current dir is', dir_name)
        signals = []
        srs = []
        for f in os.listdir(dir_name):
            print(f)
            if f.endswith('.wav'):
                signal, sampling_rate = self.load_sound_file(dir_name+f)
                signals += [signal]
                srs += [sampling_rate]
        return signals, srs


    def load_sound_file(self, wav_name, mono=False, channel=0):
        """
        Loading a sound file
        
        PARAMETERS
        -------
            wav_name (string) - location to the WAV file to open
            mono (boolean) - signal is in mono (if True) or Stereo (False, default)
            channel (integer) - which channel to load (default to 0)
        
        RETURNS
        -------
            signal (numpy array) - sound signal
            sampling_rate (float) - sampling rate detected in the file
        """
        multi_channel_data, sampling_rate = librosa.load(wav_name, sr=None, mono=mono)
        signal = np.array(multi_channel_data)[channel, :]
        
        return signal, sampling_rate


    def get_magnitude_scale(self, file, n_fft=1024, hop_length=512):
        """
        Get the magnitude scale from a wav file.
        
        PARAMETERS
        -------
            file (string) - filepath to the location of the WAV file
            n_fft (integer) - length of the windowed signal to compute the short Fourier transform on
            hop_length (integer) - window increment when computing STFT
        RETURNS
        -------
            dB (ndarray) - returns the log scaled amplitude of the sound file
        """
        # Load the sound data:
        signal, sampling_rate = self.load_sound_file(file)

        # Compute the short-time Fourier transform of the signal:
        stft = librosa.stft(signal, n_fft=n_fft, hop_length=hop_length)

        # Map the magnitude to a decibel scale:
        dB = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

        return dB

    def md5(self, fname):
        """
        This function builds an MD5 hash for the file passed as argument.
        
        PARAMETERS
        -------
            fname (string)
                Full path and filename
                
        RETURNS
        -------
            hash (string)
                The MD5 hash of the file
        """
        filesize = os.stat(fname).st_size
        hash_md5 = hashlib.md5()
        with open(fname, "rb") as f:
            for chunk in tqdm(iter(lambda: f.read(4096), b""), total=filesize/4096):
                hash_md5.update(chunk)
                
        return hash_md5.hexdigest()



    def extract_signal_features(self, signal, sr, n_mels=64, frames=5, n_fft=1024, hop_length=512):
        """
        Extract features from a sound signal, given a sampling rate sr. This function 
        computes the Mel spectrogram in log scales (getting the power of the signal).
        Then we build N frames (where N = frames passed as an argument to this function):
        each frame is a sliding window in the temporal dimension.
        
        PARAMETERS
        -------
            signal (array of floats) - numpy array as returned by load_sound_file()
            sr (integer) - sampling rate of the signal
            n_mels (integer) - number of Mel bands to generate (default: 64)
            frames (integer) - number of sliding windows to use to slice the Mel spectrogram
            n_fft (integer) - length of the windowed signal to compute the short Fourier transform on
            hop_length (integer) - number of samples between successive frames
        """
        
        # Compute a mel-scaled spectrogram:
        mel_spectrogram = librosa.feature.melspectrogram(
            y=signal,
            sr=sr,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels
        )
        
        # Convert to decibel (log scale for amplitude):
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # Generate an array of vectors as features for the current signal:
        features_vector_size = log_mel_spectrogram.shape[1] - frames + 1
        
        # Skip short signals:
        dims = frames * n_mels
        if features_vector_size < 1:
            return np.empty((0, dims), np.float32)
        
        # Build N sliding frames and concatenate them to build a feature vector:
        features = np.zeros((features_vector_size, dims), np.float32)
        for t in range(frames):
            features[:, n_mels * t: n_mels * (t + 1)] = log_mel_spectrogram[:, t:t + features_vector_size].T
            
        return features


    def extract_features_from_all_signals(self, data, srs):
        """
        Extract features from all the sound signals
    
        PARAMETERS
        -------
            data (array) - array with signals
        
        RETURNS
        -------
            all_features (list) - list of extracted features of all the signals
        """
    
        all_features = []
        for i in range(data.shape[0]):
            feat = self.extract_signal_features(data[i, :], srs[i])
            all_features += [feat]
        
        return all_features


    def scale_minmax(self, X, min_set=0.0, max_set=1.0):
        """
        Minmax scaler for a numpy array
        
        PARAMETERS
        -------
            X (numpy array) - array to scale
            min (float) - minimum value of the scaling range (default: 0.0)
            max (float) - maximum value of the scaling range (default: 1.0)
        """
        X_std = (X - X.min()) / (X.max() - X.min())
        X_scaled = X_std * (max_set - min_set) + min_set
        return X_scaled


    def cascade(self, hk, J=7):
        """
        Return (x, phi, psi) at dyadic points K/2**J from filter coefficients
        
        PARAMETERS
        -------
            hk - Coefficients of low-pass filter
            J - Values will be computed at grid points K/2**J

        """
        return scipy.signal.cascade(hk, J)



    def wawelet_transform(self, data, widths, wavelet=signal.ricker):
        """
        Performs a continuous wavelet transform on data, using the wavelet function
        
        PARAMETERS
        -------
            wavelet - a wavelet function, should take 2 arguments.
            The first argument is the number of points that the returned vector will have
            (len(wavelet(length,width)) == length). The second is a width parameter, defining the size of the wavelet
            widths - widths to use for transform

        """
        return signal.cwt(sig, wavelet, widths)


    def data_normal(self, data, axis=2, alpha=0.1):
        """
        
        Performs a normality test on the data (by D’Agostino, R. and Pearson)
        
        PARAMETERS
        -------
            data (numpy array) - array of signals
            axis - axis over which the test is performed (dim corresponding to observations number)
            alpha - significance level
            
        RETURNS
        -------
            (boolean) - true if the data is not normal, false if the hypothesis of normality can't be rejected
            
        """
        k2, p = multivariate_normality(data, axis=axis, alpha=alpha)
        if p < alpha:
            return False
        else:
            return True


    def noise_ratio_est(self, data, alpha=0.01, percentile=0.95):
        """
        Estimates ratio of the anomalies (based on Rousseeuw, Croux 2012)
        
        PARAMETERS
        -------
            data (numpy array) - array of signals
            alpha - significance level
        
        RETURNS
        -------
        
        """
        c = 1.4826
        data = np.log(data)
        #is_normal = data_normal(data)
        is_normal = True ### FIX!
        med = np.median(data, axis=2)
        if is_normal:
            S = c*np.median(abs(data-np.reshape(med,
                                        (data.shape[0], data.shape[1], 1))), axis=2)
        else:
            all_meds = []
            for i in range(data.shape[2]):
                all_meds += [np.median(abs(data-data[:, :, i]), axis=2)]
            S = c*np.median(all_meds)
            
        res = (data - med)/S
        res = np.reshape(res, (-1, res.shape[2])).T
        pca = PCA(n_components=2)
        res = pca.fit_transform(np.reshape(all_features, (data.shape[2], -1)))
        mah_dist = ec.mahalanobis(res)
        ratio = sum((mah_dist <= np.percentile(mah_dist, percentile)))/res.shape[0]
        return ratio
