#Math 260 Homework 9
#author: Max Wang

import matplotlib.pyplot as plt
import numpy as np
from numpy.fft import *
from scipy.io import wavfile


def tone_data():
    """ Builds the data for the phone number sounds...
        Returns:
            tones - list of the freqs. present in the phone number sounds
            nums - a dictionary mapping the num. k to its two freqs.
            pairs - a dictionary mapping the two freqs. to the nums
        Each number is represented by a pair of frequencies: a 'low' and 'high'
        For example, 4 is represented by 697 (low), 1336 (high),
        so nums[4] = (697, 1336)
        and pairs[(697, 1336)] = 4
    """
    lows = [697, 770, 852, 941]
    highs = [1209, 1336, 1477, 1633]  # (Hz)

    nums = {}
    for k in range(0, 3):
        nums[k+1] = (lows[k], highs[0])
        nums[k+4] = (lows[k], highs[1])
        nums[k+7] = (lows[k], highs[2])
    nums[0] = (lows[1], highs[3])

    pairs = {}
    for k, v in nums.items():
        pairs[(v[0], v[1])] = k

    tones = lows + highs  # combine to get total list of freqs.
    return tones, nums, pairs


def load_wav(fname):
    """ Loads a .wav file, returning the sound data.
        If stereo, converts to mono by averaging the two channels
        Returns:
            rate - the sample rate (in samples/sec)
            data - an np.array (1d) of the samples.
            length - the duration of the sound (sec)
    """
    rate, data = wavfile.read(fname)
    if len(data.shape) > 1 and data.shape[1] > 1:
        data = data[:, 0] + data[:, 1]  # stereo -> mono
    length = data.shape[0] / rate
    print(f"Loaded sound file {fname}.")
    return rate, data, length


def identify_digit(fname):
    rate, data, length = load_wav(fname)
    analyze_digit(rate, data, length)

# helper function for identify_digit that does not take file name as input
def analyze_digit(rate, data, length):
    # take fft of "data" and plotting
    transform = fft(data)
    n = data.shape[0]
    d = length/n
    freq = fftfreq(n, d) # frequencies
    k = freq*length # normalize for integer k
    mag = abs(transform) # magnitude |F_k|
    plt.figure(1)
    plt.xlabel('$k$')
    plt.ylabel('$|F_k|$')
    plt.title('Magnitude vs. $k$')
    plt.plot(k, mag, 'bo') #mag vs. freq
    plt.show()
    
    # analyze transform to find digit frequencies
    ranking = mag.argsort() # list of the indices used to sort the magnitudes
    '''Need the k values for the largest 4 magnitudes because each signal
    comprises 2 sines and each sine function contributes 2 frequencies
    (one + and one -). For an argsort array, the last 4 values are the
    indices of the largest 4 elements in the original array'''
    k_signal_index = np.array([ranking[n-1-i] for i in range(4)])
    k_signal = k[k_signal_index]
    freq_signal = k_signal/length
    freq_signal = freq_signal.round() # round to integers frequencies in Hz
    freq_signal.sort() # sort
    freq_digit = tuple(freq_signal[2:]) # first two negative, last two positive
    
    # compare with given digit frequencies
    tones, nums, pairs = tone_data()
    
    if freq_digit in pairs:
        digit = pairs[freq_digit]
        print(f'Found {digit}!')
        return str(digit)
    else:
        # raise ValueError('No digit found')
        print('No digit found!')
        return 'x'

def identify_dial(fname):
    # dial.wav: 555x42x
    # dial2.wav: 800x284
    # noisy_dial.wav: 555x42x
    
    tone_length = 0.7  # signal broken into 0.7 sec chunks with one num each
    rate, data, sound_length = load_wav(fname)
    
    digits = ''
    chunk_size = round(rate*0.7) # number of samples in a chunk
    chunk_length = chunk_size/rate # duration of chunk
                       
    # for each chunk, identify the digit
    for i in range(7):
        chunk = data[i*chunk_size: (i+1)*chunk_size]
        digits += analyze_digit(rate, chunk, chunk_length)

    return digits
    