import numpy as np
import scipy.io.wavfile
import matplotlib
import math


#Sound generation parameters
Fs = 10000 # Sampling Rate, in hertz -> samples/second, 44100 is standard
ref_note = 440 #This is hertz, A note.
rps = (2*math.pi)/Fs #rads/sample


#Square Wave Generation
ones = np.ones(300)
neg_ones = -1*np.ones(300)

test_signal = np.append(ones, neg_ones, ones)
