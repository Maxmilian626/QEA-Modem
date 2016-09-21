import numpy as np
import scipy.io.wavfile
import scipy.fftpack
import matplotlib.pyplot
import math


#Sound generation parameters
Fs = 300 # Sampling Rate, in hertz -> samples/second, 44100 is standard
Ts = 1.0/Fs # sampling interval
ref_note = 440 #This is hertz, A note.  The carrier wave, I guess
rps = (2*math.pi)/Fs #rads/sample


#Square Wave Generation
ones = np.ones(300)
neg_ones = -1*np.ones(300)

test_data = np.append(ones, np.append(neg_ones, ones))

#test_fft = scipy.fftpack.fft(test_signal)

#matplotlib.pyplot.plot(np.arange(len(test_signal)), np.absolute(test_fft))
#matplotlib.pyplot.plot(np.arange(len(test_data)), test_data)
#matplotlib.pyplot.ylim([-2, 2])
#matplotlib.pyplot.show()


#Convert to Sound signal
domain = np.arange(len(test_data))
time = Ts*np.array(domain)
omegaX = rps * np.array(domain)
print omegaX
test_signal = math.cos(omegaX)*test_data

test_signal_fft = scipy.fftpack.fft(test_signal)
matplotlib.pyplot.plot(time, np.absolute(test_signal_fft))
matplotlib.pyplot.show()

#scipy.io.wavfile.write('test.wav', Fs, test_signal)