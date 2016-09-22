import numpy as np
import scipy.io.wavfile
import scipy.fftpack
import matplotlib.pyplot
import math
import binascii


#Sound generation parameters
signal_length = 30. #ms
Fs = 1/signal_length # Sampling Rate, in hertz -> samples/second, 44100 is standard
Ts = 1.0/Fs # sampling interval
Carrier_Frequency = 16. #This is hertz, A note.  The carrier frequency, I guess
Fc = 1.0/Carrier_Frequency
#ps = (2*math.pi)/Fs #rads/sample



#Square Wave Generation
ones = np.ones(signal_length)
neg_ones = -1.0*np.ones(signal_length)

test_data = np.append(ones, np.append(neg_ones, ones))

#At the end it will append this array to an array of "000000000001" or so, the wakeup signal.

#This part will have a text input and use the binascii library in order to convert a string to a binary array.

#matplotlib.pyplot.plot(np.arange(len(test_signal)), np.absolute(test_fft))
#matplotlib.pyplot.plot(np.arange(len(test_data)), test_data)
#matplotlib.pyplot.ylim([-2, 2])
#matplotlib.pyplot.show()


#Convert to Sound signal
domain = np.arange(len(test_data))
time = Ts*np.array(domain) #time array

amplitude = math.sqrt(2/Fs)
omegaX = (5.0)*np.cos(2.0*math.pi*Fc*time) #cos(2pi*Fs*n), 5 is a placeholder for amplitude
print omegaX
test_signal = np.multiply(omegaX, test_data) #multiplies by 1 or -1

test_signal_fft = scipy.fftpack.fft(test_signal)
matplotlib.pyplot.plot(time, test_signal)
matplotlib.pyplot.show()

#scipy.io.wavfile.write('test.wav', Fs, test_signal)