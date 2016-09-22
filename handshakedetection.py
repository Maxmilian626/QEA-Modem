"""Listens for a handshake sound, then starts up the receiver.py function"""
import numpy as np
import scipy.io.wavfile
import scipy.fftpack
import matplotlib.pyplot
import math
import binascii

#Assuming that what is received is a filtered signal, i.e. what was sent out from part 1.

 #Parameters from sound generation used to demodulate the received signal.
signal_length = 30. #ms
Fs = 1/signal_length # Sampling Rate, in hertz -> samples/second, 44100 is standard
Ts = 1.0/Fs # sampling interval
Carrier_Frequency = 16. #This is hertz, A note.  The carrier frequency, I guess
Fc = 1.0/Carrier_Frequency
amplitude = math.sqrt(2/Fs)


#Time sensitive variables
signal = np.arange(100) #Placeholder signal array.

domain = np.arange(len(signal))
time = Ts*np.array(domain) #time array


#Here starts the demodulation

omegaX = (amplitude)*np.cos(2.0*math.pi*Fc*time)
counterMod = np.multiply(signal, omegaX)

#Here it's supposed to be integrated.  Assume this leaves us with an array of binary.
#The plan is to search the binary.  Ideally the "wakeup" signal would be a bunch of 0's, followed
#by a single "1".  This portion will be excised and the following should be the actual data.  







#Now muster even more good will and assume that this section will take the resultant binary and 
#convert it to a string.  

data = "done"
print data