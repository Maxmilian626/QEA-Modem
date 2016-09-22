import numpy as np
import scipy.io.wavfile
import scipy.fftpack
import matplotlib.pyplot
import math
import binascii
import pyaudio

#Sound generation parameters
bit_time = 500. #each bit should play for this many ms
Fs = 44100. # Sampling Rate, in hertz -> samples/second, 44100 is standard
bit_length = int(Fs*(bit_time/1000.)) #Samples required for each bit length
Ts = 1.0/Fs # sampling interval
Fc = 225. #This is hertz, A note.  The carrier frequency, I guess
rps = Fs*((2*math.pi)/Fc) #rads/sample

#Square Wave Generation
ones = np.ones(bit_length)
neg_ones = -1.0*np.ones(bit_length)

def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
	#http://stackoverflow.com/questions/7396849/convert-binary-to-ascii-and-vice-versa
    bits = bin(int(binascii.hexlify(text.encode(encoding, errors)), 16))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

word = "z"
bits = text_to_bits(word)

test_data = np.array([])

for char in bits:					#Assuming each entry plays at one entry/ms
	if char == "1":
		test_data = np.append(ones, test_data)
	else:
		test_data = np.append(neg_ones, test_data)


#test_fft = scipy.fftpack.fft(test_signal)
#Insert the wakeup signal, array of "01"
wakeupSignal = np.append(neg_ones, ones)
test_data = np.append(wakeupSignal, test_data)

#matplotlib.pyplot.plot(np.arange(len(test_signal)), np.absolute(test_fft))
#matplotlib.pyplot.plot(np.arange(len(test_data)), test_data)
#matplotlib.pyplot.ylim([-2, 2])
#matplotlib.pyplot.show()


#Convert to Sound signal
domain = np.arange(len(test_data))
time = np.array(domain)/(Fs)#time array, in seconds

amplitude = math.sqrt(2.0/(bit_length/1000.))
omegaX = amplitude*np.cos(2.0*math.pi*Fc*time) #cos(2pi*Fc*n), 
test_signal = np.multiply(omegaX, test_data) #multiplies by 1 or -1

test_signal_fft = scipy.fftpack.fft(test_signal)
matplotlib.pyplot.plot(time, test_signal)
matplotlib.pyplot.show()

def return_test_signal():
	return test_signal

# PyAudio = pyaudio.PyAudio
# p = PyAudio()
# stream = p.open(format = p.get_format_from_width(1), 
#                 channels = 1, 
#                 rate = 44100, 
#                 output = True)
# stream.write(test_signal)
# stream.stop_stream()
# stream.close()
# p.terminate()

#scipy.io.wavfile.write('test.wav', Fs, test_signal)