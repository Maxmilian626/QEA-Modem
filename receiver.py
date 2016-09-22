from scipy import signal
import math
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import wave
import struct
import itertools
import scipy.fftpack
import binascii


#getting a clean wave

def lowPass(filterorder, sig):
	#function takes in an array that represents a signal and returns a version that has been through a low pass filter
	b, a = signal.butter(filterorder, .05) #create butterworth filter which exports parameters for filtfilt, given order and cutoff
	v = signal.filtfilt(b, a, sig)
	return v

def movingAverage (values, window):
	weights = np.repeat(1.0, window)/window
	sma = np.convolve(values, weights, 'valid')
	return sma

def downconversion(signal):
	Fs = 44100 # Sampling Rate, in hertz -> samples/second, 44100 is standard
	Ts = 1.0/Fs # sampling interval
	Carrier_Frequency = 450. #This is hertz, A note.  The carrier frequency, I guess
	Fc = 1.0/Carrier_Frequency

	domain = np.arange(len(signal))
	time = Ts*np.array(domain) #time array

	sig = (5.0)*np.cos(2.0*math.pi*Fc*time) # 5 is placeholder for amplitude...

	return np.multiply(sig, signal)



#translater functions one you have a clean wave

def bits(wave):
	amp_threshold = 0.5 #this is a placeholder. it's totally too high.
	chunks = isplit(wave, (amp_threshold,))
	letterlist = []
	for chunk in chunks:
		if chunk[0] > amp_threshold: #1
			letterlist.append("1")
		else: #0
			letterlist.append("0")
	return letterlist

def isplit(iterable, splitters):
	#a very nice split function, splits the iterable(list) on the splitter value
	return [list(g) for k,g in itertools.groupby(iterable,lambda x:x in splitters) if not k]

def text_from_bits(bits, encoding='utf-8', errors='surrogatepass'):
    n = int(bits, 2)
    return int2bytes(n).decode(encoding, errors)

def int2bytes(i):
    hex_string = '%x' % i
    n = len(hex_string)
    return binascii.unhexlify(hex_string.zfill(n + (n & 1)))



#recording the signal

def record():
	""" Records audio while the space bar is pressed and saves the audio to
	recording.wav. 
	From https://gist.github.com/mabdrabo/8678538"""
	FORMAT = pyaudio.paInt16
	CHANNELS = 2
	RATE = 44100
	CHUNK = 1024
	RECORD_SECONDS = 5
	WAVE_OUTPUT_FILENAME = "file.wav"
	 
	audio = pyaudio.PyAudio()
	 
	# start Recording
	stream = audio.open(format=FORMAT, channels=CHANNELS,
					rate=RATE, input=True,
					frames_per_buffer=CHUNK)
	print "recording..."
	frames = []
	 
	for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
		data = stream.read(CHUNK)
		frames.append(data)
	print "finished recording"
	 
	 
	# stop Recording
	stream.stop_stream()
	stream.close()
	audio.terminate()
	 
	waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
	waveFile.setnchannels(CHANNELS)
	waveFile.setsampwidth(audio.get_sample_size(FORMAT))
	waveFile.setframerate(RATE)
	waveFile.writeframes(b''.join(frames))
	waveFile.close()

if __name__ == '__main__':

	record();
	w = wave.open("file.wav", "r")
	(nchannels, sampwidth, framerate, nframes, comptype, compname) = w.getparams ()
	frames = w.readframes(nframes*nchannels)
	aud = struct.unpack_from ("%dh" % nframes * nchannels, frames)

	#lowpassed = lowPass(10, audio)

	back = downconversion(aud)
	back = movingAverage(abs(back), 1000)

	fig, ay = plt.subplots()
	ay.plot(np.array(back), 'b')
	ay.plot(np.array(aud), 'm')
	plt.show()

	# bits = bits(smooth_magnitude)

	# word = ""
	# for bit in bits:
	# 	word += mrs2str[bit]

	# print word

	Aud = np.array(scipy.fftpack.fft(aud))
	Low = np.array(scipy.fftpack.fft(back))

	T = 1.0/44100
	N = len(Aud)
	xf = np.linspace(0.0, 1.0/(2.0*T), N/2)

	fig, ax = plt.subplots()
	ax.plot(xf, 2.0/N * np.abs(Aud[0:N/2]), 'b-')
	ax.plot(xf, 2.0/N * np.abs(Low[0:N/2]), 'm-')
	plt.show()