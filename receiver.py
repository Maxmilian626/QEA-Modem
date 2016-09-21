from scipy import signal
import math
import matplotlib.pyplot as plt
import numpy as np
import pyaudio
import wave
import sys
from multiprocessing import Process, Queue
import time
import json
import requests
import wave
import evdev
import struct
import itertools

mrs2str = {'113': 'u', '111': 's', '13': 'a', '131': 'r', '11': 'i', '133': 'w', '31': 'n', 
'33': 'm', '1111': 'h', '1113': 'v', '1': 'e', '1311': 'l', '3': 't', '1131': 'f',
 '1331': 'p', '1333': 'j', '3131': 'c', '3133': 'y', '3311': 'z', '3313': 'q', '3113': 'x', '3111': 'b', 
 '333': 'o', '331': 'g', '311': 'd', '313': 'k'}


def lowPass(filterorder, sig):

	#function takes in an array that represents a signal and returns a version that has been through a low pass filter

	b, a = signal.butter(filterorder, .05) #create butterworth filter which exports parameters for filtfilt, given order and cutoff
	v = signal.filtfilt(b, a, sig)
	return v

def movingAverage (values, window):
	weights = np.repeat(1.0, window)/window
	sma = np.convolve(values, weights, 'valid')
	return sma

def bits(wave):
	amp_threshold = 7 #this is a placeholder
	dash_threshold = 1000 #distinction between dot and dash
	new_threshold = 3000 #threshold to be a new letter
	chunks = isplit(wave, (amp_threshold,))
	letterlist = []
	for chunk in chunks:
		if chunk[0] > amp_threshold: #dot or dash
			if len(chunk) > dash_threshold:
				letterlist.append("3")
			else:
				letterlist.append("1")
		else: #long pause or short pause
			if len(chunk) > new_threshold:
				letterlist.append("0")
			#else do nothing
	return ("".join(letterlist).split("0"))

def isplit(iterable, splitters):
	#a very nice split function, splits the iterable(list) on the splitter value
	return [list(g) for k,g in itertools.groupby(iterable,lambda x:x in splitters) if not k]

def letters(bits):
	pass


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
	audio = struct.unpack_from ("%dh" % nframes * nchannels, frames)

	lowpassed = lowPass(10, audio)
	smooth_magnitude = movingAverage(abs(lowpassed), 5) #takes the absolute value, then a moving average of that

	#print isplit([1, 4, None, 6, 9, None, 3, 9, 4 ], (None,))
	bits = bits(smooth_magnitude)

	word = ""
	for bit in bits:
		word += mrs2str[bit]

	print word