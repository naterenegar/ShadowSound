import matplotlib.pyplot as plt
import numpy.fft
#import numpy.ndarray
import numpy as np
import scipy.fftpack
import wave
import scipy.io.wavfile
import os
from os import listdir

source_fn = 'out.wav'

#source_f = wave.open(source_fn)

delta = 70

sample_size = 44100

zfreqs = [1600, 3500]
ofreqs = [4500, 2700]

dt = np.dtype(np.int16)
dt = dt.newbyteorder('<')

class Track:
    def __init__(self, source, seconds=None, start = 0, manual = False):
        self.source = source
        if manual:
            self.size = int(sample_size)
        elif seconds is None:
            self.size = source.getnframes() * seconds
        else:
            self.size = sample_size * seconds
        self.framerate = source.getframerate()
        self.store = []
        source.setpos(start * sample_size)
        if not manual:
            self.raw = source.readframes(self.size)
            self.updateData()
            self.fft = numpy.fft.fft(self.data)
        else:
            #Read and throw away the first chunk of data
            self.source.readframes(self.size)
            self.raw = None
            self.data = None
            self.fft = None

    def advance(self):
        self.raw = self.source.readframes(self.size)
        self.updateData()

    def move(self):
        self.source.setpos(self.source.tell() + self.size)

    def updateData(self):
        self.data = np.frombuffer(self.raw, dtype=dt)
        
    def updateFFT(self):
        self.fft = numpy.fft.fft(self.data)

    def write(self,fn,dat=None):
        if dat is None:
            scipy.io.wavfile.write(fn, 44100, self.data[:int(len(self.data)/2)])
        else:
            print('final: ' + str(len(dat)))
            t = np.asarray(dat, dtype=dt)
            print(t[:10])
            scipy.io.wavfile.write(fn, 44100, t)
            

    def writeArr(self):
        graph_fft(self)
        self.store = self.store + np.real(numpy.fft.ifft(self.fft)).astype(numpy.int16).tolist()

            

    def notch(self, freq, delta):
        b = freq_to_index(max(freq-delta, 0), self.size, self.framerate)
        e = freq_to_index(min(freq+delta, self.max_freq()), self.size, self.framerate)
        bm = mirror(b, self.fft)
        em = mirror(e, self.fft)
        self.ftt = self.fft[b:e] = 0
        self.ftt = self.fft[em:bm] = 0
        

    def invert(self):
        self.data = np.real(numpy.fft.ifft(self.fft)).astype(numpy.int16)

    def max_freq(self):
        return int(2.0 * self.framerate)
        


# Invert an fft & convert it to ints
def fft_to_int(fft):
    return fft[:int(len(inv)/2)].astype(numpy.int16)

def freq_to_index(f, N, Fs):
    return int((f * N) / Fs)

def mirror(f, fft):
    n = len(fft)
    return int((n/2) + (n/2 - f))
    
    
                                                                                                
def graph_data(track):
    x = np.linspace(0.0, track.size * 1.0/track.framerate, track.size)
    y = track.data
    fig,ax = plt.subplots()
    ax.plot(x,y)
    plt.show()

def graph_fft(track):
    T = 1.0 / track.framerate
    x = np.linspace(0.0, 1.0/(2.0 * T), track.size/2)
    y = track.fft
    fig,ax = plt.subplots()
    try:
        ax.plot(x, numpy.absolute(y)[:int(len(y)/2)])
    except:
        print('Bad dimensions!')
        print('X: ' + str(len(x)))
        print('Y: ' + str(len(y)))
        raise Exception
    plt.show()

def plot_inv(track):
    inv = fft_to_int(track.fft)
    x = np.linspace(0.0, float(len(inv))/track.framerate, len(inv))
    fig,ax = plt.subplots()
    ax.plot(x,inv)
    plt.show()
    return inv
 
def swiss_cheese(track, v):
    if v == 0:
        freqs = zfreqs
    else:
        freqs = ofreqs
    for f in freqs:
        track.notch(f, 300)
    return track

def proc_file(fn):
    print("Processing: " + fn)
    f = wave.open(fn)
    t = Track(f)
    t = swiss_cheese(t)
    t.invert()
    t.write(fn[:-4] + '_edit.wav')
    f.close()
    

def encode(source, data):
    input = wave.open(source)
    track = Track(input, 0.5, manual=True)
    out = open('out.wav', 'wb')
    for i in data:
        track.advance()
        track.updateFFT()
        swiss_cheese(track, i)
#            graph_fft(track)
        track.writeArr()
    track.write(out,track.store)
    out.close()
    input.close()
            
    
encode('bluesky.wav', [1,0,1,0,1,0,1,0,1,0,1])


#source_f = wave.open(source_fn)

#track.write('out.wav')

