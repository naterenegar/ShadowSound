import wave
import numpy as np
import scipy.fftpack
import scipy.signal
import pyaudio
import matplotlib.pyplot as plt
import collections
import baudot

np.set_printoptions(threshold=np.nan)

f = wave.open('air_edit.wav')
l = f.getnframes()
frames = f.readframes(l)
dt = np.dtype(np.int16)
dt = dt.newbyteorder('<')
npbuf = np.frombuffer(frames, dt)

RATE = 44100
FRAME_SIZE = 1.0
TRACKING_FREQ = 800
DELTA = 50
F_THRESHOLD = 5
frame_width = int(FRAME_SIZE*RATE)
nframes = int(len(npbuf)*FRAME_SIZE/RATE);

print(np.arange(frame_width,nframes*frame_width,frame_width))

frames = np.split(npbuf, np.arange(frame_width,nframes*frame_width,frame_width))

out = []

for frame in frames:
    N = len(frame)
    li = int((TRACKING_FREQ-DELTA)*N/RATE)
    lu = int((TRACKING_FREQ+DELTA)*N/RATE)
    fft = scipy.fftpack.fft(frame[100:-100])
    fft = (2/N)*np.absolute(fft)
    #plt.plot(fft)
    #plt.show()
    f_val = np.mean(fft[li:lu])
    print(f_val)
    if f_val < F_THRESHOLD:
        out.insert(0,1)
    else:
        out.insert(0,0)

it = [iter(out)] * 5
baudot_list = list(zip(*it))[::-1]


string = "".join([baudot.baudot_to_str(x[::-1]) for x in baudot_list])

print(baudot_list)
print(string)
