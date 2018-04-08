import matplotlib.pyplot as plt
import numpy.fft
import numpy as np
import scipy.fftpack
import wave
import scipy.io.wavfile
import baudot

delta = 100
sample = 44100
zfreqs = []
ofreqs = [800]

dt = np.dtype(np.int16).newbyteorder('<')

source_fn = input('File: ')
message = input('Message: ')

bchars = [baudot.str_to_baudot(x) for x in message]

toencode = []
for b in bchars:
    for v in b:
        toencode.append(v)

print(toencode)


source = wave.open(source_fn)
frames = source.getnframes()
seconds = frames / sample
print(seconds)

reqseconds = len(toencode) * sample

indata = source.readframes(source.getnframes())
outdata = []
def notch(fft, freq):
    b = freq_to_index(max(freq-delta, 0), int(sample/2), source.getframerate())
    e = freq_to_index(min(freq+delta, 20000), int(sample/2), source.getframerate())
    bm = mirror(b, fft)
    em = mirror(e, fft)
    print(len(list(filter(lambda f: f == -1, fft))))
    fft[b:e] = 0
    fft[em:bm] = 0
    print(len(list(filter(lambda f: f == 0, fft))))
    return fft

def freq_to_index(f, N, Fs):
    return int((f * N) / Fs)

def mirror(f, fft):
    n = len(fft)
    return int((n/2) + (n/2 - f))
    
def parse(raw, v):
    print(len(raw))
    t = np.frombuffer(raw, dtype=dt)
    if v:
        fft = numpy.fft.fft(t)
        fft = notch(fft, 800)
        print(len(list(filter(lambda x: x==0, fft))))
        t = np.real(numpy.fft.ifft(fft))
    return t.tolist()
    

    


for v in toencode:
    print(len(indata))
    toParse = indata[:sample]
    indata = indata[sample:]
    outdata = outdata + parse(toParse, v)

final = np.asarray(outdata, dt)
print(final.shape)
scipy.io.wavfile.write(source_fn[:-4] + '_edit.wav', sample, final)
print('Done!')




