import wave
import numpy as np
import scipy.fftpack
import scipy.signal
import pyaudio
import matplotlib.pyplot as plt
import collections

np.set_printoptions(threshold=np.nan)
'''
class Decoder:
    """Singleton decoder class, storing current message"""


    
    def __init__(self,bit_size,freqs,framerate=44100,floor = 200000):
        self.bit_size = bit_size
        self.chunk_size = bit_size/2
        self.freqs = freqs
        self.n_chunks = 0
        self.f_vals = np.zeros((512,len(freqs)),dtype=int)
        self.framerate = framerate
        self.floor = floor
        
    def next_chunk(self, chunk,delta = 3):
        f_domain = scipy.fftpack.fft(chunk)
        index = self.freqs*self.chunk_size/self.framerate
        index = np.rint(index).astype(np.int32)
        vals = np.zeros(index.shape)
        #print(index)
        plt.plot(np.absolute(f_domain[:int(len(f_domain)/2)]))
        #plt.plot(chunk)
        plt.show(block=False)
        for i in range(len(index)):
            ind =index[i]
            vals[i] = np.absolute(f_domain[ind])

        #print("min of dist: "+str(np.min(np.absolute(f_domain[ind]))))
        #print("min of vals: "+str(np.min(vals.astype(int))))
        self.f_vals[self.n_chunks,:] = vals.astype(int)
        #print("min of all vals: "+str(np.min(self.f_vals)))
        self.n_chunks += 1
        #t,f,Sxx = scipy.signal.spectrogram(chunk,self.framerate)
        #plt.subplot(2,1,1)
        #plt.pcolormesh(t, f, Sxx.T)
        #fw,Pxx = scipy.signal.welch(chunk,self.framerate)
        #plt.subplot(2,1,2)
        #plt.plot(fw,Pxx)
        #plt.show()

    def process_chunks(self):
        cutoffs = np.tile(self.floor,[self.n_chunks,len(self.freqs)])

        #print(self.f_vals[:self.n_chunks,:])
        #print(cutoffs)
        greater = np.less(self.f_vals[:self.n_chunks,:],cutoffs)
        return greater

    def clear(self):
        self.n_chunks = 0
        self.f_vals = np.zeros((512,len(self.freqs)),dtype=int)
'''

f = wave.open('test4.wav')
l = f.getnframes()
frames = f.readframes(l)
dt = np.dtype(np.int16)
dt = dt.newbyteorder('<')
npbuf = np.frombuffer(frames, dt)

RATE = 44100
FRAME_SIZE = 1.0
TRACKING_FREQ = 800
DELTA = 50
F_THRESHOLD = 10
frame_width = int(FRAME_SIZE*RATE)
nframes = int(len(npbuf)*FRAME_SIZE/RATE);


frames = np.split(npbuf, np.arange(frame_width,nframes*frame_width,frame_width))

out = []

for frame in frames:
    N = len(frame)
    li = int((TRACKING_FREQ-DELTA)*N/RATE)
    lu = int((TRACKING_FREQ+DELTA)*N/RATE)
    fft = scipy.fftpack.fft(frame)
    fft = (2/N)*np.absolute(fft)
    f_val = np.median(fft[li:lu])
    if f_val < F_THRESHOLD:
        out.insert(0,1)
    else:
        out.insert(0,0)

it = [iter(out)] * 5
print(list(zip(*it)))





        

'''
decoder = Decoder(.5*44100,np.asarray([500]))


FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 10
numberOfFrames = int((RATE / CHUNK) * RECORD_SECONDS)
tracking_freq = 800
freq_delta = 50

audio = pyaudio.PyAudio()


for i in range(audio.get_device_count()):
    if((audio.get_device_info_by_index(i))['name'] == 'default'):
        pulseID = i

stream = audio.open(format=FORMAT, channels=CHANNELS,
                     rate=RATE, input=True,
                     frames_per_buffer=CHUNK,
                     input_device_index = pulseID)



frames = []
for i in range(0, numberOfFrames):
    data = stream.read(CHUNK)
    frames.append(data)
    if(len(frames) >= numberOfFrames):
        break
data_array = np.fromstring(b''.join(frames), dtype=np.int16)

samples = np.array_split(data_array,10);

V_f_data = collections.deque(10*[0], 10);

A_t = []
A_ct = []
for el in samples:
    N = len(el)
    li = int((tracking_freq-freq_delta)*N/RATE)
    lu = int((tracking_freq+freq_delta)*N/RATE)
    fft = scipy.fftpack.fft(el)
    fft = (2.0/N)*np.absolute(fft)[:int(N/2)]
    fft = fft/np.sum(fft)
    
    A_t.append(np.mean(np.absolute(el)))
    A_ct.append(np.mean(fft[li:lu]))
    V_f_data.appendleft(np.mean(fft[li:lu]))

A_t = np.mean(A_t)-.25*(np.std(A_t))
A_ct = np.mean(A_ct)+.5*(np.std(A_ct))


print(np.std(V_f_data))

print('Setup Complete')

RECORD_SECONDS = 1
numberOfFrames = int((RATE / CHUNK) * RECORD_SECONDS)

readings = collections.deque(10*[0], 10);




while True:
    frames = []
    for i in range(0, numberOfFrames):
        data = stream.read(CHUNK)
        frames.append(data)
        if(len(frames) >= numberOfFrames):
            break

    data_array = np.fromstring(b''.join(frames), dtype=np.int16)
    N = len(data_array)
    fft = scipy.fftpack.fft(data_array)
    li = int((tracking_freq-freq_delta)*N/RATE)
    lu = int((tracking_freq+freq_delta)*N/RATE)
    fft = (2.0/N)*np.absolute(fft)[:int(N/2)]
    fft = fft/np.sum(fft)
    if np.mean(np.absolute(data_array)) < A_t:
        readings.appendleft(0)
        #print(readings)

    elif np.mean(fft[li:lu]) > A_ct:
        readings.appendleft(0)
        #print(readings)

    else:
        readings.appendleft(1)
        #print(readings)

    V_f_data.appendleft(np.mean(fft[li:lu]))

    
    print("Amplitude: " + str(np.mean(np.absolute(data_array))) + " < Threshold: " + str(A_t))
    print("Frequency Amplitude: " + str(np.mean(fft[li:lu])) + " > Threshold: " + str(A_ct))
    print("Tracking amplitude stdvar: "+ str(np.std(V_f_data)))
'''

'''
RECORD_SECONDS = 5
numberOfFrames = int((RATE / CHUNK) * RECORD_SECONDS)

while True:    
    for i in range(0, numberOfFrames):
        data = stream.read(CHUNK)
        frames.append(data)
        if(len(frames) >= numberOfFrames):
            break


    print("finished recording frame")
    data_array = np.fromstring(b''.join(frames), dtype=np.int16)
    for i in range(int(len(data_array)/decoder.chunk_size)):
        decoder.next_chunk(data_array[int((i*decoder.chunk_size)):int(((i+1)*decoder.chunk_size))])

    print(decoder.process_chunks())
    decoder.clear()
'''

#print(frames)

# stop Recording
#stream.stop_stream()
#stream.close()

'''
f = wave.open('test4.wav')
l = f.getnframes()
frames = f.readframes(l)
dt = np.dtype(np.int16)
dt = dt.newbyteorder('<')
npbuf = np.frombuffer(frames, dt)
decoder_array = npbuf
'''




'''
waveFile = wave.open("test3.wav", 'wb')
waveFile.setnchannels(1)
waveFile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
waveFile.setframerate(44100)
#waveFile.writeframes(b''.join(frames))
waveFile.close()
#print(self.frames2)
'''
