import wave
import numpy as np
import scipy.fftpack
import pyaudio
import matplotlib.pyplot as plt


class Decoder:
    """Singleton decoder class, storing current message"""


    
    def __init__(self,bit_size,freqs,framerate=44100,floor = 100):
        self.bit_size = bit_size
        self.chunk_size = int(bit_size/2)
        self.freqs = freqs
        self.n_chunks = 0
        self.f_vals = np.zeros((512,len(freqs)),dtype=int)
        self.framerate = framerate
        self.floor = 100
        
    def next_chunk(self, chunk,delta = 3):
        f_domain = scipy.fftpack.fft(chunk)
        index = self.freqs*self.chunk_size/self.framerate
        index = np.rint(index).astype(np.int32)
        vals = np.zeros(index.shape)

        #print(index)
        #plt.plot(np.absolute(f_domain[:int(len(f_domain)/2)]))
        #plt.show()
        for i in range(len(index)):
            ind =index[i]
            vals[i] = np.absolute(f_domain[ind])

        #print("min of dist: "+str(np.min(np.absolute(f_domain[ind]))))
        #print("min of vals: "+str(np.min(vals.astype(int))))
        self.f_vals[self.n_chunks,:] = vals.astype(int)
        #print("min of all vals: "+str(np.min(self.f_vals)))
        self.n_chunks += 1

    def process_chunks(self):
        cutoffs = np.tile(self.floor,[self.n_chunks,len(self.freqs)])
        
        greater = np.greater(np.sum(self.f_vals[:self.n_chunks,0:1],axis=1),np.sum(self.f_vals[:self.n_chunks,2:3],axis=1))
        return greater
        



decoder = Decoder(.5*44100,np.asarray([1600,3500,2700,4500]))

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
CHUNK = 1024
RECORD_SECONDS = 6

audio = pyaudio.PyAudio()

frames = []

'''for i in range(audio.get_device_count()):
    if((audio.get_device_info_by_index(i))['name'] == 'default'):
        pulseID = i

stream = audio.open(format=FORMAT, channels=CHANNELS,
                     rate=RATE, input=True,
                     frames_per_buffer=CHUNK,
                     input_device_index = pulseID)

numberOfFrames = int(RATE / CHUNK * RECORD_SECONDS)

for i in range(0, numberOfFrames):
    data = stream.read(CHUNK)
    frames.append(data)
                
                

    if(len(frames) >= numberOfFrames):
        break

print("finished recording mic")
#print(frames)

# stop Recording
stream.stop_stream()
stream.close()
'''
f = wave.open('out.wav')
l = f.getnframes()
frames = f.readframes(l)
dt = np.dtype(np.int16)
dt = dt.newbyteorder('<')
npbuf = np.frombuffer(frames, dt)
decoder_array = npbuf



data_array = decoder_array # np.fromstring(b''.join(frames), dtype=np.int16)

for i in range(int(len(data_array)/decoder.chunk_size)):
    decoder.next_chunk(data_array[(i*decoder.chunk_size):((i+1)*decoder.chunk_size)])

print(decoder.process_chunks())

waveFile = wave.open("test.wav", 'wb')
waveFile.setnchannels(1)
waveFile.setsampwidth(audio.get_sample_size(pyaudio.paInt16))
waveFile.setframerate(44100)
#waveFile.writeframes(b''.join(frames))
waveFile.close()
#print(self.frames2)
