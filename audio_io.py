import pyaudio
import wave
import time
import sys
import numpy as np
import matplotlib.pyplot as plt

if len(sys.argv) < 2:
	print("Error: \n\nUsage: %s word" % sys.argv[0])
	sys.exit(-1)

#instantiate PyAudio
p = pyaudio.PyAudio()

#this function converts strings to binary
def text_to_bits(text, encoding='utf-8', errors='surrogatepass'):
    bits = bin(int.from_bytes(text.encode(encoding, errors), 'little'))[2:]
    return bits.zfill(8 * ((len(bits) + 7) // 8))

#present devices to users
print(str(p.get_device_count()) + ' devices found:')
for i in range(p.get_device_count()):
	print('\t' + str(i) + ': ' + p.get_device_info_by_index(i)['name'])

#ask for input device, output device, and message
input_device = int(input("Select an input device: "))
output_device = int(input("Select an output device: "))
message = text_to_bits(input("Enter a message to be encoded: "))
message_count = 0

#create dicts with device info
input_info = p.get_device_info_by_index(input_device)
output_info = p.get_device_info_by_index(output_device)

#pulling the 
framerate = int(input_info['defaultSampleRate'])
size = 44100

#frequencies for zero, followed by frequences for 1
zfreqs = [1600, 3500]
ofreqs = [4500, 2700]

def graph_fft(fft):
    T = 1.0 / framerate
    x = np.linspace(0.0, 1.0/(2.0 * T), size/2)
    y = fft
    fig,ax = plt.subplots()
    try:
        ax.plot(x, np.absolute(y)[:int(len(y)/2)])
    except:
        print('Bad dimensions!')
        print('X: ' + str(len(x)))
        print('Y: ' + str(len(y)))
        raise Exception
    plt.show()


def mirror(f, fft):
    n = len(fft)
    return int((n/2) + (n/2 - f))

def max_freq():
    return int(2.0 * framerate)

def freq_to_index(f, N, Fs):
    return int((f * N) / Fs)

def notch(fft, freq, delta):
    b = freq_to_index(max(freq-delta, 0), size, framerate)
    e = freq_to_index(min(freq+delta, max_freq()), size, framerate)
    bm = mirror(b, fft)
    em = mirror(e, fft)
    fft[b:e] = 0
    fft[em:bm] = 0
    #graph_fft(fft)
    return fft

def swiss_cheese(data, v):
    if v == 0:
        freqs = zfreqs
    else:
        freqs = ofreqs
    for f in freqs:
        data = notch(data, f, 300)
    return data

def encode(source, data):
	fft = np.fft.fft(source)
	fft = swiss_cheese(fft, data)
	fft = np.fft.ifft(fft)
	return fft.astype(np.int16)


#setting up dt for np.frombuffer
dt = np.dtype(int)
dt = dt.newbyteorder('<')

#callback takes an audio sample 
def callback(in_data, frame_count, time_info, status):
	data = np.fromstring(in_data, dtype=np.int16)

	#this encodes a 1 or 0 if message hasn't completed
	global message_count
	print('Encoding a ' + str(message[message_count]))
	data = encode(data, int(message[message_count]))
	if(message_count < len(message)-1):
		message_count += 1
	else:
		message_count = 0 
		print('Starting message over')


	return (data, pyaudio.paContinue)

#open an input/output stream with callback defined above and 
#data from selected devices
stream = p.open(format=pyaudio.paInt16,
				channels=int(input_info['maxInputChannels']),
				rate=framerate,
				input_device_index=input_device,
				output_device_index=output_device,
				input=True,
				output=True,
				frames_per_buffer=size,
				stream_callback=callback)

#start up that bad boi
stream.start_stream()

#wait for it to finish
while stream.is_active():
	time.sleep(0.1)

# stop stream
stream.stop_stream()
stream.close()

#close PyAudio
p.terminate()
