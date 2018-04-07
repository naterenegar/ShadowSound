import sounddevice as sd 
import numpy as np

#prompt user to select recorded device
print(sd.query_devices())
device = int(input("Pick a device to record:"))
sd.default.device = device;


#record some audio from the device
fs = 44100 #Hz
sd.default.samplerate = fs #Hz
sd.default.channels = 2
duration = 10 #seconds
recording = sd.rec(int(duration * fs))


#get a device to play back the recorded audio
device = int(input("Pick a device to play:"))
sd.default.device = device;

sd.play(recording, fs)
sd.wait()
