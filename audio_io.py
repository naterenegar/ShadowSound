import pyaudio
import wave
import time
import sys

if len(sys.argv) < 2:
	print("Plays a wave file.\n\nUsage: $s filename.wav" % sys.argv[0])
	sys.exit(-1)

wf = wave.open(sys.argv[1], 'rb')

#instantiate PyAudio
p = pyaudio.PyAudio()

#define callback
def callback(in_data, frame_count, time_info, status):
	data = wf.readframes(frame_count)
	return (data, pyaudio.paContinue)

#open stream using callback function
stream = p.open(format=p.get_format_from_width(wf.getsampwidth()),
				channels=wf.getnchannels(),
				rate=wf.getframerate(),
				output=True,
				stream_callback=callback)

#start up that bad boi
stream.start_stream()

#wait for it to finish
while stream.is_active():
	time.sleep(0.1)

# stop stream
stream.stop_stream()
stream.close()
wf.close()

#close PyAudio
p.terminate()
