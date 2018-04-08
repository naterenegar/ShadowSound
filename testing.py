fn = 'test.wav'

f = wave.open(fn)
l = f.getnframes()
frames = f.readframes(l)
dt = np.dtype(np.int16)
dt = dt.newbyteorder('<')
npbuf = np.frombuffer(frames, dt)
decoder_array = npbuf



data_array = decoder_array # np.fromstring(b''.join(frames), dtype=np.int16)
x
