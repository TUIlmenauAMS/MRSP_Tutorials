"""
Using Pyaudio, record sound from the audio device and plot a waterfall spectrum display, for 8 seconds.
Usage example: python pyrecspecwaterfall.py
Gerald Schuller, November 2014 
"""

import pyaudio
import struct
#import math
#import array
import numpy as np
#import sys
#import wave
#import matplotlib.pyplot as plt
#import matplotlib.animation as animation
#import pylab
import cv2

CHUNK = 2048 #Blocksize
WIDTH = 2 #2 bytes per sample
CHANNELS = 1 #2
RATE = 48000  #Sampling Rate in Hz


p = pyaudio.PyAudio()

a = p.get_device_count()
print("device count=",a)

for i in range(0, a):
    print("i = ",i)
    b = p.get_device_info_by_index(i)['maxInputChannels']
    print(b)
    b = p.get_device_info_by_index(i)['defaultSampleRate']
    print(b)

stream = p.open(format=p.get_format_from_width(WIDTH),
                channels=CHANNELS,
                rate=RATE,
                input=True,
                output=True,
                #input_device_index=3,
                frames_per_buffer=CHUNK)


print("* recording")
#Size of waterfall diagramm:
#max CHUNK/2 rows:
rows=500
cols=512
fftlen=cols*2
frame=0.0*np.ones((rows,cols,3));

ctr=0
while(True):
    ctr=ctr+1  
    #Reading from audio input stream into data with block length "CHUNK":
    data = stream.read(CHUNK)
    #Convert from stream of bytes to a list of short integers (2 bytes here) in "samples":
    #shorts = (struct.unpack( "128h", data ))
    shorts = (struct.unpack( 'h' * CHUNK, data ));
    samples=np.array(list(shorts),dtype=float);

    if (ctr%1 ==0):  #Downsampling for dislayed blocks, make it ctr%4 for slow computers!
       #shift "frame" 1 up:
       frame[0:(rows-1),:]=frame[1:rows,:]; 
       #compute magnitude of 1D FFT of sound 
       #with suitable normalization for the display:
       #frame=np.abs(np.ffqt.fft2(frame[:,:,1]/255.0))/512.0
       #write magnitude spectrum in lowes row of "frame":
       R=0.25*np.log((np.abs(np.fft.fft(samples[0:fftlen])[0:int(fftlen/2)]/np.sqrt(fftlen))+1))/np.log(10.0)
       #Color mapping:
       #Red:
       frame[rows-1,:,2]=R
       #Green:
       frame[rows-1,:,1]=np.abs(1-2*R)
       #Blue:
       frame[rows-1,:,0]=1.0-R
       #frame[rows-1,:,0]=frame[rows-1,:,1]**3
       # Display the resulting frame
       cv2.imshow('frame',frame)
    #Keep window open until key 'q' is pressed:
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
# When everything done, release the capture

cv2.destroyAllWindows()

stream.stop_stream()
stream.close()
p.terminate()

