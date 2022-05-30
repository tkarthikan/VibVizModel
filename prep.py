import librosa, librosa.display
import matplotlib.pyplot as plt
import numpy as np
from torch import sign

file = "/Users/karthik/Documents/GenHap/VibvizModel/VibVizModel/audio_wav/viblib/v-09-09-8-11.wav"

#waveform
signal, sr = librosa.load(file, sr=22050)  #sr * T = the lenght of the signal array
print(len(signal))
# librosa.display.waveshow(signal, sr=sr)
# plt.xlabel("Time")
# plt.ylabel("Amplitude")
# plt.show()

#fft->spectrum
fft = np.fft.fft(signal) #We expect a 1D numpy array with as many as values as signal array
print(len(fft))

magnitude = np.abs(fft) #getting the magnitude which indicates the contribution of each frequency to overall sound
frequency = np.linspace(0, sr, len(magnitude)) #Gives the number

left_frequency = frequency[:int(len(frequency)/2)]
left_magnitude = magnitude[:int(len(frequency)/2)]

# plt.plot(left_frequency, left_magnitude)
# plt.xlabel("Frequency")
# plt.ylabel("Magnitude")
# plt.show()

n_fft = 2048
hop_length = 512

stft = librosa.core.stft(signal, hop_length=hop_length, n_fft=n_fft)

spectrogram = np.abs(stft)
log_spectrogram = librosa.amplitude_to_db(spectrogram)
# librosa.display.specshow(log_spectrogram, sr=sr, hop_length=hop_length)
# plt.xlabel("Time")
# plt.ylabel("Frequency")
# plt.colorbar()
# plt.show()

#MFCC
MFCCs = librosa.feature.mfcc(signal,n_fft=n_fft, hop_length=hop_length, n_mfcc=13)
librosa.display.specshow(MFCCs, sr=sr, hop_length=hop_length)
plt.xlabel("Time")
plt.ylabel("MFCC")
plt.colorbar()
plt.show()