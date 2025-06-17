import pandas as pd
import numpy as np
import matplotlib.pyplab as plt
import seaborn as sns 

from glob import glob

import librosa.display 
import IPython.display as ipd

audio_files =glob('../audio/*/*.mp3') #파일 경로

ipd.audio(audio_files[0]) # play audio file 

y, sr = librosa.load(audio_files[0])

print(f'y: {y[:10]}')
print(f'shape y: {y.shape}')
print(f'sr: {sr}')

pd.Series(y).plot(figsize=(10, 5), lw=1, titile = 'Raw Audio Example', color='gray')
plt.show()

y_trimmed, _ = librosa.effects.trim(y, top_db=20)
pd.Series(y_trimmed).plot(figsize=(10, 5), lw=1, titile = 'Raw Audio Trimmed Example', color='blue')
plt.show()

pd.Series(y[30000:3050000]).plot(figsize=(10, 5), lw=1, titile = 'Raw Audio Zoomed In Example', color='blue')
plt.show()

D = librosa.stft(y)
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
S_db.shape

fig, ax = plt.subplots(figsize=(10, 5))
img = librosa.display.specshow(S_db, x_axis='time', y_axis='log', ax=ax) 
ax.set_title('Spectrogram Example', fontsize=20)
plt.show() 

S = librosa.feature.melspectrogram(y=y, sr=sr,n_mels=128,) 
S_db_mel = librosa.amplitude_to_db(S, ref=np.max)

fig, ax = plt.subplots(figsize=(15, 5))
img = librosa.display.specshow(S_db_mel, x_axis='time', y_axis='mel', ax=ax) 
ax.set_title('Mel-Spectrogram Example', fontsize=20)
plt.show()


