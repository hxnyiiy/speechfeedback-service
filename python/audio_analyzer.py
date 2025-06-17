import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

# --- 1. 오디오 파일 로드 ---
# 파일 경로를 지정합니다.
file_path = 'speech_ai_liam.mp3' 

try:
    # y: 오디오 신호 데이터 (waveform), sr: 샘플링 레이트 (Sampling Rate)
    # sr=None으로 설정하면 파일의 원본 샘플링 레이트를 사용합니다.
    y, sr = librosa.load(file_path, sr=None)
    print(f"오디오 파일을 성공적으로 로드했습니다.")
    print(f"샘플링 레이트 (sr): {sr}, 오디오 길이 (초): {len(y)/sr:.2f}s")

    # --- 2. 웨이브폼 시각화 ---
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr, alpha=0.8)
    plt.title('Waveform of ' + file_path)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('waveform.png') # 그래프를 이미지 파일로 저장
    plt.show()

except Exception as e:
    print(f"오류 발생: {e}")
    print("MP3 파일을 로드하려면 ffmpeg이 설치 및 설정되어 있어야 합니다.")

# --- 3. 스펙트로그램 분석 및 시각화 ---
# STFT(Short-Time Fourier Transform)를 사용하여 스펙트로그램 데이터 생성
D = librosa.stft(y)

# STFT 결과를 데시벨(dB) 스케일로 변환
S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)

plt.figure(figsize=(14, 5))
# 스펙트로그램 시각화
librosa.display.specshow(S_db, sr=sr, x_axis='time', y_axis='hz')
plt.colorbar(format='%+2.0f dB')
plt.title('Spectrogram of ' + file_path)
plt.xlabel('Time (s)')
plt.ylabel('Frequency (Hz)')
plt.tight_layout()
plt.savefig('spectrogram.png')
plt.show()

# --- 4. 유사도 분석을 위한 특징(Feature) 추출 (MFCC) ---
# n_mfcc: 추출할 MFCC 계수의 수
mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)

print(f"추출된 MFCC의 형태(shape): {mfccs.shape}")
# (추출된 계수 수, 시간 프레임 수) -> (13, 시간 길이) 형태의 2차원 배열

# MFCC 시각화
plt.figure(figsize=(14, 5))
librosa.display.specshow(mfccs, sr=sr, x_axis='time')
plt.colorbar(label='MFCC Coefficient')
plt.title('MFCC of ' + file_path)
plt.xlabel('Time (s)')
plt.ylabel('MFCC Coefficients')
plt.tight_layout()
plt.savefig('mfcc.png')
plt.show()

# --- 향후 유사도 비교 방법 (개념) ---
# 1. 비교하고 싶은 다른 MP3 파일 (예: 'other_audio.mp3')에 대해서도 위와 동일하게 MFCC를 추출합니다.
# other_mfccs = librosa.feature.mfcc(y=y2, sr=sr2, n_mfcc=13)

# 2. 두 MFCC 행렬(mfccs, other_mfccs)의 유사도를 계산합니다.
#    - 가장 간단한 방법은 각 MFCC 행렬의 평균을 내어 코사인 유사도(Cosine Similarity)를 계산하는 것입니다.
#    - 더 정교한 방법으로는 DTW(Dynamic Time Warping) 알고리즘을 사용하여 두 시계열 데이터의 거리를 측정할 수 있습니다.

# 예시: MFCC 평균 벡터 간의 코사인 유사도 (간단한 방법)
# from sklearn.metrics.pairwise import cosine_similarity
#
# mfccs_mean1 = np.mean(mfccs, axis=1)
# mfccs_mean2 = np.mean(other_mfccs, axis=1)
#
# similarity = cosine_similarity(mfccs_mean1.reshape(1, -1), mfccs_mean2.reshape(1, -1))
# print(f"두 오디오의 MFCC 기반 유사도: {similarity[0][0]}")