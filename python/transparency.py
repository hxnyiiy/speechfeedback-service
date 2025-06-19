import matplotlib.pyplot as plt
import librosa
import librosa.display
import numpy as np

# --- 1. 오디오 파일 경로 설정 ---
# 실제 파일 경로에 맞게 변경해주세요.
standard_audio_path = '/work/speechfeedback-service/python/mp3/speech_ai_liam.mp3'
uploaded_audio_path = '/work/speechfeedback-service/python/mp3/speech_ai_satoori.mp3'

# --- 2. 오디오 파일 로드 ---
try:
    # 첫 번째 오디오 파일 로드
    y_standard, sr_standard = librosa.load(standard_audio_path, sr=None)
    # 두 번째 오디오 파일 로드
    y_uploaded, sr_uploaded = librosa.load(uploaded_audio_path, sr=None)
except FileNotFoundError as e:
    print(f"Error: {e}. Make sure the audio files are in the correct directory.")
    exit()

# --- 3. 데이터 전처리 및 정규화 (선택 사항이지만 비교를 위해 권장) ---
# 두 오디오의 샘플링 레이트(sr)가 다를 경우 resampling이 필요할 수 있습니다.
# 여기서는 sr이 같다고 가정하거나, 각자 자신의 sr을 유지한 채 시간축을 맞춰 그립니다.

# 진폭 정규화: -1.0에서 1.0 사이로 스케일링
# 이미 librosa.load는 기본적으로 -1.0 ~ 1.0으로 정규화하므로, 추가 정규화는 필요 없을 수도 있습니다.
# 만약 필요하다면, 예를 들어 peak normalization을 사용할 수 있습니다.
# y_standard = librosa.util.normalize(y_standard)
# y_uploaded = librosa.util.normalize(y_uploaded)

# 시간 축 생성 (초 단위)
time_standard = np.linspace(0, len(y_standard) / sr_standard, num=len(y_standard))
time_uploaded = np.linspace(0, len(y_uploaded) / sr_uploaded, num=len(y_uploaded))

# --- 4. 파형 플로팅 ---
plt.figure(figsize=(14, 6)) # 전체 그래프 크기 설정

# 첫 번째 파형 플로팅 (투명도 0.7)
plt.plot(time_standard, y_standard, label='Standard Audio (speech_ai_liam.mp3)', alpha=0.7, color='red')

# 두 번째 파형 플로팅 (투명도 0.7)
# 'zorder'를 사용하면 플로팅 순서와 관계없이 레이어를 조절할 수 있습니다.
# 여기서는 단순히 겹쳐 그리는 것이므로 큰 의미는 없지만, 필요시 활용 가능합니다.
plt.plot(time_uploaded, y_uploaded, label='Uploaded Audio (speech_ai_satoori.mp3)', alpha=0.7, color='blue')

# --- 5. 그래프 꾸미기 ---
plt.title('Waveform Comparison with Transparency')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.grid(True) # 격자 표시
plt.legend(loc='upper right') # 범례 표시

# x축 범위 조정 (두 파형의 길이가 다를 경우, 더 긴 파형에 맞춰 자동으로 설정될 것입니다)
# 필요하다면 특정 시간 범위로 수동으로 설정할 수 있습니다.
# max_time = max(len(y_standard) / sr_standard, len(y_uploaded) / sr_uploaded)
# plt.xlim(0, max_time)

# y축 범위 조정 (이미지처럼 -1.0 ~ 1.0 또는 비슷한 범위로)
plt.ylim(-1.0, 1.0) # 이미지에 맞춰 y축 범위 설정

plt.tight_layout() # 그래프 요소들이 겹치지 않도록 자동 조정

# plt.show() # 그래프를 화면에 보여주는 대신 파일로 저장합니다.
plt.savefig('waveform_comparison.png') # 그래프를 PNG 파일로 저장합니다.
print("그래프가 'waveform_comparison.png' 파일로 저장되었습니다.")