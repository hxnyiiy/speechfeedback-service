import librosa
import librosa.display
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastdtw import fastdtw # fastdtw 라이브러리 추가
from scipy.spatial.distance import euclidean # DTW에 사용될 거리 함수
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import io
import os
import uvicorn
import matplotlib.pyplot as plt

app = FastAPI()

# --- 0. 표준 오디오 파일 로드 (미리 로드하여 효율성 높임) ---
# 이 경로는 Docker 컨테이너 내의 경로를 고려해야 합니다.
# Dockerfile에서 /work/speechfeedback-service/python/mp3/ 경로로 파일을 복사했다면 이대로 유지합니다.
# 그렇지 않다면 Dockerfile에 맞게 경로를 조정해야 합니다.
STANDARD_AUDIO_PATH = "/work/speechfeedback-service/python/mp3/speech_ai_liam.mp3"
STANDARD_N_MFCC = 13

# 표준 오디오 데이터를 전역 변수로 저장
y_standard_audio_data = None
sr_standard_audio_data = None
mfccs_standard_full = None # 전체 MFCC 시퀀스를 저장하도록 변경

try:
    y_standard_audio_data, sr_standard_audio_data = librosa.load(STANDARD_AUDIO_PATH, sr=None)
    # 전체 MFCC 시퀀스를 계산하여 저장합니다.
    mfccs_standard_full = librosa.feature.mfcc(y=y_standard_audio_data, sr=sr_standard_audio_data, n_mfcc=STANDARD_N_MFCC)
    print(f"표준 오디오 파일 '{STANDARD_AUDIO_PATH}' 로드 및 MFCC 계산 완료.")
except Exception as e:
    print(f"FATAL ERROR: 표준 오디오 파일 로드 또는 MFCC 계산 실패: {e}")
    print("FastAPI 서버가 시작되지 않습니다. 표준 오디오 파일 경로와 파일 존재 여부를 확인하세요.")
    exit(1)

class SimilarityResult(BaseModel):
    similarity_score: float
    message: str
    detail: str = None

# --- 기존 엔드포인트 1: 파일 직접 받아서 유사도 분석 ---
@app.post("/analyze_audio_similarity", response_model=SimilarityResult)
async def analyze_audio_similarity(file: UploadFile = File(...)):
    """
    업로드된 오디오 파일과 표준 AI 오디오 파일 간의 MFCC 기반 유사도를 분석합니다.
    DTW (Dynamic Time Warping)를 사용하여 더욱 상세한 유사도 비교를 수행합니다.
    """
    try:
        audio_content = await file.read() # 업로드된 파일 내용을 읽음
        audio_io = io.BytesIO(audio_content) # 메모리 버퍼로 변환

        y_uploaded, sr_uploaded = librosa.load(audio_io, sr=None)
        
        # 업로드된 오디오의 전체 MFCC 시퀀스를 계산합니다.
        mfccs_uploaded_full = librosa.feature.mfcc(y=y_uploaded, sr=sr_uploaded, n_mfcc=STANDARD_N_MFCC)
        
        # DTW를 사용하여 두 MFCC 시퀀스 간의 거리 계산
        # fastdtw는 (거리, 경로) 튜플을 반환합니다.
        distance, path = fastdtw(mfccs_standard_full.T, mfccs_uploaded_full.T, dist=euclidean)

        # DTW 거리를 유사도 점수로 변환 (0과 1 사이로 정규화)
        # 거리가 0에 가까울수록 유사도가 높음
        # 일반적으로 1 / (1 + distance) 와 같은 형태나,
        # max_distance로 정규화하는 방법을 사용합니다.
        # 여기서는 경로의 길이를 활용하여 정규화합니다.
        normalized_distance = distance / len(path) # 경로 길이로 나누어 정규화
        
        # 유사도 점수 계산 (0: 유사성 없음, 1: 완벽하게 유사)
        # 값이 너무 커지는 것을 방지하고 유사도로 변환하기 위한 경험적인 방법
        similarity_score = np.exp(-normalized_distance / 100) # 지수 함수를 사용하여 유사도를 0-1 사이로 스케일링

        # 유사도 점수에 따른 메시지
        if similarity_score > 0.9:
            message = "두 오디오는 매우 유사합니다."
        elif similarity_score > 0.7:
            message = "두 오디오는 유사합니다."
        elif similarity_score > 0.5:
            message = "두 오디오는 어느 정도 유사성을 보이지만, 차이가 있을 수 있습니다."
        else:
            message = "두 오디오는 유사성이 낮습니다."
        
        uploaded_filename = file.filename 
        
        return SimilarityResult(
            similarity_score=float(similarity_score),
            message=message,
            detail=f"업로드된 파일: {uploaded_filename}, 표준 파일: {os.path.basename(STANDARD_AUDIO_PATH)}, DTW 정규화 거리: {normalized_distance:.2f}"
        )
    except Exception as e:
        if "ffmpeg" in str(e).lower() or "codec" in str(e).lower():
            raise HTTPException(status_code=500, detail=f"오디오 처리 중 오류 발생: {str(e)}. MP3 파일을 로드하려면 ffmpeg이 설치 및 설정되어 있어야 합니다.")
        else:
            raise HTTPException(status_code=500, detail=f"오디오 분석 중 오류 발생: {str(e)}")


# --- 기존 엔드포인트 2: 파일 직접 받아서 파형 이미지 생성 ---
@app.post("/generate_waveform")
async def generate_waveform(file: UploadFile = File(...)):
    """
    업로드된 오디오 파일의 파형(waveform) 이미지를 생성하여 반환합니다.
    (S3를 거치지 않고 직접 파일을 받습니다)
    """
    try:
        audio_content = await file.read()
        audio_io = io.BytesIO(audio_content)

        y, sr = librosa.load(audio_io, sr=None)

        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(y, sr=sr, alpha=0.8, ax=ax)
        
        # 파일명은 직접 받은 파일에서 가져옴
        uploaded_filename = file.filename 
        ax.set(title='Waveform of ' + uploaded_filename)
        
        ax.set(xlabel='Time (s)', ylabel='Amplitude')
        ax.grid(True)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        if "ffmpeg" in str(e).lower() or "codec" in str(e).lower():
            raise HTTPException(status_code=500, detail=f"오디오 처리 중 오류 발생: {str(e)}. MP3 파일을 로드하려면 ffmpeg이 설치 및 설정되어 있어야 합니다.")
        else:
            raise HTTPException(status_code=500, detail=f"파형 생성 중 오류 발생: {str(e)}")


# --- 기존 엔드포인트 3: 표준 MP3 파일의 파형 가져오기 ---
@app.get("/generate_standard_waveform")
async def generate_standard_waveform():
    """
    미리 로드된 표준 오디오 파일의 파형(waveform) 이미지를 생성하여 반환합니다.
    """
    if y_standard_audio_data is None or sr_standard_audio_data is None:
        raise HTTPException(status_code=500, detail="표준 오디오 파일이 로드되지 않았습니다.")

    try:
        fig, ax = plt.subplots(figsize=(10, 4))
        librosa.display.waveshow(y_standard_audio_data, sr=sr_standard_audio_data, alpha=0.8, ax=ax)
        ax.set(title='Waveform of Standard Audio (' + os.path.basename(STANDARD_AUDIO_PATH) + ')')
        ax.set(xlabel='Time (s)', ylabel='Amplitude')
        ax.grid(True)
        plt.tight_layout()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig)

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"표준 파형 생성 중 오류 발생: {str(e)}")


# --- **새로운 엔드포인트 4: 두 오디오 파일 파형을 투명하게 겹쳐서 생성** ---
@app.post("/generate_overlapped_waveform")
async def generate_overlapped_waveform(
    uploaded_file: UploadFile = File(..., alias="uploaded_audio_file")
):
    """
    업로드된 오디오 파일과 미리 로드된 표준 오디오 파일의 파형을
    투명하게 겹쳐서 하나의 이미지로 생성하여 반환합니다.
    """
    try:
        # 1. 업로드된 오디오 파일 로드
        uploaded_audio_content = await uploaded_file.read()
        uploaded_audio_io = io.BytesIO(uploaded_audio_content)
        y_uploaded, sr_uploaded = librosa.load(uploaded_audio_io, sr=None)
        
        # 2. 표준 오디오 파일 데이터 사용 (전역 변수 활용)
        if y_standard_audio_data is None or sr_standard_audio_data is None:
            raise HTTPException(status_code=500, detail="표준 오디오 파일이 서버에 로드되지 않았습니다.")
        
        y_standard = y_standard_audio_data
        sr_standard = sr_standard_audio_data

        # 3. 시간 축 생성 (두 오디오 중 더 긴 길이에 맞춤)
        time_standard = np.linspace(0, len(y_standard) / sr_standard, num=len(y_standard))
        time_uploaded = np.linspace(0, len(y_uploaded) / sr_uploaded, num=len(y_uploaded))

        # 그래프의 x축 길이를 두 오디오 중 더 긴 오디오에 맞춥니다.
        max_duration = max(len(y_standard) / sr_standard, len(y_uploaded) / sr_uploaded)

        # 4. 파형 플로팅 (투명도 조절)
        fig, ax = plt.subplots(figsize=(14, 6)) # 전체 그래프 크기 설정

        # 첫 번째 파형 (표준 오디오) 플로팅 (투명도 0.6)
        ax.plot(time_standard, y_standard, label='Standard Audio (' + os.path.basename(STANDARD_AUDIO_PATH) + ')', alpha=0.6, color='blue', linewidth=1)

        # 두 번째 파형 (업로드된 오디오) 플로팅 (투명도 0.6)
        ax.plot(time_uploaded, y_uploaded, label='Uploaded Audio (' + uploaded_file.filename + ')', alpha=0.6, color='red', linewidth=1)

        # 5. 그래프 꾸미기
        ax.set_title('Overlapped Waveform Comparison')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude')
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend()
        ax.set_ylim(-1.0, 1.0) # Y축 범위 고정
        ax.set_xlim(0, max_duration) # X축 범위 조정
        plt.tight_layout()

        # 6. 이미지를 메모리에 저장하고 반환
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig) # 메모리 누수 방지를 위해 figure 닫기

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        if "ffmpeg" in str(e).lower() or "codec" in str(e).lower():
            raise HTTPException(status_code=500, detail=f"오디오 처리 중 오류 발생: {str(e)}. MP3 파일을 로드하려면 ffmpeg이 설치 및 설정되어 있어야 합니다.")
        else:
            raise HTTPException(status_code=500, detail=f"겹쳐진 파형 생성 중 오류 발생: {str(e)}")

# FastAPI 서버 실행
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000, reload=True)