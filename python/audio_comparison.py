# python/audio_comparison.py (S3 도입 전 버전)
import librosa
import librosa.display
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
import io
import os
import uvicorn
import matplotlib.pyplot as plt
# requests 모듈은 S3 URL 다운로드에 필요했으므로, 이 버전에서는 필요 없습니다.
# import requests 

app = FastAPI()

# --- 0. 표준 오디오 파일 로드 (미리 로드하여 효율성 높임) ---
STANDARD_AUDIO_PATH = "/work/speechfeedback-service/python/mp3/speech_ai_liam.mp3"
STANDARD_N_MFCC = 13

# 표준 오디오 데이터를 전역 변수로 저장
y_standard_audio_data = None
sr_standard_audio_data = None
mfccs_standard_mean = None

try:
    y_standard_audio_data, sr_standard_audio_data = librosa.load(STANDARD_AUDIO_PATH, sr=None)
    mfccs_standard = librosa.feature.mfcc(y=y_standard_audio_data, sr=sr_standard_audio_data, n_mfcc=STANDARD_N_MFCC)
    mfccs_standard_mean = np.mean(mfccs_standard, axis=1)
    print(f"표준 오디오 파일 '{STANDARD_AUDIO_PATH}' 로드 및 MFCC 계산 완료.")
except Exception as e:
    print(f"FATAL ERROR: 표준 오디오 파일 로드 또는 MFCC 계산 실패: {e}")
    print("FastAPI 서버가 시작되지 않습니다. 표준 오디오 파일 경로와 파일 존재 여부를 확인하세요.")
    exit(1)

class SimilarityResult(BaseModel):
    similarity_score: float
    message: str
    detail: str = None

# S3 관련 모델은 이 버전에서 필요 없습니다.
# class S3FileRequest(BaseModel):
#     file_url: str

# --- 기존 엔드포인트 1: 파일 직접 받아서 유사도 분석 ---
@app.post("/analyze_audio_similarity", response_model=SimilarityResult)
async def analyze_audio_similarity(file: UploadFile = File(...)):
    """
    업로드된 오디오 파일과 표준 AI 오디오 파일 간의 MFCC 기반 유사도를 분석합니다.
    (S3를 거치지 않고 직접 파일을 받습니다)
    """
    try:
        audio_content = await file.read() # 업로드된 파일 내용을 읽음
        audio_io = io.BytesIO(audio_content) # 메모리 버퍼로 변환

        y_uploaded, sr_uploaded = librosa.load(audio_io, sr=None)
        mfccs_uploaded = librosa.feature.mfcc(y=y_uploaded, sr=sr_uploaded, n_mfcc=STANDARD_N_MFCC)
        mfccs_mean_uploaded = np.mean(mfccs_uploaded, axis=1)
        similarity_score = cosine_similarity(
            mfccs_standard_mean.reshape(1, -1),
            mfccs_mean_uploaded.reshape(1, -1)
        )[0][0]
        if similarity_score > 0.9:
            message = "두 오디오는 매우 유사합니다."
        elif similarity_score > 0.7:
            message = "두 오디오는 유사합니다."
        elif similarity_score > 0.5:
            message = "두 오디오는 어느 정도 유사성을 보이지만, 차이가 있을 수 있습니다."
        else:
            message = "두 오디오는 유사성이 낮습니다."
        
        # 파일명은 직접 받은 파일에서 가져옴
        uploaded_filename = file.filename 
        
        return SimilarityResult(
            similarity_score=float(similarity_score),
            message=message,
            detail=f"업로드된 파일: {uploaded_filename}, 표준 파일: {os.path.basename(STANDARD_AUDIO_PATH)}"
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


# --- 새로운 엔드포인트 3: 표준 MP3 파일의 파형 생성 ---
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

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000, reload=True)