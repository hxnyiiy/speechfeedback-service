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

app = FastAPI()

# --- 기존 코드 (표준 오디오 로드 및 MFCC 관련) ---
STANDARD_AUDIO_PATH = "/work/speechfeedback-service/python/mp3/speech_ai_liam.mp3"
STANDARD_N_MFCC = 13

try:
    y_standard, sr_standard = librosa.load(STANDARD_AUDIO_PATH, sr=None)
    mfccs_standard = librosa.feature.mfcc(y=y_standard, sr=sr_standard, n_mfcc=STANDARD_N_MFCC)
    mfccs_mean_standard = np.mean(mfccs_standard, axis=1)
    print(f"표준 오디오 파일 '{STANDARD_AUDIO_PATH}' 로드 및 MFCC 계산 완료.")
except Exception as e:
    print(f"FATAL ERROR: 표준 오디오 파일 로드 또는 MFCC 계산 실패: {e}")
    print("FastAPI 서버가 시작되지 않습니다. 표준 오디오 파일 경로와 파일 존재 여부를 확인하세요.")
    exit(1)

class SimilarityResult(BaseModel):
    similarity_score: float
    message: str
    detail: str = None

@app.post("/analyze_audio_similarity", response_model=SimilarityResult)
async def analyze_audio_similarity(file: UploadFile = File(...)):
    # --- 기존 유사도 분석 로직 (변경 없음) ---
    try:
        audio_content = await file.read()
        audio_io = io.BytesIO(audio_content)
        y_uploaded, sr_uploaded = librosa.load(audio_io, sr=None)
        mfccs_uploaded = librosa.feature.mfcc(y=y_uploaded, sr=sr_uploaded, n_mfcc=STANDARD_N_MFCC)
        mfccs_mean_uploaded = np.mean(mfccs_uploaded, axis=1)
        similarity_score = cosine_similarity(
            mfccs_mean_standard.reshape(1, -1),
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
        return SimilarityResult(
            similarity_score=float(similarity_score),
            message=message,
            detail=f"업로드된 파일: {file.filename}, 표준 파일: {os.path.basename(STANDARD_AUDIO_PATH)}"
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오디오 분석 중 오류 발생: {str(e)}")

# --- 변경된 부분: 파형 이미지 생성 엔드포인트 ---
@app.post("/generate_waveform")
async def generate_waveform(file: UploadFile = File(...)):
    """
    업로드된 오디오 파일의 파형(waveform) 이미지를 생성하여 반환합니다.
    """
    try:
        audio_content = await file.read()
        audio_io = io.BytesIO(audio_content)
        y, sr = librosa.load(audio_io, sr=None)

        # Matplotlib 플롯 설정 (스펙트로그램과 동일한 방식으로)
        fig, ax = plt.subplots(figsize=(10, 4)) # figsize 조정 가능
        librosa.display.waveshow(y, sr=sr, alpha=0.8, ax=ax)
        ax.set(title='Waveform of ' + file.filename)
        ax.set(xlabel='Time (s)', ylabel='Amplitude')
        ax.grid(True)
        plt.tight_layout()

        # 이미지를 메모리 버퍼에 저장
        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close(fig) # 메모리 해제

        return StreamingResponse(buf, media_type="image/png")

    except Exception as e:
        # MP3 파일 로드 실패 시 ffmpeg 관련 메시지 포함
        if "ffmpeg" in str(e).lower() or "codec" in str(e).lower():
            raise HTTPException(status_code=500, detail=f"오디오 로드 중 오류 발생: {str(e)}. MP3 파일을 로드하려면 ffmpeg이 설치 및 설정되어 있어야 합니다.")
        else:
            raise HTTPException(status_code=500, detail=f"파형 생성 중 오류 발생: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000, reload=True)