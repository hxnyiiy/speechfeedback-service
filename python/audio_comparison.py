import librosa
import librosa.display
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import io
import os
import uvicorn # uvicorn을 import 해야 합니다.

app = FastAPI()

# --- 0. 표준 오디오 파일 로드 (미리 로드하여 효율성 높임) ---
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
    exit(1) # 서버 시작 전에 종료

# MFCC 유사도 결과 모델
class SimilarityResult(BaseModel):
    similarity_score: float
    message: str
    detail: str = None

@app.post("/audio_comparison", response_model=SimilarityResult)
async def audio_comparison(file: UploadFile = File(...)):
    """
    업로드된 오디오 파일과 표준 AI 오디오 파일 간의 MFCC 기반 유사도를 분석합니다.
    """
    try:
        # 업로드된 파일 읽기
        audio_content = await file.read()
        audio_io = io.BytesIO(audio_content)

        # librosa로 오디오 로드
        y_uploaded, sr_uploaded = librosa.load(audio_io, sr=None)

        # MFCC 특징 추출 (업로드된 파일)
        mfccs_uploaded = librosa.feature.mfcc(y=y_uploaded, sr=sr_uploaded, n_mfcc=STANDARD_N_MFCC)
        mfccs_mean_uploaded = np.mean(mfccs_uploaded, axis=1)

        # 코사인 유사도 계산
        # cosine_similarity 함수는 2D 배열을 입력으로 받으므로, reshape(1, -1)를 사용합니다.
        similarity_score = cosine_similarity(
            mfccs_mean_standard.reshape(1, -1),
            mfccs_mean_uploaded.reshape(1, -1)
        )[0][0]

        # 유사도 값 해석
        if similarity_score > 0.9:
            message = "두 오디오는 매우 유사합니다."
        elif similarity_score > 0.7:
            message = "두 오디오는 유사합니다."
        elif similarity_score > 0.5:
            message = "두 오디오는 어느 정도 유사성을 보이지만, 차이가 있을 수 있습니다."
        else:
            message = "두 오디오는 유사성이 낮습니다."

        return SimilarityResult(
            similarity_score=float(similarity_score), # numpy float64를 일반 float로 변환
            message=message,
            detail=f"업로드된 파일: {file.filename}, 표준 파일: {os.path.basename(STANDARD_AUDIO_PATH)}"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"오디오 분석 중 오류 발생: {str(e)}")

# --- 서버 실행 부분 추가 ---
if __name__ == "__main__":
    # uvicorn.run(app)을 사용하여 FastAPI 애플리케이션을 실행합니다.
    # host="0.0.0.0"은 모든 네트워크 인터페이스에서 접근 가능하게 합니다.
    # port=3000은 지정된 포트로 서버를 띄웁니다.
    # reload=True는 코드 변경 시 자동으로 서버를 재시작합니다 (개발용).
    uvicorn.run(app, host="0.0.0.0", port=3000, reload=True)