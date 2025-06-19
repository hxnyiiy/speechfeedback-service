# python_service/Dockerfile

# 1. Python 3.9 버전을 기반으로 이미지 생성
FROM python:3.9-slim-buster

# 2. 시스템 업데이트 및 ffmpeg 설치
# librosa가 MP3 파일을 처리하려면 ffmpeg이 필요합니다.
RUN apt-get update && \
    apt-get install -y ffmpeg && \
    rm -rf /var/lib/apt/lists/*

# 3. 작업 디렉토리 설정
WORKDIR /app

# 4. requirements.txt 복사 및 Python 의존성 설치
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 5. 모든 Python 소스 코드 및 MP3 폴더 복사
# FastAPI 코드와 mp3 폴더를 /app/python/ 경로에 복사
# 이는 audio_comparison.py 내부의 STANDARD_AUDIO_PATH와 일치해야 합니다.
COPY . /app/python/ 

# 6. 환경 변수 설정 (FastAPI 앱 실행 시 필요)
ENV PYTHONPATH=/app/python

# 7. FastAPI 애플리케이션이 사용할 포트 노출
EXPOSE 3000

# 8. 컨테이너 시작 시 실행될 명령어 (Uvicorn으로 FastAPI 앱 실행)
# FastAPI 앱 파일이 /app/python/audio_comparison.py 이므로 모듈 경로를 정확히 지정합니다.
# --host 0.0.0.0 은 외부 접속 허용, --port 3000은 FastAPI가 리스닝할 포트
CMD ["uvicorn", "python.audio_comparison:app", "--host", "0.0.0.0", "--port", "3000"]
