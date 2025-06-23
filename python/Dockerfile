# 1. Python 3.11 버전을 기반으로 이미지 생성 (Better compatibility with BCC)
# 'slim-bookworm'은 Debian Bookworm 기반의 최소화된 Python 3.11 이미지입니다.
FROM python:3.12-slim-bookworm

# 2. 시스템 패키지 업데이트 및 필요한 도구 설치 (ffmpeg 등)
# '--no-install-recommends' 옵션으로 불필요한 추천 패키지 설치를 방지하여 이미지 크기를 줄이고 충돌을 피합니다.
# 'libsndfile1'은 'soundfile' 라이브러리 (audio_analyzer.py에서 사용될 가능성 높음)의 의존성입니다.
# 추가로 build-essential을 설치하여 C 컴파일러와 기타 빌드 도구를 제공합니다.
RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# 3. 컨테이너 내부의 작업 디렉토리 설정
# 모든 이후 명령어는 이 디렉토리를 기준으로 실행됩니다.
WORKDIR /app

# 4. Python 의존성 파일 (requirements.txt) 복사
# 의존성 파일을 먼저 복사하여 pip install이 캐싱될 수 있도록 합니다.
# 이는 코드 변경 시 불필요한 의존성 재설치를 방지하여 빌드 시간을 단축시킵니다.
COPY requirements.txt .

# 5. requirements.txt에 명시된 Python 의존성 설치
# '--no-cache-dir'은 pip 캐시를 사용하지 않아 최종 이미지 크기를 줄입니다.
RUN pip install --no-cache-dir -r requirements.txt

# 6. 모든 Python 소스 코드 및 관련 데이터 (예: mp3 폴더)를 컨테이너로 복사
# 현재 로컬 디렉토리(python_service)의 모든 내용을 컨테이너의 /app 디렉토리로 복사합니다.
# 만약 python/audio_comparison.py에서 mp3 폴더를 상대 경로로 참조한다면,
# 컨테이너 내부 구조가 'python/mp3'와 같이 유지되도록 './python/'으로 복사하는 것이 적절합니다.
COPY . /app/python/

# 7. PYTHONPATH 환경 변수 설정
# Python 인터프리터가 모듈을 찾을 때 /app/python 디렉토리도 검색하도록 경로를 추가합니다.
# 이는 'python.audio_comparison:app'과 같이 모듈 경로를 지정할 때 필요합니다.
ENV PYTHONPATH=/app/python

# 8. FastAPI 애플리케이션이 외부 연결을 수신 대기할 포트 선언 (문서화 목적)
# 실제 포트 매핑은 docker-compose.yml에서 이루어집니다.
EXPOSE 3000

# 9. 컨테이너 시작 시 실행될 기본 명령어
# Uvicorn은 ASGI 서버로 FastAPI 애플리케이션을 실행합니다.
# 'python.audio_comparison:app'은 /app/python/audio_comparison.py 파일 내의 'app' 객체를 실행함을 의미합니다.
# '--host 0.0.0.0'은 컨테이너 외부에서의 접근을 허용합니다.
# '--port 3000'은 Uvicorn이 컨테이너 내부에서 리스닝할 포트입니다.
CMD ["uvicorn", "python.audio_comparison:app", "--host", "0.0.0.0", "--port", "3000"]