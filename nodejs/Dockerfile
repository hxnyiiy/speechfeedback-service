# node_service/Dockerfile

# 1. Node.js 18 버전을 기반으로 이미지 생성
FROM node:18-alpine

# 2. 작업 디렉토리 설정
WORKDIR /app

# 3. package.json 복사 및 Node.js 의존성 설치
COPY package.json .
RUN npm install

# 4. 모든 Node.js 소스 코드 및 public 폴더 복사
COPY . .

# 5. Express 서버가 사용할 포트 노출
EXPOSE 8888

# 6. 컨테이너 시작 시 실행될 명령어
CMD ["npm", "start"]
