// nodejs/node_server.js

const express = require('express');
const app = express();
const multer = require('multer');
const axios = require('axios');
const path = require('path');
const cors = require('cors');
const FormData = require('form-data'); // FormData 라이브러리

app.use(cors()); // CORS 미들웨어 적용
app.use(express.json()); // JSON 요청 바디 파싱
app.use(express.urlencoded({ extended: true })); // URL-encoded 요청 바디 파싱

const port = 8888; // Node.js 서버 포트
// FastAPI 기본 URL (EC2 인스턴스의 퍼블릭 IP와 포트를 사용)
const fastapiUrl = 'http://13.250.114.125:3000'; 

// Multer 설정 (파일을 서버의 메모리에 임시로 저장)
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

// 정적 파일 제공 (프론트엔드 HTML/JS 파일이 있는 public 디렉토리)
app.use(express.static(path.join(__dirname, 'public')));

// 루트 경로 ('/') 접속 시 index.html 파일 서빙
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// --- 엔드포인트 1: 오디오 파일 업로드 후 FastAPI에 직접 전달 (유사도 분석) ---
app.post('/upload_and_analyze', upload.single('audioFile'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: '오디오 파일이 업로드되지 않았습니다.' });
    }

    try {
        // FormData 객체 생성 및 파일 데이터 추가
        const formData = new FormData();
        formData.append('file', req.file.buffer, { // FastAPI의 UploadFile 파라미터 이름 'file'과 일치
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });
        
        // FastAPI 서버의 유사도 분석 엔드포인트 호출
        // S3가 없으므로 /analyze_audio_similarity 엔드포인트로 직접 파일 전송
        const response = await axios.post(fastapiUrl + '/analyze_audio_similarity', formData, {
            headers: formData.getHeaders(), // FormData의 Content-Type 헤더를 자동으로 설정
            maxBodyLength: Infinity, // 큰 파일에 대한 요청 본문 길이 제한 해제
            maxContentLength: Infinity // 큰 파일에 대한 응답 본문 길이 제한 해제
        });

        console.log('Response from FastAPI (similarity):', response.data);
        res.json(response.data); // FastAPI의 응답을 프론트엔드로 전달

    } catch (error) {
        console.error('FastAPI 호출 중 오류 발생 (similarity):', error.message);
        if (error.response) {
            console.error('FastAPI 응답 오류 데이터 (similarity):', error.response.data);
            res.status(error.response.status).json(error.response.data);
        } else {
            res.status(500).json({ error: '오디오 분석 서버 통신 오류 (similarity)', detail: error.message });
        }
    }
});

// --- 엔드포인트 2: 오디오 파일 업로드 후 FastAPI에 직접 전달 (파형 생성) ---
app.post('/get_waveform', upload.single('audioFile'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: '오디오 파일이 업로드되지 않았습니다.' });
    }

    try {
        const formData = new FormData();
        formData.append('file', req.file.buffer, { // FastAPI의 UploadFile 파라미터 이름 'file'과 일치
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });

        // FastAPI 서버의 파형 생성 엔드포인트 호출
        // S3가 없으므로 /generate_waveform 엔드포인트로 직접 파일 전송
        const response = await axios.post(fastapiUrl + '/generate_waveform', formData, {
            headers: formData.getHeaders(),
            responseType: 'stream', // 이미지 데이터를 스트림으로 받음
            maxBodyLength: Infinity,
            maxContentLength: Infinity
        });

        res.setHeader('Content-Type', response.headers['content-type']); // FastAPI가 보낸 이미지 MIME 타입 설정
        response.data.pipe(res); // 스트림 파이프 연결

    } catch (error) {
        console.error('FastAPI 호출 중 오류 발생 (waveform):', error.message);
        if (error.response) {
            console.error('FastAPI 응답 오류 데이터 (waveform):', error.response.data);
            res.status(error.response.status).send(error.response.data);
        } else {
            res.status(500).json({ error: '파형 생성 중 오류 발생', detail: error.message });
        }
    }
});

// --- 엔드포인트 3: 표준 MP3 파일의 파형 가져오기 ---
app.get('/get_standard_waveform', async (req, res) => {
    try {
        // FastAPI 서버의 표준 파형 생성 엔드포인트 호출
        const response = await axios.get(fastapiUrl + '/generate_standard_waveform', {
            responseType: 'stream', // 이미지 데이터를 스트림으로 받음
            maxBodyLength: Infinity,
            maxContentLength: Infinity
        });

        res.setHeader('Content-Type', response.headers['content-type']);
        response.data.pipe(res);

    } catch (error) {
        console.error('FastAPI 호출 중 오류 발생 (표준 파형):', error.message);
        if (error.response) {
            console.error('FastAPI 응답 오류 데이터 (표준 파형):', error.response.data);
            res.status(error.response.status).send(error.response.data);
        } else {
            res.status(500).json({ error: '표준 파형 생성 중 오류 발생', detail: error.message });
        }
    }
});

// --- **새로운 엔드포인트 4: 업로드된 파일과 표준 파일을 받아 겹쳐진 파형 이미지 생성** ---
app.post('/get_overlapped_waveform', upload.single('audioFile'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: '오디오 파일이 업로드되지 않았습니다.' });
    }

    try {
        const formData = new FormData();
        // 'uploaded_audio_file'은 FastAPI 엔드포인트에서 alias로 지정한 이름과 일치해야 합니다.
        formData.append('uploaded_audio_file', req.file.buffer, { 
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });

        // FastAPI 서버의 겹쳐진 파형 생성 엔드포인트 호출
        const response = await axios.post(fastapiUrl + '/generate_overlapped_waveform', formData, {
            headers: formData.getHeaders(),
            responseType: 'stream', // 이미지 데이터를 스트림으로 받음
            maxBodyLength: Infinity,
            maxContentLength: Infinity
        });

        res.setHeader('Content-Type', response.headers['content-type']);
        response.data.pipe(res);

    } catch (error) {
        console.error('FastAPI 호출 중 오류 발생 (겹쳐진 파형):', error.message);
        if (error.response) {
            console.error('FastAPI 응답 오류 데이터 (겹쳐진 파형):', error.response.data);
            res.status(error.response.status).send(error.response.data);
        } else {
            res.status(500).json({ error: '겹쳐진 파형 생성 중 오류 발생', detail: error.message });
        }
    }
});

// Express 서버 리스닝 시작
app.listen(port, () => {
    console.log(`Node.js 서버가 http://localhost:${port} 에서 실행 중입니다.`);
    console.log(`(실제 EC2 접속 주소: http://13.250.114.125:${port})`);
});