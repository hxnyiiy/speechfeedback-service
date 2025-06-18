const express = require('express');
const multer = require('multer');
const axios = require('axios');
const path = require('path');
const cors = require('cors');
const FormData = require('form-data');

const app = express();
const port = 8888; // 또는 3001

app.use(cors());
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });
app.use(express.static(path.join(__dirname, 'public')));
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

const fastapiUrl = 'http://13.250.114.125:3000'; // FastAPI 기본 URL

app.post('/upload_and_analyze', upload.single('audioFile'), async (req, res) => {
    // --- 기존 유사도 분석 로직 (변경 없음) ---
    if (!req.file) {
        return res.status(400).json({ error: '오디오 파일이 업로드되지 않았습니다.' });
    }

    try {
        const formData = new FormData();
        formData.append('file', req.file.buffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });

        const response = await axios.post(fastapiUrl + '/analyze_audio_similarity', formData, {
            headers: formData.getHeaders(),
            maxBodyLength: Infinity,
            maxContentLength: Infinity
        });

        console.log('Response from FastAPI (similarity):', response.data);
        res.json(response.data);

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

// --- 변경된 부분: 파형 이미지 요청 엔드포인트 ---
app.post('/get_waveform', upload.single('audioFile'), async (req, res) => {
    if (!req.file) {
        return res.status(400).json({ error: '오디오 파일이 업로드되지 않았습니다.' });
    }

    try {
        const formData = new FormData();
        formData.append('file', req.file.buffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });

        const response = await axios.post(fastapiUrl + '/generate_waveform', formData, {
            headers: formData.getHeaders(),
            responseType: 'stream', // 중요: 이미지 데이터를 스트림으로 받음
            maxBodyLength: Infinity,
            maxContentLength: Infinity
        });

        res.setHeader('Content-Type', response.headers['content-type']);
        response.data.pipe(res); // FastAPI에서 받은 이미지 스트림을 그대로 프론트엔드로 전달

    } catch (error) {
        console.error('FastAPI 호출 중 오류 발생 (waveform):', error.message);
        if (error.response) {
            console.error('FastAPI 응답 오류 데이터 (waveform):', error.response.data);
            res.status(error.response.status).send(error.response.data); // send로 변경하여 오류 메시지 보장
        } else {
            res.status(500).json({ error: '파형 서버 통신 오류', detail: error.message });
        }
    }
});

app.listen(port, () => {
    console.log(`Node.js 서버가 http://13.250.114.125:${port} 에서 실행 중입니다.`);
});