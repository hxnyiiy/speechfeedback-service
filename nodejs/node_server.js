// node_server.js

const express = require('express');
const app = express();
const multer = require('multer');
const axios = require('axios');
const path = require('path');
const cors = require('cors');
const FormData = require('form-data');

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

const port = 8000;
const fastapiUrl = 'http://13.250.114.125:3000';

const storage = multer.memoryStorage();
const upload = multer({ storage: storage });

app.use(express.static(path.join(__dirname, 'public')));

app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// ✅ 변경된 upload_and_analyze 엔드포인트
app.post('/upload_and_analyze', upload.single('audioFile'), async (req, res) => {
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

        const data = response.data;
        const score = data.similarity_score;

        // 🎯 유사도에 따른 발표 피드백 메시지 추가
        let feedback = '';
        if (score >= 0.9) {
            feedback = '🎯 훌륭해요! AI와 거의 유사한 발음이에요!';
        } else if (score >= 0.75) {
            feedback = '👍 좋아요! 발음을 조금만 더 다듬으면 완벽할 수 있어요.';
        } else if (score >= 0.5) {
            feedback = '📝 기본은 되어 있어요. 계속 연습하면 더 나아질 수 있어요!';
        } else {
            feedback = '📢 조금 더 연습이 필요해요. 반복 학습을 추천드려요!';
        }

        // 기존 응답 + 피드백 포함해서 클라이언트로 전달
        res.json({
            similarity_score: score,
            message: data.message,
            detail: data.detail,
            feedback: feedback
        });

    } catch (error) {
        console.error('FastAPI 호출 중 오류 발생 (similarity):', error.message);
        if (error.response) {
            res.status(error.response.status).json(error.response.data);
        } else {
            res.status(500).json({ error: '오디오 분석 서버 통신 오류', detail: error.message });
        }
    }
});

// --- 이하 기존 엔드포인트들 동일 ---
app.post('/get_waveform', upload.single('audioFile'), async (req, res) => {
    if (!req.file) return res.status(400).json({ error: '오디오 파일이 업로드되지 않았습니다.' });

    try {
        const formData = new FormData();
        formData.append('file', req.file.buffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });

        const response = await axios.post(fastapiUrl + '/generate_waveform', formData, {
            headers: formData.getHeaders(),
            responseType: 'stream',
            maxBodyLength: Infinity,
            maxContentLength: Infinity
        });

        res.setHeader('Content-Type', response.headers['content-type']);
        response.data.pipe(res);

    } catch (error) {
        console.error('FastAPI 호출 중 오류 발생 (waveform):', error.message);
        res.status(500).json({ error: '파형 생성 오류', detail: error.message });
    }
});

app.get('/get_standard_waveform', async (req, res) => {
    try {
        const response = await axios.get(fastapiUrl + '/generate_standard_waveform', {
            responseType: 'stream',
            maxBodyLength: Infinity,
            maxContentLength: Infinity
        });

        res.setHeader('Content-Type', response.headers['content-type']);
        response.data.pipe(res);
    } catch (error) {
        console.error('FastAPI 호출 중 오류 발생 (표준 파형):', error.message);
        res.status(500).json({ error: '표준 파형 오류', detail: error.message });
    }
});

app.post('/get_overlapped_waveform', upload.single('audioFile'), async (req, res) => {
    if (!req.file) return res.status(400).json({ error: '오디오 파일이 업로드되지 않았습니다.' });

    try {
        const formData = new FormData();
        formData.append('uploaded_audio_file', req.file.buffer, {
            filename: req.file.originalname,
            contentType: req.file.mimetype
        });

        const response = await axios.post(fastapiUrl + '/generate_overlapped_waveform', formData, {
            headers: formData.getHeaders(),
            responseType: 'stream',
            maxBodyLength: Infinity,
            maxContentLength: Infinity
        });

        res.setHeader('Content-Type', response.headers['content-type']);
        response.data.pipe(res);
    } catch (error) {
        console.error('FastAPI 호출 중 오류 발생 (겹쳐진 파형):', error.message);
        res.status(500).json({ error: '겹쳐진 파형 오류', detail: error.message });
    }
});

app.listen(port, () => {
    console.log(`Node.js 서버가 http://localhost:${port} 에서 실행 중입니다.`);
});
