const express = require('express');
const multer = require('multer');
const axios = require('axios');
const path = require('path');
const cors = require('cors');
const FormData = require('form-data'); // 이 줄이 여전히 있어야 합니다.

const app = express();
const port = 8888; // Node.js 서버 포트

app.use(cors());
const storage = multer.memoryStorage();
const upload = multer({ storage: storage });
app.use(express.static(path.join(__dirname, 'public')));
app.get('/', (req, res) => {
    res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

// --- 변경된 부분 ---
// FastAPI 서버가 실행되는 EC2 인스턴스의 퍼블릭 IP 주소와 포트
const fastapiUrl = 'http://13.250.114.125:3000/audio_comparison';

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

        const response = await axios.post(fastapiUrl, formData, {
            headers: formData.getHeaders(),
            maxBodyLength: Infinity,
            maxContentLength: Infinity
        });

        console.log('Response from FastAPI:', response.data);
        res.json(response.data);

    } catch (error) {
        console.error('FastAPI 호출 중 오류 발생:', error.message);
        if (error.response) {
            console.error('FastAPI 응답 오류 데이터:', error.response.data);
            res.status(error.response.status).json(error.response.data);
        } else {
            res.status(500).json({ error: '오디오 분석 서버 통신 오류', detail: error.message });
        }
    }
});

app.listen(port, () => {
    console.log(`Node.js 서버가 http://localhost:${port} 에서 실행 중입니다.`);
    // 이 메시지는 로컬에서 Node.js 서버를 실행할 때만 해당하며,
    // EC2에 배포할 경우 실제 접속 주소는 EC2 IP가 됩니다.
    // console.log(`Node.js 서버가 http://13.250.114.125:${port} 에서 실행 중입니다.`); // 실제 배포 시 이렇게 변경 가능
});