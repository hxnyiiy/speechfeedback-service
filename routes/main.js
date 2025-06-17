// routes/main.js
const express = require('express');
const router = express.Router(); // 라우터 인스턴스
const multer = require('multer');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

// .env는 app.js에서 로드되므로 여기서 다시 로드할 필요가 없습니다.

const AWS = require('aws-sdk');
const ID = process.env.ID;
const SECRET = process.env.SECRET;
const BUCKET_NAME = 'kibwa-06'; // 사용하는 S3 버킷 이름
const MYREGION = 'ap-northeast-3'; // S3 버킷이 위치한 리전
const s3 = new AWS.S3({ accessKeyId: ID, secretAccessKey: SECRET, region: MYREGION });

// Multer 스토리지 설정: 로컬 디스크에 파일 저장
var storage = multer.diskStorage({
    destination(req, file, cb) {
        cb(null, 'uploadedFiles/');
    },
    filename(req, file, cb) {
        cb(null, `${Date.now()}_${file.originalname}`);
    },
});
var upload = multer({ storage: storage }).fields([
    { name: 'audiofile1', maxCount: 1 },
    { name: 'audiofile2', maxCount: 1 }
]);

// LLM을 사용하여 오디오 분석을 시뮬레이션하는 함수
// 실제 오디오 분석은 복잡하며, 여기서는 LLM을 사용하여 더미 데이터를 생성합니다.
const { transcribeAudio } = require('../lib/whisper');
const Transcription = require('../models/transcription');

async function analyzeAudioFiles(file1Key, file2Key) {
    try {
        // Transcribe both files using Whisper
        const transcription1 = await transcribeAudio(file1Key);
        const transcription2 = await transcribeAudio(file2Key);

        // Analyze the transcripts
        const analysis1 = analyzeTranscript(transcription1.transcript);
        const analysis2 = analyzeTranscript(transcription2.transcript);

        // Save results to MongoDB
        await saveTranscriptionToDB(file1Key, transcription1, analysis1);
        await saveTranscriptionToDB(file2Key, transcription2, analysis2);

        // Calculate differences
        const speechRateChange = analysis2.speechRate - analysis1.speechRate;
        const fillerWordsChange = analysis2.fillerWordsCount - analysis1.fillerWordsCount;

        return {
            file1: {
                key: file1Key,
                transcript: transcription1.transcript,
                speechRate: analysis1.speechRate,
                fillerWordsCount: analysis1.fillerWordsCount,
                analysis: analysis1.analysis,
                segments: transcription1.segments
            },
            file2: {
                key: file2Key,
                transcript: transcription2.transcript,
                speechRate: analysis2.speechRate,
                fillerWordsCount: analysis2.fillerWordsCount,
                analysis: analysis2.analysis,
                segments: transcription2.segments
            },
            speechRateChange,
            fillerWordsChange
        };
    } catch (error) {
        console.error('Error analyzing audio files:', error);
        throw error;
    }
}

// Function to save transcription to MongoDB
async function saveTranscriptionToDB(fileKey, transcription, analysis) {
    try {
        const existingTranscription = await Transcription.findOne({ fileKey });
        if (existingTranscription) {
            // Update existing transcription
            existingTranscription.transcript = transcription.transcript;
            existingTranscription.segments = transcription.segments;
            existingTranscription.analysis = analysis;
            existingTranscription.language = transcription.language;
            await existingTranscription.save();
        } else {
            // Create new transcription
            const newTranscription = new Transcription({
                fileKey,
                fileName: fileKey.split('/').pop(),
                transcript: transcription.transcript,
                segments: transcription.segments,
                analysis: analysis,
                language: transcription.language
            });
            await newTranscription.save();
        }
    } catch (error) {
        console.error('Error saving transcription to MongoDB:', error);
        throw error;
    }
}

// Function to analyze transcribed text
function analyzeTranscript(text) {
    // Simple analysis of transcribed text
    const words = text.split(' ').filter(word => word.trim() !== '');
    const wordCount = words.length;
    
    // Korean filler words
    const fillerWords = ['음', '아', '어', '여', '이', '네', '그', '그런데', '그리고', '그런', '그러면', '그러니까'];
    const fillerWordsCount = words.filter(word => fillerWords.includes(word)).length;
    
    // Estimate speech rate (words per minute)
    const speechRate = Math.floor(wordCount * 1.5);

    return {
        speechRate,
        fillerWordsCount,
        totalWords: wordCount,
        analysis: `Transcription analysis:\n\nTotal words: ${wordCount}\nFiller words: ${fillerWordsCount}\nEstimated speech rate: ${speechRate} WPM\n\nNote: This analysis is based on the transcribed text using Whisper.`
    };
}


// --- 라우터 정의 ---

// 기본 경로('/') GET 요청: 파일 업로드 폼을 렌더링
router.get('/', function (req, res) {
    res.render('upload'); // public/upload.ejs 파일을 렌더링
});

// S3 파일 목록 GET 요청 ('/list'): S3 버킷의 객체 목록을 가져와 HTML 테이블로 렌더링
router.get('/list', (req, res) => {
    const params = {
        Bucket: BUCKET_NAME,
        Delimiter: '/',
        Prefix: 'uploadedFiles/', // 'uploadedFiles/' 접두사로 시작하는 객체만 리스팅
    };

    s3.listObjects(params, function (err, data) {
        if (err) {
            console.error("S3 listObjects 오류:", err);
            return res.status(500).send("<h1>Error!</h1><p>S3 객체 목록을 가져오는 데 실패했습니다.</p><pre>" + err.message + "</pre>");
        }

        res.writeHead(200, { 'Content-Type': 'text/html; charset=utf-8' });
        let template = `
            <!doctype html>
            <html>
            <head>
                <title>S3 파일 목록</title>
                <meta charset="utf-8">
                <style>
                    body { font-family: sans-serif; margin: 0; padding: 0; }
                    table { width: 100%; border-collapse: collapse; margin-top: 1rem; }
                    th, td { border: 1px solid #ddd; padding: 8px; text-align: left; word-break: break-all; }
                    th { background-color: #f2f2f2; }
                    .audio-player { width: 100%; max-width: 200px; }
                    button { background-color: #007bff; color: white; padding: 5px 10px; border: none; border-radius: 4px; cursor: pointer; font-size: 0.8em; }
                    button:hover { background-color: #0056b3; }
                    form { display: inline; margin: 0; padding: 0; }
                </style>
            </head>
            <body>
                <table border="1">
                    <tr>
                        <th>Key</th>
                        <th>LastModified</th>
                        <th>Size (Bytes)</th>
                        <th>StorageClass</th>
                        <th>재생</th>
                        <th>다운로드</th>
                        <th>삭제</th>
                    </tr>
        `;

        if (data.Contents && data.Contents.length > 0) {
            data.Contents.forEach(item => {
                if (item.Key === params.Prefix || item.Size === 0) {
                    return; // 폴더 자체이거나 빈 파일은 건너뛰기
                }
                const filename = item.Key;
                const fileurl = `https://${BUCKET_NAME}.s3.${MYREGION}.amazonaws.com/${encodeURIComponent(filename)}`;

                template += `
                    <tr>
                        <td>${filename}</td>
                        <td>${item['LastModified']}</td>
                        <td>${item['Size']}</td>
                        <td>${item['StorageClass']}</td>
                        <td>
                            <audio controls class="audio-player">
                                <source src="${fileurl}" type="audio/mpeg">
                                Your browser does not support the audio element.
                            </audio>
                        </td>
                        <td>
                            <form method='post' action='/downloadFile'>
                                <button type='submit' name='dlKey' value="${filename}">Down</button>
                            </form>
                        </td>
                        <td>
                            <form method='post' action='/deleteFile'>
                                <button type='submit' name='dlKey' value="${filename}">Del</button>
                            </form>
                        </td>
                    </tr>
                `;
            });
        } else {
            template += `<tr><td colspan="7">업로드된 파일이 없습니다.</td></tr>`;
        }

        template += `
                </table>
            </body>
            </html>
            `;
        res.end(template);
    });
});

// 파일 업로드 POST 요청 ('/uploadFile'): 로컬에 저장 후 S3에 업로드, 로컬 파일 삭제
router.post('/uploadFile', upload.single('audiofile'), async (req, res) => {
    if (!req.file) {
        return res.status(400).render('confirmation', { file: { error: '업로드할 파일이 없습니다.' }, files: null });
    }

    const localFilename = req.file.filename;
    const localFilePath = path.join(__dirname, '../uploadedFiles', localFilename);
    const s3Key = `uploadedFiles/${localFilename}`; // S3 Key는 로컬 파일의 이름을 사용

    try {
        const fileContent = fs.readFileSync(localFilePath); // 동기적 파일 읽기

        const params = {
            Bucket: BUCKET_NAME,
            Key: s3Key,
            Body: fileContent,
            ContentType: req.file.mimetype || 'application/octet-stream',
            // ACL: 'public-read' // <-- 제거: S3 버킷 정책으로 공개 접근 관리
        };

        const s3UploadResult = await new Promise((resolve, reject) => {
            s3.upload(params, (err, data) => {
                if (err) {
                    console.error("S3 업로드 오류:", err);
                    reject(err);
                } else {
                    console.log(`File uploaded to S3 successfully. ${data.Location}`);
                    resolve(data);
                }
            });
        });

        fs.unlink(localFilePath, err => { // 로컬 파일 삭제
            if (err) console.error(`Local file delete error: ${err}`);
            else console.log(`Local file deleted successfully: ${localFilePath}`);
        });

        res.render('confirmation', { file: s3UploadResult, files: null });

    } catch (error) {
        console.error('파일 업로드 및 S3 처리 중 오류 발생:', error);
        res.status(500).render('confirmation', { file: { error: error.message || '파일 업로드 실패' }, files: null });
    }
});

// 파일 다운로드 POST 요청 ('/downloadFile')
router.post('/downloadFile', async (req, res) => {
    const filename = req.body.dlKey;
    console.log(`Downloading file: ${filename}`);

    const params = {
        Bucket: BUCKET_NAME,
        Key: filename,
    };

    try {
        const data = await s3.getObject(params).promise();
        res.attachment(filename.split('/').pop()); // S3 Key에서 파일명만 추출
        res.send(data.Body);
    } catch (err) {
        console.error(`Download error for ${filename}:`, err);
        res.status(500).send(`파일 다운로드 중 오류가 발생했습니다: ${err.message}`);
    }
});

// 파일 삭제 POST 요청 ('/deleteFile')
router.post('/deleteFile', async (req, res) => {
    const filename = req.body.dlKey;
    console.log(`Deleting file: ${filename}`);

    const params = {
        Bucket: BUCKET_NAME,
        Key: filename,
    };

    try {
        await s3.deleteObject(params).promise();
        console.log(`File deleted from S3: ${filename}`);
        res.redirect('/list');
    } catch (err) {
        console.error(`Delete error for ${filename}:`, err);
        res.status(500).send(`파일 삭제 중 오류가 발생했습니다: ${err.message}`);
    }
});

// --- 파일 비교 POST 라우트 ---
router.post('/compareAudio', async (req, res) => {
    const { file1Key, file2Key } = req.body;

    if (!file1Key || !file2Key) {
        return res.status(400).render('comparison_result', { error: '비교할 두 파일을 모두 선택해주세요.', result: null });
    }

    if (file1Key === file2Key) {
        return res.status(400).render('comparison_result', { error: '동일한 파일을 선택할 수 없습니다.', result: null });
    }

    try {
        console.log(`Comparing: ${file1Key} vs ${file2Key}`);
        
        // 파일1 분석 시뮬레이션
        const analysis1 = await simulateAudioAnalysis(file1Key);
        console.log(`Analysis for ${file1Key}:`, analysis1);

        // 파일2 분석 시뮬레이션
        const analysis2 = await simulateAudioAnalysis(file2Key);
        console.log(`Analysis for ${file2Key}:`, analysis2);

        // 비교 결과 계산
        const comparisonResult = {
            file1: { key: file1Key, ...analysis1 },
            file2: { key: file2Key, ...analysis2 },
            speechRateChange: analysis2.speechRate - analysis1.speechRate,
            fillerWordsChange: analysis2.fillerWordsCount - analysis1.fillerWordsCount
        };

        // comparison_result.ejs 템플릿 렌더링
        res.render('comparison_result', { result: comparisonResult, error: null });

    } catch (error) {
        console.error("오디오 파일 비교 중 오류 발생:", error);
        res.status(500).render('comparison_result', { error: `오디오 파일 비교 중 오류가 발생했습니다: ${error.message}`, result: null });
    }
});

// POST /transcribe
router.post('/transcribe', async (req, res) => {
    const audioFilePath = path.join(__dirname, '../uploadedFiles', req.body.filename);

    if (!fs.existsSync(audioFilePath)) {
        return res.status(404).json({ error: 'Audio file not found' });
    }

    const python = spawn('python3', ['transcribe.py', audioFilePath]);

    let result = '';
    let error = '';

    python.stdout.on('data', (data) => {
        result += data.toString();
    });

    python.stderr.on('data', (data) => {
        error += data.toString();
    });

    python.on('close', (code) => {
        if (code !== 0) {
            return res.status(500).json({ error: 'Python error', details: error });
        }
        try {
            const parsed = JSON.parse(result);
            res.json(parsed);
        } catch (err) {
            res.status(500).json({ error: 'Failed to parse Python output', raw: result });
        }
    });
});

module.exports = router; // router 인스턴스를 내보내기
