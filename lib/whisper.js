const { spawn } = require('child_process');
const fs = require('fs').promises;
const AWS = require('aws-sdk');
const path = require('path');

// Initialize S3 client
const s3 = new AWS.S3({
    accessKeyId: process.env.ID,
    secretAccessKey: process.env.SECRET,
    region: process.env.MYREGION
});

// Run Python whisper script
async function runPythonWhisper(filePath) {
    return new Promise((resolve, reject) => {
        const python = spawn('python3.12', ['transcribe.py', filePath]);

        let data = '';
        let errorData = '';

        python.stdout.on('data', (chunk) => {
            data += chunk;
        });

        python.stderr.on('data', (chunk) => {
            errorData += chunk;
        });

        python.on('close', (code) => {
            if (code !== 0) {
                reject(new Error(`Python process exited with code ${code}\n${errorData}`));
            } else {
                try {
                    const result = JSON.parse(data);
                    resolve(result);
                } catch (err) {
                    reject(new Error(`Failed to parse Python output: ${data}`));
                }
            }
        });
    });
}

async function transcribeAudio(fileKey) {
    try {
        const params = {
            Bucket: process.env.BUCKET_NAME,
            Key: fileKey
        };

        const file = await s3.getObject(params).promise();
        const tempFilePath = path.join('./temp', fileKey);

        await fs.mkdir('./temp', { recursive: true });
        await fs.writeFile(tempFilePath, file.Body);

        const result = await runPythonWhisper(tempFilePath);

        await fs.unlink(tempFilePath);
        return result;
    } catch (error) {
        console.error('Error in transcribeAudio:', error);
        throw error;
    }
}

module.exports = {
    transcribeAudio
};
