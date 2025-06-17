const mongoose = require('mongoose');

const transcriptionSchema = new mongoose.Schema({
    fileKey: {
        type: String,
        required: true,
        unique: true
    },
    fileName: {
        type: String,
        required: true
    },
    transcript: {
        type: String,
        required: true
    },
    segments: [{
        startTime: Number,
        endTime: Number,
        text: String
    }],
    analysis: {
        speechRate: {
            type: Number,
            required: true
        },
        fillerWordsCount: {
            type: Number,
            required: true
        },
        totalWords: {
            type: Number,
            required: true
        }
    },
    language: {
        type: String,
        required: true
    },
    createdAt: {
        type: Date,
        default: Date.now
    },
    updatedAt: {
        type: Date,
        default: Date.now
    }
});

// Add timestamp updates on save
transcriptionSchema.pre('save', function(next) {
    this.updatedAt = new Date();
    next();
});

const Transcription = mongoose.model('Transcription', transcriptionSchema);

module.exports = Transcription;
