# transcribe.py(whisper+ wpm logic)
import sys()
import whisper
import sys
import json

def transcribe_audio(audio_path):
    model = whisper.load_model("base")  # You can use "tiny", "small", "medium", "large"
    result = model.transcribe(audio_path)
    text = result["text"]
    segments = result["segments"]  # Each segment has start and end time
    return text, segments

def calculate_wpm(text, segments):
    word_count = len(text.split())
    start_time = segments[0]["start"]
    end_time = segments[-1]["end"]
    duration_minutes = (end_time - start_time) / 60
    wpm = word_count / duration_minutes if duration_minutes > 0 else 0
    return word_count, duration_minutes, wpm

def analyze_audio(audio_file_path):
    text, segments = transcribe_audio(audio_file_path)
    word_count, duration_minutes, wpm = calculate_wpm(text, segments)

    print(f"Transcription: {text[:100]}...")  # preview first 100 chars
    print(f"Total Words: {word_count}")
    print(f"Total Duration (min): {duration_minutes:.2f}")
    print(f"Speech Speed: {wpm:.2f} WPM")

# Example
analyze_audio("your_audio_file.wav")  # or .mp3, .m4a etc.


model = whisper.load_model("base")
result = model.transcribe(sys.argv[1], language='ko')
print(json.dumps({
    "text": result["text"],
    "segments": result.get("segments"),
    "language": result.get("language")
}))
