from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile
import shutil

app = FastAPI()

try:
    model = WhisperModel("distil-large-v3", device="cuda", compute_type="float16")
except Exception:
    model = WhisperModel("base", device="cpu", compute_type="int8")
    
@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        shutil.copyfileobj(audio.file, tmp)
        path = tmp.name

    segments, _ = model.transcribe(path, beam_size=5)
    text = " ".join([s.text for s in segments]).strip()

    return {"text": text}