from fastapi import FastAPI, UploadFile, File
from faster_whisper import WhisperModel
import tempfile
import shutil
import os

app = FastAPI()

# Load best available model
try:
    model = WhisperModel(
        "distil-large-v3",
        device="cuda",
        compute_type="float16"
    )
    print("Loaded distil-large-v3 on GPU")
except Exception as e:
    print(f"GPU model load failed: {e}")

    model = WhisperModel(
        "medium",
        device="cpu",
        compute_type="int8"
    )
    print("Loaded medium on CPU")


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    tmp_path = None

    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(audio.file, tmp)
            tmp_path = tmp.name

        segments, info = model.transcribe(
            tmp_path,
            language="bn",
            task="transcribe",

            beam_size=5,
            best_of=5,

            temperature=0.0,

            vad_filter=True,

            condition_on_previous_text=True
        )

        text = " ".join(
            segment.text.strip()
            for segment in segments
        ).strip()

        return {
            "text": text,
            "language": info.language,
            "language_probability": info.language_probability
        }

    finally:
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)
