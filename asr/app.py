from fastapi import FastAPI, UploadFile, File
from banglaspeech2text import Speech2Text
import tempfile
import shutil
import os

app = FastAPI()

# Initialize the BanglaSpeech2Text model
# You can pass "tiny", "base", "small", or "large". If left empty, it defaults to "large".
# It will automatically utilize your GPU if PyTorch with CUDA is installed in your environment.
try:
    print("Loading BanglaSpeech2Text model...")
    stt = Speech2Text("base") # Feel free to change this based on your VRAM capability
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")


@app.post("/api/transcribe")
async def transcribe(audio: UploadFile = File(...)):
    tmp_path = None

    try:
        # Save uploaded file to a temporary location
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            shutil.copyfileobj(audio.file, tmp)
            tmp_path = tmp.name

        # Transcribe using the dedicated package
        # The recognize() method handles format conversion and returns the text string directly
        transcription_text = stt.recognize(tmp_path)

        return {
            "text": transcription_text.strip(),
            "language": "bn",
            "model_used": "BanglaSpeech2Text"
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        # Clean up the temporary file from the system
        if tmp_path and os.path.exists(tmp_path):
            os.remove(tmp_path)