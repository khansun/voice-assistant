import os
import uuid
import requests
import os
import time
import uuid
import wave
import shutil
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import gradio as gr
import requests
import uvicorn

LLM_URL = os.environ.get("LLM_URL", "http://localhost:11434/api/generate")
TTS_URL = os.environ.get("TTS_URL", "http://tts:5002/api/tts")
ASR_URL = os.environ.get("ASR_URL", "http://asr:6000/transcribe")
AUDIO_CACHE = "/app/audio_cache"
os.makedirs(AUDIO_CACHE, exist_ok=True)
 


def generate_voice(text: str) -> str:
    """Ask Coqui TTS server to create a WAV file, return local path."""
    filename = f"{uuid.uuid4()}.wav"
    local_path = os.path.join(AUDIO_CACHE, filename)

    try:
        resp = requests.get(
            TTS_URL,
            params={"text": text},
            timeout=60
        )
        resp.raise_for_status()

        with open(local_path, "wb") as f:
            f.write(resp.content)

        if os.path.getsize(local_path) == 0:
            raise RuntimeError("TTS API returned empty audio")

        return local_path

    except requests.exceptions.RequestException as e:
        raise RuntimeError(f"TTS API call failed: {e}")

def transcribe_audio(file_path: str) -> str:
    try:
        with open(file_path, "rb") as f:
            resp = requests.post(
                ASR_URL,
                files={"audio": f},
                timeout=60
            )
        resp.raise_for_status()
        return resp.json().get("text", "[silence]")
    except Exception as e:
        return f"[ASR error: {e}]"
    
def ask_llm(prompt: str) -> str:
    try:
        resp = requests.post(
            LLM_URL,
            json={
                "model": "llama3",
                "prompt": f"You are a helpful voice assistant for a customer. Answer the customer's question concisely: {prompt}",
                "stream": False,
                "options": {"num_predict": 128}
            },
            timeout=30
        )
        resp.raise_for_status()
        return resp.json().get("response", "").strip() or "I have no answer."
    except Exception as e:
        return f"LLM error: {str(e)}"

app = FastAPI(title="Voice Assistant API", description="Test ASR, LLM, TTS individually")

@app.post("/asr")
async def api_asr(audio: UploadFile = File(...)):
    temp_path = os.path.join(AUDIO_CACHE, f"temp_asr_{uuid.uuid4()}.wav")
    try:
        with open(temp_path, "wb") as f:
            shutil.copyfileobj(audio.file, f)
        text = transcribe_audio(temp_path)
        return {"text": text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/llm")
async def api_llm(prompt: str = Form(...)):
    response = ask_llm(prompt)
    return {"response": response}

@app.post("/tts")
async def api_tts(text: str = Form(...)):
    try:
        wav_path = generate_voice(text)
        return FileResponse(wav_path, media_type="audio/wav", filename="output.wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def process_turn(audio_filepath, history):
    if history is None:
        history = []

    if not audio_filepath:
        yield history, None, "No audio recorded."
        return

    pipeline_start = time.time()

    try:
        start = time.time()
        user_text = transcribe_audio(audio_filepath)
        asr_time = time.time() - start

        if user_text == "[silence]":
            yield history, None, f"ASR {asr_time:.1f}s | No speech detected"
            return

    except Exception as e:
        yield history, None, f"ASR failed: {str(e)}"
        return

    history.append({"role": "user", "content": user_text})
    history.append({"role": "assistant", "content": "..."})

    yield history, None, (
        f"ASR {asr_time:.1f}s | LLM ... | TTS ... | Total ..."
    )

    try:
        start = time.time()
        bot_response = ask_llm(user_text)
        llm_time = time.time() - start

        history[-1]["content"] = bot_response

        yield history, None, (
            f"ASR {asr_time:.1f}s | "
            f"LLM {llm_time:.1f}s | "
            f"TTS ... | Total ..."
        )

    except Exception as e:
        total_time = time.time() - pipeline_start
        yield history, None, (
            f"ASR {asr_time:.1f}s | "
            f"LLM failed | "
            f"Total {total_time:.1f}s"
        )
        return

    # ---------------- TTS ----------------
    try:
        start = time.time()
        wav_path = generate_voice(bot_response)
        tts_time = time.time() - start

    except Exception as e:
        total_time = time.time() - pipeline_start
        yield history, None, (
            f"ASR {asr_time:.1f}s | "
            f"LLM {llm_time:.1f}s | "
            f"TTS failed | "
            f"Total {total_time:.1f}s"
        )
        return

    total_time = time.time() - pipeline_start

    yield history, wav_path, (
        f"ASR {asr_time:.1f}s | "
        f"LLM {llm_time:.1f}s | "
        f"TTS {tts_time:.1f}s | "
        f"Total {total_time:.1f}s"
    )
def create_gradio_app():
    with gr.Blocks(title="Voice Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## Voice ChatBot System")
        with gr.Row():
            with gr.Column(scale=2):
                chatbot = gr.Chatbot(label="Conversation", height=500)
                audio_input = gr.Audio(sources=["microphone"], type="filepath", label="Record voice")
                submit_btn = gr.Button("Submit", variant="primary")
            with gr.Column(scale=1):
                audio_output = gr.Audio(label="Assistant's voice", autoplay=True)
                status = gr.Textbox(label="Status", lines=2, interactive=False)

        submit_btn.click(
            fn=process_turn,
            inputs=[audio_input, chatbot],
            outputs=[chatbot, audio_output, status]
        ).then(lambda: None, None, audio_input, queue=False)
        
    return demo


if __name__ == "__main__":
    gradio_app = create_gradio_app()
    app = gr.mount_gradio_app(app, gradio_app, path="/")
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
