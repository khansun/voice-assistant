import os
import uuid
import requests
import time
import shutil
import logging
import json
from typing import TypedDict
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
import gradio as gr
import uvicorn
from langgraph.graph import END, StateGraph
import re
import bangla
from bnnumerizer import numerize
from bnunicodenormalizer import Normalizer


# ==========================================
# CONFIGURATION
# ==========================================
LLM_URL = os.environ.get("LLM_URL", "http://localhost:11434/api/generate")
TTS_URL = os.environ.get("TTS_URL", "http://tts:5002/api/tts")
ASR_URL = os.environ.get("ASR_URL", "http://asr:6000/transcribe")
AUDIO_CACHE = "/app/audio_cache"
os.makedirs(AUDIO_CACHE, exist_ok=True)

logger = logging.getLogger("app")
logger.setLevel(logging.INFO)
BN_NORMALIZER=Normalizer()

# ==========================================
# LANGGRAPH STATE & NLU SETUP
# ==========================================
class AssistantState(TypedDict):
    user_input: str
    intent: str
    response: str

def nlu_node(state: dict) -> dict:
    """Uses a micro-LLM strictly to extract the intent from the user's text."""
    logger.info("=== LANGGRAPH: NLU NODE (MICRO-LLM) ===")
    
    # Enriched prompt with bilingual context for better zero-shot classification
    system_prompt = """
    You are an intent classifier for a Bengali voice assistant.
    Analyze the text and classify it into ONE of these exact intents:
    - "check_balance": Money, balance, accounts (e.g., "টাকা কত আছে", "আমার ব্যালেন্স কত", "অ্যাকাউন্টে টাকা")
    - "call_rate": Call charges, minute rates (e.g., "কল রেট কত", "মিনিটে কত কাটে", "কথা বলতে কত কাটবে")
    - "greeting": Hello, hi, greetings (e.g., "হ্যালো", "আসসালামু আলাইকুম", "কেমন আছেন")
    - "unknown": Anything else
    """
    
    try:
        payload = {
            "model": "qwen2.5:0.5b", 
            "prompt": f"{system_prompt}\nText: {state['user_input']}\nIntent:",
            "stream": False,
            "format": "json",       # Native JSON mode (no regex cleanup needed)
            "keep_alive": "15m",    # Crucial: Keeps model loaded in VRAM between calls
            "options": {
                "temperature": 0.0, 
                "num_predict": 20,  # We only need enough tokens to output {"intent": "x"}
                "top_k": 1,         # Stop it from "thinking" about alternative words
            }
        }
        
        # 15s timeout allows the first "cold boot" load to succeed. 
        # Subsequent queries will return in milliseconds.
        resp = requests.post(LLM_URL, json=payload, timeout=15)
        resp.raise_for_status()
        
        raw_response = resp.json().get("response", "{}")
        intent = json.loads(raw_response).get("intent", "unknown")
        
        # Failsafe: Ensure the model didn't hallucinate a weird intent name
        valid_intents = ["check_balance", "call_rate", "greeting", "unknown"]
        if intent not in valid_intents:
            intent = "unknown"
            
        logger.info("Extracted Intent: %s", intent)
        return {"intent": intent}
        
    except Exception as e:
        logger.exception("NLU Parsing failed, defaulting to unknown")
        return {"intent": "unknown"}
# ==========================================
# ACTION NODES (BACKEND APIs)
# ==========================================
def balance_action(state: AssistantState) -> dict:
    # TODO: Connect to your real  API here
    balance_amount = "৪৫ টাকা ৫০ পয়সা"
    return {"response": f"আপনার বর্তমান ব্যালেন্স হলো {balance_amount}।"}

def call_rate_action(state: AssistantState) -> dict:
    # TODO: Connect to your real API here
    rate = "১ টাকা ২৫ পয়সা"
    return {"response": f"আপনার বর্তমান কল রেট প্রতি মিনিটে {rate}।"}

def greeting_action(state: AssistantState) -> dict:
    return {"response": "আমি আপনাকে কীভাবে সাহায্য করতে পারি বলুন?"}

def fallback_action(state: AssistantState) -> dict:
    return {"response": "বিষয়টি আমি ঠিক বুঝতে পারিনি। দয়া করে আবার বলবেন কি?"}

# ==========================================
# GRAPH ROUTING & COMPILATION
# ==========================================
def route_intent(state: AssistantState) -> str:
    """Directs traffic based on the extracted intent."""
    intent = state.get("intent", "unknown")
    if intent in ["check_balance", "call_rate", "greeting"]:
        return intent
    return "unknown"

workflow = StateGraph(AssistantState)

# Add Nodes
workflow.add_node("nlu", nlu_node)
workflow.add_node("check_balance", balance_action)
workflow.add_node("call_rate", call_rate_action)
workflow.add_node("greeting", greeting_action)
workflow.add_node("unknown", fallback_action)

# Set Graph Entry Point
workflow.set_entry_point("nlu")

# Add Edges
workflow.add_conditional_edges(
    "nlu",
    route_intent,
    {
        "check_balance": "check_balance",
        "call_rate": "call_rate",
        "greeting": "greeting",
        "unknown": "unknown"
    }
)

workflow.add_edge("check_balance", END)
workflow.add_edge("call_rate", END)
workflow.add_edge("greeting", END)
workflow.add_edge("unknown", END)

# Compile Application
assistant_app = workflow.compile()

def handle_voice_assistant_response(user_text: str) -> str:
    """Invokes the LangGraph pipeline and extracts the final text response."""
    initial_state = {"user_input": user_text, "intent": "", "response": ""}
    final_state = assistant_app.invoke(initial_state)
    return final_state["response"]

# ==========================================
# ASR & TTS SERVICES
# ==========================================
def generate_voice(text: str) -> str:
    filename = f"{uuid.uuid4()}.wav"
    local_path = os.path.join(AUDIO_CACHE, filename)
    try:
        resp = requests.get(TTS_URL, params={"text": text, "language": "bn"}, timeout=60)
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
            resp = requests.post(ASR_URL, files={"audio": f}, timeout=60)
        resp.raise_for_status()
        return resp.json().get("text", "[silence]")
    except Exception as e:
        return f"[ASR error: {e}]"

# ==========================================
# FASTAPI ENDPOINTS
# ==========================================
app = FastAPI(title="Voice Assistant API", description="LangGraph + NLU Integrated")

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
    response = handle_voice_assistant_response(prompt)
    return {"response": response}

@app.post("/tts")
async def api_tts(text: str = Form(...)):
    try:
        normalized_text = process_text_for_tts(text)
        logger.info(f"Generating voice for text: {text} as normalized: {normalized_text}")
        wav_path = generate_voice(normalized_text)
        return FileResponse(wav_path, media_type="audio/wav", filename="output.wav")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# GRADIO INTERFACE
# ==========================================
def process_turn(audio_filepath, history):
    if history is None:
        history = []

    if not audio_filepath:
        yield history, None, "No audio recorded."
        return

    pipeline_start = time.time()

    # ---------------- ASR ----------------
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
    yield history, None, f"ASR {asr_time:.1f}s | NLU routing... | TTS ... | Total ..."

    # ---------------- NLU & ROUTING (LANGGRAPH) ----------------
    try:
        start = time.time()
        bot_response = handle_voice_assistant_response(user_text)
        llm_time = time.time() - start
        
        history[-1]["content"] = bot_response
        yield history, None, f"ASR {asr_time:.1f}s | NLU {llm_time:.1f}s | TTS ... | Total ..."
    except Exception as e:
        total_time = time.time() - pipeline_start
        yield history, None, f"ASR {asr_time:.1f}s | NLU failed | Total {total_time:.1f}s"
        return

    # ---------------- TTS ----------------
    try:
        start = time.time()
        wav_path = generate_voice(bot_response)
        tts_time = time.time() - start
    except Exception as e:
        total_time = time.time() - pipeline_start
        yield history, None, f"ASR {asr_time:.1f}s | NLU {llm_time:.1f}s | TTS failed | Total {total_time:.1f}s"
        return

    total_time = time.time() - pipeline_start
    yield history, wav_path, (
        f"ASR {asr_time:.1f}s | "
        f"NLU/Logic {llm_time:.1f}s | "
        f"TTS {tts_time:.1f}s | "
        f"Total {total_time:.1f}s"
    )

def create_gradio_app():
    with gr.Blocks(title="Voice Assistant", theme=gr.themes.Soft()) as demo:
        gr.Markdown("## Voice ChatBot System (LangGraph Integrated)")
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



def process_text_for_tts(text: str) -> list:
    """
    Cleans, normalizes, and splits Bangla text for TTS inference.
    Returns a list of processed sentences ready for the synthesizer.
    """
    if not text:
        return []

    # 1. Ensure ending punctuation
    if text[-1] not in ['।', '!', '?']:
        text += '।'

    # 2. English numbers to Bangla conversion
    if re.search(r'[0-9]', text):
        text = bangla.convert_english_digit_to_bangla_digit(text)
    
    # 3. Replace ':' in between two Bangla numbers with ' এর '
    # Note: Cleaned up the regex to strictly match Bangla digits 0-9
    pattern = r"[০-৯]:[০-৯]" 
    matches = re.findall(pattern, text)
    for m in matches:
        r = m.replace(":", " এর ")
        text = text.replace(m, r)
    
    # 4. Convert numerical digits to Bangla words
    try:
        text = numerize(text)
    except Exception:
        pass
        
    # 5. Unicode Normalization (using your existing normalize function)
    text = normalize(text)
    
    # 6. Split into individual sentences based on punctuation
    sentenceEnders = re.compile(r'[।!?]')
    sentences = sentenceEnders.split(str(text))
    
    # 7. Clean up and re-attach Dari to each chunk
    processed_sentences = []
    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        processed_sentences.append(sent + '।')
        
    return processed_sentences

def normalize(sen):
    _words = [BN_NORMALIZER(word)['normalized']  for word in sen.split()]
    return " ".join([word for word in _words if word is not None])

if __name__ == "__main__":
    gradio_app = create_gradio_app()
    app = gr.mount_gradio_app(app, gradio_app, path="/")
    uvicorn.run(app, host="0.0.0.0", port=8000)