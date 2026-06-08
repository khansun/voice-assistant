#!/usr/bin/env python3

import sys, os, json, uuid, subprocess, logging, requests
from pathlib import Path

BASE = Path("/var/lib/asterisk/agi-bin")
TMP = Path("/var/spool/asterisk/tmp/ai-agent")
LOG = "/var/log/asterisk/ai_agent.log"
BASE_URL = "http://localhost"
ASR_URL = f"{BASE_URL}:6000/api/transcribe"
TTS_URL = f"{BASE_URL}:5002/api/tts"
CHATBOT_URL = f"{BASE_URL}:9000/api/v1/chat"
LANGUAGE = "bn"

TMP.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG,
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s"
)

class AGI:
    def __init__(self):
        while sys.stdin.readline().strip():
            pass

    def cmd(self, c):
        print(c, flush=True)
        r = sys.stdin.readline().strip()
        logging.info(f"AGI CMD={c} RESP={r}")
        return r

    def answer(self):
        return self.cmd("ANSWER")

    def hangup(self):
        return self.cmd("HANGUP")

    def stream(self, f, digits=""):
        return self.cmd(f'STREAM FILE "{f}" "{digits}"')

    def record(self, f):
        # 12000 = 12 seconds max record time. Adjust if you want longer inputs.
        return self.cmd(f'RECORD FILE "{f}" wav "#" 12000 0 BEEP s=3')
    
    def get_variable(self, varname):
        resp = self.cmd(f"GET VARIABLE {varname}")
        # Typical response: 200 result=1 (+8801833183781)
        if "result=" in resp:
            parts = resp.split("result=")[1].strip()
            if "(" in parts and ")" in parts:
                value = parts.split("(")[1].split(")")[0]
                # Remove leading '+' if present
                return value.lstrip("+")
            return parts
        return ""

def parse_digit(resp):
    if "result=" not in resp:
        return ""

    val = resp.split("result=")[1].split()[0]

    try:
        code = int(val)
        return chr(code) if code > 0 else ""
    except Exception:
        return ""

def run(cmd):
    logging.info("RUN: " + " ".join(cmd))
    r = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    if r.returncode != 0:
        raise RuntimeError(r.stderr)

    return r

def tts(text, out8k, lang):
    logging.info(f"TTS_LANG={lang} TTS_TEXT={text}")

    resp = requests.get(
    TTS_URL,
    params={"text": text, "lang": lang},
    timeout=60
    )
    resp.raise_for_status()

    raw_tmp = Path(str(out8k) + ".raw")
    with open(raw_tmp, "wb") as f:
        f.write(resp.content)

    try:
        ffmpeg_cmd = [
            "ffmpeg", "-y",
            "-i", str(raw_tmp),
            "-ar", "8000",
            "-ac", "1",
            "-acodec", "pcm_s16le",
            str(out8k)
        ]
        run(ffmpeg_cmd)
    finally:
        if raw_tmp.exists():
            os.remove(raw_tmp)

    return out8k


def call_asr_api(audio_file):
    logging.info(f"CALL_ASR_API file={audio_file}")
    with open(audio_file, "rb") as f:
        files = {
            "audio": (
                os.path.basename(audio_file),
                f,
                "audio/wav"
            )
        }

        headers = {
            "accept": "application/json"
        }

        resp = requests.post(
            ASR_URL,
            headers=headers,
            files=files,
            timeout=60
        )

    resp.raise_for_status()
    data = resp.json()
    return data.get("text", "").strip()


def record_and_asr(agi, call_id, name):
    rec_base = TMP / f"{call_id}_{name}"
    rec_wav = TMP / f"{call_id}_{name}.wav"

    agi.record(str(rec_base))

    if not rec_wav.exists() or rec_wav.stat().st_size < 5:
        return ""

    return call_asr_api(rec_wav)


def cleanup_session_files(call_id):
    """Deletes all temporary wav and raw files associated with this specific call_id"""
    logging.info(f"CLEANUP: Removing temporary files for {call_id}")
    for file_path in TMP.glob(f"{call_id}_*"):
        try:
            file_path.unlink()
        except Exception as e:
            logging.error(f"Failed to delete {file_path}: {e}")


def getChatBotResponse(prompt: str, session_id: str) -> str:
    logging.info(f"getChatBotResponse prompt={prompt}, {session_id}")

    headers = {
        "accept": "application/json",
        "content-type": "application/json"
    }

    # Pass the session_id so your backend can track conversation history
    data = {
        "text": prompt,
        "msisdn": session_id
    }

    resp = requests.post(
        CHATBOT_URL,
        headers=headers,
        json=data,
        timeout=600
    )

    resp.raise_for_status()
    result = resp.json()
    return result.get("reply", "").strip()


def main():

    agi = AGI()
    call_id = str(uuid.uuid4())[:8]

    try:
        call_id = agi.get_variable("CALLERID(num)")
        logging.info(f"CALLER_ID={call_id}")
    except Exception:
        logging.exception("Failed to get CALLERID(num)")

    logging.info(f"{call_id} STARTED")

    try:
        agi.answer()
        
        # 1. Initial Greeting
        first_prompt = TMP / f"{call_id}_turn_0_reply.wav"
        tts(getChatBotResponse("Hello", call_id), first_prompt, LANGUAGE)
        agi.stream(str(first_prompt.with_suffix("")))

        silence_strikes = 0
        turn_counter = 1

        # 2. The Conversation Loop
        while True:
            # Dynamically name the input file so we don't trip over Asterisk caching
            text = record_and_asr(agi, call_id, f"input_turn_{turn_counter}")

            if not text:
                silence_strikes += 1
                if silence_strikes >= 2:
                    # Hang up after 2 consecutive silent turns
                    msg = "আমি কিছুই শুনিনি। বিদায়।"
                    out = TMP / f"{call_id}_goodbye.wav"
                    tts(msg, out, LANGUAGE)
                    agi.stream(str(out.with_suffix("")))
                    break
                else:
                    # Prompt the user if they are still there
                    msg = "আমি ঠিকভাবে বুঝতে পারিনি। দয়া করে আবার বলবেন কি?"
                    out = TMP / f"{call_id}_silence_{turn_counter}.wav"
                    tts(msg, out, LANGUAGE)
                    agi.stream(str(out.with_suffix("")))
                    turn_counter += 1
                    continue

            # Reset strikes if we heard something
            silence_strikes = 0
            logging.info(f"{call_id} STT_TEXT={text}")

            # 3. LLM Processing (Passing call_id for state management)
            reply = getChatBotResponse(text, call_id)
            logging.info(f"{call_id} REPLY_TEXT={reply}")

            # 4. Agent Reply
            out = TMP / f"{call_id}_turn_{turn_counter}_reply.wav"
            tts(reply, out, LANGUAGE)
            agi.stream(str(out.with_suffix("")))
            
            turn_counter += 1

    # Catch Asterisk hangup signals gracefully
    except (BrokenPipeError, EOFError):
        logging.info(f"{call_id} Caller hung up.")
    except Exception as e:
        logging.exception(f"{call_id} ERROR={e}")
        try:
            err = TMP / f"{call_id}_error.wav"
            tts("আমি ঠিক বুঝতে পারিনি, বিদায়", err, LANGUAGE)
            agi.stream(str(err.with_suffix("")))
        except Exception:
            pass

    finally:
            # 1. Attempt to tell Asterisk to hang up, but catch the error if the pipe is already dead
            try:
                agi.hangup()
            except (BrokenPipeError, EOFError):
                logging.info(f"{call_id} Pipe already closed by Asterisk.")
            except Exception as e:
                logging.error(f"{call_id} Unexpected error during hangup: {e}")

            # 2. This is now guaranteed to run, preventing disk space leaks
            cleanup_session_files(call_id)
            logging.info(f"{call_id} COMPLETED")


if __name__ == "__main__":
    main()
