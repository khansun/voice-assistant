import re, os
import time
from typing import Dict, Any, List, Tuple
import logging

from fastapi import APIRouter, Depends, HTTPException, status
from app import ChatRequest, ChatResponse, ChatSession, settings

logger = logging.getLogger(__name__)
MODEL_DIR = os.environ.get("MODEL_DIR", "./models")

# --- SERVICES ---

import fasttext

class IntentService:
    def __init__(self):
        try:
            self.model = fasttext.load_model('intent_model.bin')
            logger.info("Loaded FastText intent model successfully.")
        except Exception as e:
            logger.error(f"Failed to load FastText intent model: {e}")
            self.model = None

    def detect(self, text: str) -> str:
        if not self.model:
            return "fallback"
            
        text = text.lower().strip()
        
        # Get prediction and probabilities
        labels, probs = self.model.predict(text, k=1)
        max_prob = probs[0]
        
        # Extract intent name by removing '__label__'
        intent = labels[0].replace('__label__', '')
        
        # If the model is not very confident, fallback
        if max_prob < 0.15:
            return "fallback"
            
        return intent

class SessionService:
    def __init__(self, ttl_seconds: int = 900):
        self.ttl_seconds = ttl_seconds
        self._sessions: Dict[str, ChatSession] = {}

    def get_session(self, msisdn: str) -> ChatSession:
        now = time.time()
        session = self._sessions.get(msisdn)
        if not session or now - session.updated_at > self.ttl_seconds:
            logger.info(f"Creating new session for {msisdn}")
            session = ChatSession(msisdn=msisdn)
            self._sessions[msisdn] = session
        else:
            session.updated_at = now
        return session

    def update_session(self, msisdn: str, intent: str, context: Dict[str, Any]) -> ChatSession:
        session = self.get_session(msisdn)
        session.update(intent, context)
        self._sessions[msisdn] = session
        return session

class TelcoService:
    async def balance_check(self, msisdn: str) -> Dict[str, Any]:
        return {"balance": "৫০.২৫ টাকা"}

    async def internet_offer(self, msisdn: str) -> Dict[str, Any]:
        return {
            "offer_id": "INT_1GB_5TK",
            "volume": "১ জিবি",
            "price": "৫ টাকা",
            "validity": "৫ দিন"
        }

    async def voice_offer(self, msisdn: str) -> Dict[str, Any]:
        return {
             "offer_id": "VOICE_200MIN",
             "volume": "২০০ মিনিট",
            "price": "৯৯ টাকা",
            "validity": "৭ দিন"
        }

    async def vas_service(self, msisdn: str) -> Dict[str, Any]:
        return {"services": ["মিসড কল অ্যালার্ট", "ওয়েলকাম টিউন", "সিআরবিটি"]}

    async def register_complain(self, msisdn: str) -> Dict[str, Any]:
        return {"ticket_id": "CMP123456", "status": "নিবন্ধিত"}

    async def buy_package(self, msisdn: str, offer_id: str) -> Dict[str, Any]:
        return {"status": "সফল", "offer_id": offer_id, "transaction_id": "TXN_987654321"}


# Dependencies Instances
_intent_service = IntentService()
_session_service = SessionService(ttl_seconds=settings.SESSION_TTL_SECONDS)
_telco_service = TelcoService()

def get_intent_service(): return _intent_service
def get_session_service(): return _session_service
def get_telco_service(): return _telco_service

# --- API ROUTER ---

router = APIRouter()

def normalize_msisdn(msisdn: str) -> str | None:
    msisdn = msisdn.strip().replace("+88", "").replace(" ", "")
    if msisdn.startswith("88"): msisdn = msisdn[2:]
    if len(msisdn) == 11 and msisdn.startswith("01"): return msisdn
    return None

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(
    req: ChatRequest,
    intent_service: IntentService = Depends(get_intent_service),
    session_service: SessionService = Depends(get_session_service),
    telco_service: TelcoService = Depends(get_telco_service)
):
    msisdn = normalize_msisdn(req.msisdn)
    if not msisdn: raise HTTPException(status_code=400, detail="Invalid MSISDN")
    if not req.text.strip(): raise HTTPException(status_code=400, detail="Text is required")

    session = session_service.get_session(msisdn)
    intent = intent_service.detect(req.text)
    
    reply = ""
    data = {}
    logger.info(f"request {req.text} intent {intent}")
    try:
        if intent == "greeting":
            reply = "আমি আপনাকে কীভাবে সাহায্য করতে পারি বলুন"
        elif intent == "balance_check":
            data = await telco_service.balance_check(msisdn)
            reply = f"আপনার বর্তমান ব্যালেন্স {data['balance']} টাকা।"
        elif intent == "internet_offer":
            data = await telco_service.internet_offer(msisdn)
            session_service.update_session(msisdn, intent, {"pending_offer": data})
            reply = (f"আপনার জন্য {data['volume']} ইন্টারনেট অফার আছে। "
                     f"মূল্য {data['price']} টাকা, মেয়াদ {data['validity']}। ")
        elif intent == "internet_package_buy":
            data = await telco_service.internet_offer(msisdn)
            buy_result = await telco_service.buy_package(msisdn, data["offer_id"])
            reply = f"অভিনন্দন! আপনি সফলভাবে {data['volume']} ইন্টারনেট প্যাকেজ কিনেছেন। মেয়াদ {data['validity']}। ধন্যবাদ।"
            data = {**data, **buy_result}
        elif intent == "voice_offer":
            data = await telco_service.voice_offer(msisdn)
            reply = f"আপনার জন্য {data['minutes']} মিনিটের অফার আছে। মূল্য {data['price']} টাকা, মেয়াদ {data['validity']}।"
        elif intent == "vas_service":
            data = await telco_service.vas_service(msisdn)
            reply = "আপনার জন্য মিসড কল অ্যালার্ট, ওয়েলকাম টিউন এবং সিআরবিটি সার্ভিস আছে।"
        elif intent == "complain_registration":
            data = await telco_service.register_complain(msisdn)
            reply = f"আপনার অভিযোগ গ্রহণ করা হয়েছে। টিকেট নম্বর {data['ticket_id']}।"
        elif intent == "confirm_yes":
            pending_offer = session.context.get("pending_offer")
            if pending_offer:
                data = await telco_service.buy_package(msisdn, pending_offer["offer_id"])
                reply = f"অভিনন্দন! আপনি সফলভাবে {pending_offer['volume']} ইন্টারনেট প্যাকেজ কিনেছেন। মেয়াদ {pending_offer['validity']}। ধন্যবাদ।"
                session_service.update_session(msisdn, "buy_success", {"pending_offer": None})
            else:
                reply = "কোনো অফার পাওয়া যায়নি। আপনি কোন প্যাকেজ কিনতে চান বলুন।"
        elif intent == "confirm_no":
            reply = "আপনার অনুরোধ বাতিল করা হয়েছে। আর কিছু জানতে চাইলে বলুন।"
        else:
            reply = "দুঃখিত, আমি বুঝতে পারিনি। আপনি ব্যালেন্স, ইন্টারনেট অফার, মিনিট অফার, সার্ভিস অথবা অভিযোগ সম্পর্কে জানতে পারেন।"

        session = session_service.update_session(msisdn, intent, {})

    except Exception as e:
        logger.error(f"Error processing intent: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Internal server error")

    return ChatResponse(
        success=True,
        msisdn=msisdn,
        intent=intent,
        reply=reply,
        session_id=session.session_id,
        data=data
    )
