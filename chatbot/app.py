import logging
import sys
import time
import uuid
from typing import Dict, Any, Optional

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings

# --- CONFIGURATION ---
class Settings(BaseSettings):
    APP_NAME: str = "Robi AI Chatbot API"
    VERSION: str = "1.0.0"
    ENV: str = "dev"
    SESSION_TTL_SECONDS: int = 900
    LOG_LEVEL: str = "INFO"

    class Config:
        env_file = ".env"

settings = Settings()

# --- LOGGING SETUP ---
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(settings.APP_NAME)

# --- MODELS ---
class ChatRequest(BaseModel):
    msisdn: str = Field(..., description="Mobile Subscriber ISDN Number", example="01833555543")
    text: str = Field(..., description="The message sent by the user", example="amar internet offer lagbe")

class ChatResponse(BaseModel):
    success: bool
    msisdn: str
    intent: str
    reply: str
    session_id: str
    data: Dict[str, Any] = Field(default_factory=dict)

class ChatSession(BaseModel):
    session_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    msisdn: str
    last_intent: Optional[str] = None
    context: Dict[str, Any] = Field(default_factory=dict)
    created_at: float = Field(default_factory=time.time)
    updated_at: float = Field(default_factory=time.time)

    def update(self, intent: str, new_context: Dict[str, Any]):
        self.last_intent = intent
        self.context.update(new_context)
        self.updated_at = time.time()

# --- FASTAPI APP ---
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="Production grade AI Chatbot API for Robi",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    logger.info("Application starting up...")

@app.get("/", tags=["Health"])
def health_check():
    return {
        "status": "up",
        "app": settings.APP_NAME,
        "version": settings.VERSION
    }

# Import and include routers at the end to avoid circular imports
from chat import router as chat_router
app.include_router(chat_router, prefix="/api/v1", tags=["Chat"])
