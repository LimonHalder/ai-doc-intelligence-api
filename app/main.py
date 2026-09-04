import logging

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

from app.database import init_db
from app.rate_limit import limiter
from app.routers import auth, chat, documents

logging.basicConfig(level=logging.INFO)

app = FastAPI(
    title="AI-Powered Document Intelligence API",
    description=(
        "Upload PDF/DOCX documents, then chat with them, summarize them, or extract "
        "structured data using retrieval-augmented generation."
    ),
    version="1.0.0",
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # tighten this for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router)
app.include_router(documents.router)
app.include_router(chat.router)


@app.on_event("startup")
async def on_startup():
    await init_db()


@app.get("/health", tags=["health"])
async def health_check():
    return {"status": "ok"}
