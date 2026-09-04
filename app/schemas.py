from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, EmailStr, Field

from app.models import DocumentStatus


# ---------- Auth ----------

class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    email: EmailStr
    created_at: datetime


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


# ---------- Documents ----------

class DocumentOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)
    id: str
    filename: str
    content_type: str
    status: DocumentStatus
    error_message: str | None = None
    num_chunks: int
    created_at: datetime
    updated_at: datetime


class DocumentList(BaseModel):
    documents: list[DocumentOut]


# ---------- Chat / RAG ----------

class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    top_k: int | None = Field(default=None, ge=1, le=20)


class SourceChunk(BaseModel):
    chunk_index: int
    text: str
    score: float


class AskResponse(BaseModel):
    answer: str
    sources: list[SourceChunk]


class SummarizeRequest(BaseModel):
    style: str = Field(default="concise", description="'concise', 'detailed', or 'bullet_points'")


class SummarizeResponse(BaseModel):
    summary: str


class ExtractRequest(BaseModel):
    fields: dict[str, str] = Field(
        description="Mapping of field_name -> description of what to extract, e.g. "
        '{"invoice_number": "the invoice number", "total": "the total amount due"}'
    )


class ExtractResponse(BaseModel):
    data: dict[str, Any]
