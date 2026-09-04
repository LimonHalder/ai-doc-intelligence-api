import json

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import StreamingResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.deps import get_current_user
from app.models import Document, DocumentStatus, User
from app.rate_limit import limiter
from app.routers.documents import _get_owned_document
from app.schemas import (
    AskRequest,
    ExtractRequest,
    ExtractResponse,
    SourceChunk,
    SummarizeRequest,
    SummarizeResponse,
)
from app.services import embeddings, llm, vectorstore

router = APIRouter(prefix="/chat", tags=["chat"])

RAG_SYSTEM_PROMPT = (
    "You are a precise document assistant. Answer the user's question using ONLY the "
    "provided context excerpts from their document. If the answer isn't in the context, "
    "say you don't have enough information — do not make anything up. Keep answers focused."
)

SUMMARY_SYSTEM_PROMPT = "You are an expert summarizer. Produce accurate, faithful summaries with no invented facts."

EXTRACT_SYSTEM_PROMPT = (
    "You extract structured data from documents. Respond with ONLY a valid JSON object "
    "matching the requested fields — no markdown fences, no commentary. If a field can't "
    "be found in the text, set its value to null."
)


async def _ready_document(db: AsyncSession, document_id: str, owner_id: str) -> Document:
    document = await _get_owned_document(db, document_id, owner_id)
    if document.status != DocumentStatus.READY:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=f"Document is not ready for queries (status: {document.status.value}).",
        )
    return document


@router.post("/{document_id}/ask")
@limiter.limit(settings.RATE_LIMIT_CHAT)
async def ask_document(
    request: Request,
    document_id: str,
    payload: AskRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """RAG Q&A with a server-sent-events streaming answer.

    Event stream format:
      event: sources  -> JSON array of retrieved chunks (sent first)
      event: delta     -> a text fragment of the answer (sent repeatedly)
      event: done       -> stream finished
      event: error      -> something went wrong
    """
    document = await _ready_document(db, document_id, current_user.id)

    query_embedding = embeddings.embed_query(payload.question)
    results = vectorstore.query(
        document_id=document.id,
        owner_id=current_user.id,
        query_embedding=query_embedding,
        top_k=payload.top_k or settings.TOP_K_RETRIEVAL,
    )

    if not results:
        async def empty_stream():
            yield _sse("sources", [])
            yield _sse(
                "delta",
                "I couldn't find any relevant content in this document to answer that.",
            )
            yield _sse("done", {})

        return StreamingResponse(empty_stream(), media_type="text/event-stream")

    context = "\n\n---\n\n".join(f"[Excerpt {i+1}]\n{r['text']}" for i, r in enumerate(results))
    prompt = f"Context excerpts from the document:\n\n{context}\n\nQuestion: {payload.question}"

    sources = [
        SourceChunk(chunk_index=r["chunk_index"], text=r["text"], score=round(r["score"], 4)).model_dump()
        for r in results
    ]

    async def event_stream():
        yield _sse("sources", sources)
        try:
            async for delta in llm.stream(RAG_SYSTEM_PROMPT, prompt):
                yield _sse("delta", delta)
            yield _sse("done", {})
        except Exception as exc:  # noqa: BLE001
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@router.post("/{document_id}/summarize", response_model=SummarizeResponse)
@limiter.limit(settings.RATE_LIMIT_CHAT)
async def summarize_document(
    request: Request,
    document_id: str,
    payload: SummarizeRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    document = await _ready_document(db, document_id, current_user.id)
    chunks = vectorstore.get_all_chunks(document.id, current_user.id)
    full_text = "\n\n".join(chunks)

    style_instructions = {
        "concise": "Write a concise summary in 3-5 sentences.",
        "detailed": "Write a detailed, thorough summary covering all major points.",
        "bullet_points": "Write a summary as a bulleted list of key points.",
    }.get(payload.style, "Write a concise summary in 3-5 sentences.")

    prompt = f"{style_instructions}\n\nDocument:\n\n{full_text}"
    summary = await llm.complete(SUMMARY_SYSTEM_PROMPT, prompt)
    return SummarizeResponse(summary=summary)


@router.post("/{document_id}/extract", response_model=ExtractResponse)
@limiter.limit(settings.RATE_LIMIT_CHAT)
async def extract_structured_data(
    request: Request,
    document_id: str,
    payload: ExtractRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    if not payload.fields:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No fields requested.")

    document = await _ready_document(db, document_id, current_user.id)
    chunks = vectorstore.get_all_chunks(document.id, current_user.id)
    full_text = "\n\n".join(chunks)

    field_lines = "\n".join(f'- "{name}": {desc}' for name, desc in payload.fields.items())
    prompt = (
        f"Extract the following fields from the document below:\n{field_lines}\n\n"
        f"Document:\n\n{full_text}\n\n"
        f"Respond with a single JSON object with exactly these keys: "
        f"{list(payload.fields.keys())}."
    )

    raw = await llm.complete(EXTRACT_SYSTEM_PROMPT, prompt)
    data = _parse_json_response(raw)
    return ExtractResponse(data=data)


def _parse_json_response(raw: str) -> dict:
    cleaned = raw.strip()
    if cleaned.startswith("```"):
        cleaned = cleaned.strip("`")
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        return {"_raw_response": raw, "_error": "Model did not return valid JSON"}


def _sse(event: str, data) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"
