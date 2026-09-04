# AI-Powered Document Intelligence API

A FastAPI backend that lets users upload PDF/DOCX documents and then chat with
them, summarize them, or extract structured data — powered by retrieval-augmented
generation (RAG).

## Features

- **JWT auth** — register/login, bcrypt-hashed passwords, protected routes
- **Upload → parse → chunk → embed → store**, processed via FastAPI `BackgroundTasks`
  so the upload response returns immediately instead of blocking on parsing
- **Local, free embeddings** via `sentence-transformers` (no API cost)
- **Persistent local vector store** via Chroma (embedded, no separate server needed)
- **RAG Q&A** with **SSE streaming** answers (`text/event-stream`)
- **Summarization** and **structured data extraction** (JSON) endpoints
- **Pluggable LLM backend**: Claude (Anthropic API) or a fully local, zero-cost
  model via Ollama — swap with one env var
- **Rate limiting** via `slowapi` on upload/chat endpoints
- **Postgres** for document/user metadata, **Pydantic v2** schemas throughout
- Fully **Dockerized** (`docker-compose up`)

## Architecture

```
Client
  │
  ├─ POST /auth/register, /auth/login          (JWT issued)
  │
  ├─ POST /documents/upload  ──► saved to Postgres (status=pending)
  │                          ──► BackgroundTask: parse → chunk → embed → Chroma
  │                              (status becomes ready/failed)
  │
  ├─ GET  /documents, /documents/{id}           (poll status)
  │
  └─ POST /chat/{id}/ask       ──► embed question → similarity search in Chroma
         (SSE stream)          ──► inject top-k chunks into prompt → LLM streams answer
     POST /chat/{id}/summarize ──► full-document summary
     POST /chat/{id}/extract   ──► LLM returns structured JSON per requested fields
```

**Why BackgroundTasks instead of Celery:** for a single-worker demo/portfolio
deployment this keeps the stack simple (no Redis/broker to run). The
`process_document` function in `app/services/processing.py` is written
standalone, so swapping it into a Celery task later is a small change — it
already opens its own DB session rather than relying on the request-scoped one.

## Quickstart (Docker)

```bash
cp .env.example .env
# edit .env: set ANTHROPIC_API_KEY (or switch LLM_PROVIDER=ollama for zero-cost local)

docker compose up --build
```

API docs: http://localhost:8000/docs

### Zero-cost local LLM option

```bash
docker compose --profile local-llm up --build
docker exec docai-ollama ollama pull llama3.1
# then set LLM_PROVIDER=ollama in .env and restart the api service
```

## Quickstart (local, no Docker)

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt

# Run Postgres separately, or point DATABASE_URL at any Postgres instance
cp .env.example .env   # edit DATABASE_URL to localhost, add your API key

uvicorn app.main:app --reload
```

## Example usage

```bash
# 1. Register + login
curl -X POST localhost:8000/auth/register -H "Content-Type: application/json" \
  -d '{"email":"you@example.com","password":"supersecret123"}'

TOKEN=$(curl -s -X POST localhost:8000/auth/login \
  -d "username=you@example.com&password=supersecret123" | jq -r .access_token)

# 2. Upload a document
DOC_ID=$(curl -s -X POST localhost:8000/documents/upload \
  -H "Authorization: Bearer $TOKEN" -F "file=@report.pdf" | jq -r .id)

# 3. Poll until status is "ready"
curl -s localhost:8000/documents/$DOC_ID -H "Authorization: Bearer $TOKEN"

# 4. Ask a question (streams SSE)
curl -N -X POST localhost:8000/chat/$DOC_ID/ask \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"question":"What are the key findings?"}'

# 5. Summarize
curl -X POST localhost:8000/chat/$DOC_ID/summarize \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"style":"bullet_points"}'

# 6. Extract structured data
curl -X POST localhost:8000/chat/$DOC_ID/extract \
  -H "Authorization: Bearer $TOKEN" -H "Content-Type: application/json" \
  -d '{"fields":{"total_revenue":"the total revenue figure","fiscal_year":"the fiscal year covered"}}'
```

## Project layout

```
app/
├── main.py              # FastAPI app, middleware, router wiring
├── config.py            # pydantic-settings (.env driven)
├── database.py           # async SQLAlchemy engine/session
├── models.py              # User, Document ORM models
├── schemas.py              # Pydantic request/response models
├── security.py              # password hashing + JWT
├── deps.py                    # get_db, get_current_user
├── rate_limit.py               # slowapi limiter
├── routers/
│   ├── auth.py                  # register/login
│   ├── documents.py               # upload/list/status/delete
│   └── chat.py                     # ask (SSE), summarize, extract
└── services/
    ├── parser.py                    # PDF/DOCX -> text
    ├── chunker.py                     # overlapping text chunking
    ├── embeddings.py                    # sentence-transformers (local, free)
    ├── vectorstore.py                     # Chroma persistent client
    ├── llm.py                               # Claude / Ollama provider abstraction
    └── processing.py                          # background pipeline orchestrator
```

## Notes / next steps for production hardening

- Add Alembic migrations instead of `create_all` on startup
- Move `BackgroundTasks` to Celery + Redis if you need multi-worker/retry-safe
  processing at scale
- Add per-user storage quotas and virus scanning on uploads
- Restrict CORS `allow_origins` instead of `*`
- Put the rate limiter's key function behind `X-Forwarded-For` handling if
  deployed behind a reverse proxy/load balancer
