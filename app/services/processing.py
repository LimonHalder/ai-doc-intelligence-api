import logging

from sqlalchemy import update

from app.config import settings
from app.database import AsyncSessionLocal
from app.models import Document, DocumentStatus
from app.services import embeddings, vectorstore
from app.services.chunker import chunk_text
from app.services.parser import ParsingError, extract_text

logger = logging.getLogger(__name__)


async def process_document(document_id: str, owner_id: str, file_bytes: bytes, content_type: str, filename: str) -> None:
    """Runs in a FastAPI BackgroundTask after the upload response is already sent.

    Owns its own DB session since the request-scoped session is closed by
    the time this runs.
    """
    async with AsyncSessionLocal() as db:
        try:
            await _set_status(db, document_id, DocumentStatus.PROCESSING)

            text = extract_text(file_bytes, content_type, filename)
            chunks = chunk_text(text, chunk_size=settings.CHUNK_SIZE, overlap=settings.CHUNK_OVERLAP)
            if not chunks:
                raise ParsingError("Document produced no usable text chunks.")

            vectors = embeddings.embed_texts(chunks)
            vectorstore.add_chunks(document_id, owner_id, chunks, vectors)

            await db.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(status=DocumentStatus.READY, num_chunks=len(chunks), error_message=None)
            )
            await db.commit()
            logger.info("Document %s processed: %d chunks", document_id, len(chunks))

        except Exception as exc:  # noqa: BLE001
            logger.exception("Failed to process document %s", document_id)
            await db.execute(
                update(Document)
                .where(Document.id == document_id)
                .values(status=DocumentStatus.FAILED, error_message=str(exc)[:2000])
            )
            await db.commit()


async def _set_status(db, document_id: str, status: DocumentStatus) -> None:
    await db.execute(update(Document).where(Document.id == document_id).values(status=status))
    await db.commit()
