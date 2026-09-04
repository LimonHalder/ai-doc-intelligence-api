"""Persistent local Chroma vector store.

Runs embedded (no separate server process required) and persists to disk,
so it survives container restarts as long as the volume is mounted.
"""
from functools import lru_cache

import chromadb

from app.config import settings

COLLECTION_NAME = "documents"


@lru_cache(maxsize=1)
def _get_client():
    return chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)


def _get_collection():
    client = _get_client()
    return client.get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def add_chunks(
    document_id: str,
    owner_id: str,
    chunks: list[str],
    embeddings: list[list[float]],
) -> None:
    if not chunks:
        return
    collection = _get_collection()
    ids = [f"{document_id}:{i}" for i in range(len(chunks))]
    metadatas = [
        {"document_id": document_id, "owner_id": owner_id, "chunk_index": i}
        for i in range(len(chunks))
    ]
    collection.add(ids=ids, embeddings=embeddings, documents=chunks, metadatas=metadatas)


def query(
    document_id: str,
    owner_id: str,
    query_embedding: list[float],
    top_k: int = 5,
) -> list[dict]:
    collection = _get_collection()
    result = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        where={"$and": [{"document_id": document_id}, {"owner_id": owner_id}]},
    )
    if not result["ids"] or not result["ids"][0]:
        return []

    out = []
    for i in range(len(result["ids"][0])):
        distance = result["distances"][0][i]
        out.append(
            {
                "text": result["documents"][0][i],
                "chunk_index": result["metadatas"][0][i]["chunk_index"],
                "score": 1 - distance,  # cosine distance -> similarity
            }
        )
    return out


def get_all_chunks(document_id: str, owner_id: str) -> list[str]:
    """Fetch every chunk for a document, ordered, for summarization/extraction."""
    collection = _get_collection()
    result = collection.get(
        where={"$and": [{"document_id": document_id}, {"owner_id": owner_id}]},
    )
    if not result["ids"]:
        return []
    pairs = sorted(
        zip(result["metadatas"], result["documents"]), key=lambda p: p[0]["chunk_index"]
    )
    return [text for _, text in pairs]


def delete_document(document_id: str, owner_id: str) -> None:
    collection = _get_collection()
    collection.delete(where={"$and": [{"document_id": document_id}, {"owner_id": owner_id}]})
