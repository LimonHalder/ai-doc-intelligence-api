"""LLM provider abstraction.

Supports two backends, selected via settings.LLM_PROVIDER:
  - "anthropic": Claude via the Anthropic API (needs ANTHROPIC_API_KEY)
  - "ollama":    a local model via Ollama (zero API cost, needs Ollama running)

Both expose the same two functions:
  - complete(system, prompt) -> str
  - stream(system, prompt)   -> async generator[str]  (yields text deltas)
"""
import json
from collections.abc import AsyncGenerator

import httpx

from app.config import settings


# ---------------------------------------------------------------- Anthropic

async def _anthropic_complete(system: str, prompt: str) -> str:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    response = await client.messages.create(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=1500,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    )
    return "".join(block.text for block in response.content if block.type == "text")


async def _anthropic_stream(system: str, prompt: str) -> AsyncGenerator[str, None]:
    from anthropic import AsyncAnthropic

    client = AsyncAnthropic(api_key=settings.ANTHROPIC_API_KEY)
    async with client.messages.stream(
        model=settings.ANTHROPIC_MODEL,
        max_tokens=1500,
        system=system,
        messages=[{"role": "user", "content": prompt}],
    ) as stream:
        async for text in stream.text_stream:
            yield text


# ------------------------------------------------------------------ Ollama

async def _ollama_complete(system: str, prompt: str) -> str:
    async with httpx.AsyncClient(base_url=settings.OLLAMA_BASE_URL, timeout=120) as client:
        resp = await client.post(
            "/api/chat",
            json={
                "model": settings.OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": False,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        return data.get("message", {}).get("content", "")


async def _ollama_stream(system: str, prompt: str) -> AsyncGenerator[str, None]:
    async with httpx.AsyncClient(base_url=settings.OLLAMA_BASE_URL, timeout=120) as client:
        async with client.stream(
            "POST",
            "/api/chat",
            json={
                "model": settings.OLLAMA_MODEL,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": prompt},
                ],
                "stream": True,
            },
        ) as resp:
            resp.raise_for_status()
            async for line in resp.aiter_lines():
                if not line.strip():
                    continue
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                if content:
                    yield content


# ---------------------------------------------------------------- Dispatch

async def complete(system: str, prompt: str) -> str:
    if settings.LLM_PROVIDER == "ollama":
        return await _ollama_complete(system, prompt)
    return await _anthropic_complete(system, prompt)


async def stream(system: str, prompt: str) -> AsyncGenerator[str, None]:
    if settings.LLM_PROVIDER == "ollama":
        async for chunk in _ollama_stream(system, prompt):
            yield chunk
    else:
        async for chunk in _anthropic_stream(system, prompt):
            yield chunk
