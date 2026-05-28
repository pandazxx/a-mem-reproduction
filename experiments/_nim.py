"""
Shared NIM client layer for the A-Mem reproduction.

Uses the OpenAI SDK pointed at NVIDIA's NIM endpoint. All LLM calls go
through ``call()`` which retries indefinitely on 429 rate-limit errors
(the free NIM tier caps at ~40 req/min).

Pattern borrowed from the HippoRAG reproduction:
  https://github.com/pandazxx/hipporag-reproduction/blob/main/experiments/_nim.py
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from openai import OpenAI, RateLimitError

# ---------------------------------------------------------------------------
# Lazy singleton client
# ---------------------------------------------------------------------------

_client: OpenAI | None = None

LLM_MODEL = "meta/llama-3.1-70b-instruct"


def nim() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            base_url="https://integrate.api.nvidia.com/v1",
            api_key=os.environ["NVIDIA_API_KEY"],
        )
    return _client


# ---------------------------------------------------------------------------
# 429 retry wrapper — retries indefinitely with a fixed 5 s back-off
# ---------------------------------------------------------------------------

def call(fn, *args, **kwargs):
    while True:
        try:
            return fn(*args, **kwargs)
        except RateLimitError:
            print("    [429] rate-limited — retrying in 5 s …", flush=True)
            time.sleep(5)


# ---------------------------------------------------------------------------
# JSON parsing helper — extracts the first {...} block from raw LLM output
# ---------------------------------------------------------------------------

def parse_json(text: str) -> dict[str, Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    match = re.search(r"\{[\s\S]*\}", text)
    if match:
        return json.loads(match.group())
    raise ValueError(f"no JSON object found in LLM response:\n{text[:200]}")


# ---------------------------------------------------------------------------
# High-level chat helper
# ---------------------------------------------------------------------------

def chat(
    prompt: str,
    *,
    system: str = "You must respond with a JSON object.",
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> dict[str, Any]:
    resp = call(
        nim().chat.completions.create,
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return parse_json(resp.choices[0].message.content)


def chat_text(
    prompt: str,
    *,
    system: str,
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """Free-text chat (no JSON parsing). Used by the QA reader."""
    resp = call(
        nim().chat.completions.create,
        model=LLM_MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()
