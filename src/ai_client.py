"""Gemini API adapter with a deterministic local fallback."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Optional


GEMINI_URL = (
    "https://generativelanguage.googleapis.com/v1beta/models/"
    "gemini-flash-latest:generateContent"
)


@dataclass
class AIResult:
    text: str
    provider: str
    error: Optional[str] = None


class GeminiClient:
    """Small wrapper around Gemini generateContent.

    The API key is read from GEMINI_API_KEY so secrets stay out of source code.
    """

    def __init__(self, api_key: Optional[str] = None, timeout_seconds: int = 20):
        self.api_key = api_key or os.getenv("GEMINI_API_KEY")
        self.timeout_seconds = timeout_seconds

    @property
    def is_configured(self) -> bool:
        return bool(self.api_key)

    def generate(self, prompt: str) -> AIResult:
        if not self.api_key:
            return AIResult(
                text="",
                provider="local_fallback",
                error="GEMINI_API_KEY is not set.",
            )

        payload = {"contents": [{"parts": [{"text": prompt}]}]}
        request = urllib.request.Request(
            GEMINI_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "X-goog-api-key": self.api_key,
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                data = json.loads(response.read().decode("utf-8"))
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            return AIResult(text="", provider="gemini", error=str(exc))

        try:
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
        except (KeyError, IndexError, TypeError) as exc:
            return AIResult(
                text="",
                provider="gemini",
                error=f"Unexpected Gemini response shape: {exc}",
            )

        return AIResult(text=text, provider="gemini")

