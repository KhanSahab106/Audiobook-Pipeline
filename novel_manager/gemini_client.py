"""
novel_manager/gemini_client.py

Gemini API wrapper using the new google.genai SDK.
Provides two call modes:
  - call_gemini()             standard generation
  - call_gemini_with_search() generation with Google Search grounding

Functions:
    _get_client()                     — Initialise and return a google.genai Client.
    _log_usage(usage_metadata, elapsed, tag) — Print token usage stats.
    call_gemini(prompt, label)        — Standard Gemini generation (no search).
    call_gemini_with_search(prompt, label) — Gemini generation with Google Search grounding.
    _extract_sources(response)        — Extract source URLs from grounding metadata.
"""

import os
import time
from dotenv import load_dotenv

load_dotenv()

# ── Model config ──────────────────────────────────────────────────────────────
MODEL_NAME  = "gemini-2.5-flash"
MAX_RETRIES = 3
RETRY_DELAY = 10
# ─────────────────────────────────────────────────────────────────────────────


def _get_client():
    from google import genai

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY not set. Add it to your .env file:\n"
            "  GEMINI_API_KEY=your_key_here"
        )
    return genai.Client(api_key=api_key)


def _log_usage(usage_metadata, elapsed: float, tag: str):
    if usage_metadata:
        print(f"  {tag}Done in {elapsed:.1f}s | "
              f"tokens — in: {usage_metadata.prompt_token_count} | "
              f"out: {usage_metadata.candidates_token_count}")
    else:
        print(f"  {tag}Done in {elapsed:.1f}s")


# ═══════════════════════════════════════════════════════════════════════════
#  STANDARD CALL
# ═══════════════════════════════════════════════════════════════════════════

def call_gemini(prompt: str, label: str = "") -> str:
    """
    Standard Gemini generation — no search grounding.
    Used for: arc updates, character analysis from local chapter text.
    """
    from google import genai

    client = _get_client()
    tag    = f"[{label}] " if label else ""

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  {tag}Sending to Gemini ({MODEL_NAME})...")
            t0       = time.perf_counter()
            response = client.models.generate_content(
                model   = MODEL_NAME,
                contents= prompt,
            )
            elapsed = time.perf_counter() - t0

            _log_usage(getattr(response, "usage_metadata", None), elapsed, tag)
            return response.text.strip()

        except Exception as e:
            print(f"  {tag}Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"  {tag}Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Gemini call [{label}] failed after {MAX_RETRIES} retries: {e}"
                )


# ═══════════════════════════════════════════════════════════════════════════
#  SEARCH-GROUNDED CALL
# ═══════════════════════════════════════════════════════════════════════════

def call_gemini_with_search(prompt: str, label: str = "") -> tuple[str, list[str]]:
    """
    Gemini generation with Google Search grounding enabled.
    Uses the new google_search tool (not the deprecated google_search_retrieval).

    Returns:
        (response_text, sources)
        sources — list of URLs Gemini used (may be empty)
    """
    from google import genai
    from google.genai import types

    client = _get_client()
    tag    = f"[{label}] " if label else ""

    # New SDK search tool format
    search_tool = types.Tool(google_search=types.GoogleSearch())

    config = types.GenerateContentConfig(
        tools=[search_tool],
    )

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"  {tag}Sending to Gemini ({MODEL_NAME}) with Google Search...")
            t0       = time.perf_counter()
            response = client.models.generate_content(
                model   = MODEL_NAME,
                contents= prompt,
                config  = config,
            )
            elapsed = time.perf_counter() - t0

            _log_usage(getattr(response, "usage_metadata", None), elapsed, tag)

            text    = response.text.strip()
            sources = _extract_sources(response)

            if sources:
                print(f"  {tag}Sources used: {len(sources)}")
                for src in sources[:5]:
                    print(f"    • {src}")
                if len(sources) > 5:
                    print(f"    ... and {len(sources) - 5} more")

            return text, sources

        except Exception as e:
            print(f"  {tag}Attempt {attempt}/{MAX_RETRIES} failed: {e}")
            if attempt < MAX_RETRIES:
                wait = RETRY_DELAY * attempt
                print(f"  {tag}Retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise RuntimeError(
                    f"Gemini search call [{label}] failed after {MAX_RETRIES} retries: {e}"
                )


def _extract_sources(response) -> list[str]:
    """Extract source URLs from Gemini's grounding metadata."""
    sources = []
    try:
        for candidate in response.candidates:
            metadata = getattr(candidate, "grounding_metadata", None)
            if not metadata:
                continue
            chunks = getattr(metadata, "grounding_chunks", [])
            for chunk in chunks:
                web = getattr(chunk, "web", None)
                if web:
                    uri = getattr(web, "uri", None)
                    if uri:
                        sources.append(uri)
    except Exception:
        pass
    return sources