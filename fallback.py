"""
fallback.py

Fallback rendering engine — wraps the TTS renderer with a 5-stage retry
strategy so every segment always produces audio, even if the primary render
fails. Stages: normal render → text cleanup → sentence splitting → narrator
fallback → audible placeholder tone.

Functions:
    clean_text_for_tts(text)          — Sanitise stutters, non-ASCII, repeated punctuation.
    split_segment_text(text, max_chars) — Split text at sentence boundaries into smaller chunks.
    generate_placeholder(duration_ms)  — Create a 440 Hz tone placeholder for failed segments.
    _try_render(tts, tts_config, text, speaker, tone, label) — Attempt render with retries.
    render_with_fallback(tts, tts_config, seg, xtts_speaker, failure_log) — Full 5-stage fallback pipeline.
"""

import time
import numpy as np
import re
from renderer import render_segment

# ── Config ──────────────────────────────────────────────────────────────────
MAX_RETRIES     = 3
RETRY_DELAY     = 2       # seconds between retries
SAMPLE_RATE     = 24000
FALLBACK_SPEAKER = "Ana Florence"
# ────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
#  TEXT UTILITIES
# ═══════════════════════════════════════════════════════════════════════════

def clean_text_for_tts(text: str) -> str:
    """
    Sanitize text that may be causing TTS failures.
    - Normalize stutters (w-who → who, I-I → I)
    - Strip non-ASCII characters
    - Collapse multiple spaces
    - Remove excessive repeated punctuation
    """
    # Stutters: single letter(s) + hyphen + word starting with same letter
    # e.g. "w-who" → "who", "s-stop" → "stop", "th-the" → "the"
    text = re.sub(r"\b([a-zA-Z]{1,3})-(?=\1)", "", text, flags=re.IGNORECASE)
    # Repeated word stutters: "I-I" → "I", "no-no" → "no"
    text = re.sub(r"\b(\w+)(?:-\1)+\b", r"\1", text, flags=re.IGNORECASE)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r" +", " ", text)
    text = re.sub(r"([.!?,]){3,}", r"\1\1", text)
    return text.strip()


def split_segment_text(text: str, max_chars: int = 200) -> list[str]:
    """
    Split text at sentence boundaries into chunks under max_chars.
    Used as last-resort before placeholder insertion.
    """
    if len(text) <= max_chars:
        return [text]

    sentences = re.split(r'(?<=[.!?])\s+', text)
    parts, current = [], ""

    for sent in sentences:
        if len(current) + len(sent) + 1 <= max_chars:
            current = (current + " " + sent).strip()
        else:
            if current:
                parts.append(current)
            current = sent

    if current:
        parts.append(current)

    return parts if parts else [text]


# ═══════════════════════════════════════════════════════════════════════════
#  PLACEHOLDER GENERATOR
# ═══════════════════════════════════════════════════════════════════════════

def generate_placeholder(duration_ms: int = 2000) -> np.ndarray:
    """
    Audible placeholder for segments that failed all render strategies.
    Inserts a soft 440Hz tone at start and end so the listener hears
    a clear gap marker rather than a silent cut.
    """
    total_samples = int(SAMPLE_RATE * duration_ms / 1000)
    silence       = np.zeros(total_samples, dtype=np.float32)

    tone_samples = int(SAMPLE_RATE * 0.08)
    t    = np.linspace(0, 0.08, tone_samples)
    tone = (np.sin(2 * np.pi * 440 * t) * 0.15).astype(np.float32)

    silence[:tone_samples]  = tone
    silence[-tone_samples:] = tone

    return silence


# ═══════════════════════════════════════════════════════════════════════════
#  CORE FALLBACK ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def _try_render(
    tts,
    tts_config,
    text: str,
    speaker: str,
    tone: str,
    label: str
) -> np.ndarray | None:
    """
    Attempt to render text up to MAX_RETRIES times.
    Returns numpy array on success, None on all retries exhausted.
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            print(f"      [{label}] attempt {attempt}/{MAX_RETRIES}...")
            return render_segment(tts, tts_config, text, speaker, tone)
        except Exception as e:
            print(f"      [{label}] attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
    return None


def render_with_fallback(
    tts,
    tts_config,
    seg: dict,
    xtts_speaker: str,
    failure_log: list
) -> np.ndarray:
    """
    Render a segment with 5-stage fallback strategy.
    Always returns audio — never skips, never returns None.

    Stage 1 — Normal render
              Same text, same speaker, same tone.
              Retried MAX_RETRIES times to handle transient CUDA errors.

    Stage 2 — Cleaned text
              Strip non-ASCII, fix punctuation, retry with same speaker.
              Handles encoding issues from source text.

    Stage 3 — Fallback speaker
              Replace character voice with Ana Florence.
              Handles corrupted or missing speaker latents.

    Stage 4 — Split render
              Break text at sentence boundaries into ≤150 char parts.
              Render each part individually with fallback speaker.
              Handles segments that are too long for the model.

    Stage 5 — Audible placeholder
              All strategies exhausted. Insert tone marker + silence.
              Log segment details for manual fix after batch run.
    """
    text  = seg["text"]
    tone  = seg["tone"]
    index = seg["index"]

    # ── Stage 1: Normal render ───────────────────────────────
    print(f"    → Stage 1: normal render")
    wav = _try_render(tts, tts_config, text, xtts_speaker, tone, "normal")
    if wav is not None:
        return wav

    # ── Stage 2: Cleaned text ────────────────────────────────
    cleaned = clean_text_for_tts(text)
    if cleaned and cleaned != text:
        print(f"    → Stage 2: cleaned text")
        wav = _try_render(tts, tts_config, cleaned, xtts_speaker, tone, "cleaned")
        if wav is not None:
            failure_log.append({
                "index":    index,
                "stage":    2,
                "strategy": "cleaned_text",
                "original": text,
                "used":     cleaned,
                "speaker":  xtts_speaker
            })
            return wav

    # ── Stage 3: Fallback speaker ────────────────────────────
    if xtts_speaker != FALLBACK_SPEAKER:
        print(f"    → Stage 3: fallback speaker ({FALLBACK_SPEAKER})")
        wav = _try_render(
            tts, tts_config,
            cleaned or text,
            FALLBACK_SPEAKER,
            tone,
            "fallback_spk"
        )
        if wav is not None:
            failure_log.append({
                "index":            index,
                "stage":            3,
                "strategy":         "fallback_speaker",
                "original_speaker": xtts_speaker,
                "used_speaker":     FALLBACK_SPEAKER,
                "text":             text
            })
            return wav

    # ── Stage 4: Split render ────────────────────────────────
    parts = split_segment_text(cleaned or text, max_chars=150)
    if len(parts) > 1:
        print(f"    → Stage 4: split into {len(parts)} parts")
        part_wavs   = []
        gap_silence = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.float32)
        all_ok      = True

        for j, part in enumerate(parts):
            wav = _try_render(
                tts, tts_config,
                part,
                FALLBACK_SPEAKER,
                "neutral",
                f"part_{j+1}"
            )
            if wav is not None:
                part_wavs.append(wav)
                if j < len(parts) - 1:
                    part_wavs.append(gap_silence)
            else:
                all_ok = False
                break

        if all_ok and part_wavs:
            failure_log.append({
                "index":    index,
                "stage":    4,
                "strategy": "split_render",
                "parts":    len(parts),
                "text":     text
            })
            return np.concatenate(part_wavs)

    # ── Stage 5: Placeholder ─────────────────────────────────
    print(f"    → Stage 5: all strategies failed")
    print(f"    ✗ MANUAL FIX REQUIRED — segment {index}")
    print(f"      Text: {text[:80]}")
    failure_log.append({
        "index":    index,
        "stage":    5,
        "strategy": "placeholder_inserted",
        "text":     text,
        "speaker":  xtts_speaker,
        "action":   "MANUAL FIX REQUIRED"
    })
    return generate_placeholder(duration_ms=2000)