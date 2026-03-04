"""
renderer.py

TTS rendering engine — converts text segments into audio using XTTS v2.
Handles model loading, text cleaning, sentence-level splitting for the
XTTS character limit, tone-based inference parameter adjustments (speed,
temperature, length_penalty, repetition_penalty, top_k, top_p), and
trailing noise trimming to remove hallucinated audio.

16 tone profiles shape how each segment sounds — from whisper to
triumphant — by tuning XTTS inference parameters per tone.

Functions:
    clean_text_for_tts(text)          — Normalise stutters, encoding, punctuation for TTS.
    load_tts()                        — Load XTTS v2 model and config; return (model, config).
    _split_for_xtts(text)             — Split text into chunks under the XTTS char limit.
    _trim_trailing_noise(wav, text, sr) — Trim hallucinated trailing audio via duration cap + energy fade.
    _infer(model, config, text, speaker_name, temperature, length_penalty,
           repetition_penalty, top_k, top_p) — Raw XTTS inference for a single text chunk.
    render_segment(model, config, text, speaker, tone)
                                      — Render a full segment with tone-driven parameter adjustments.
"""

import os
import re
import numpy as np
import torch
import scipy.signal as sps
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

MODEL_DIR   = r"C:\Users\ABRAR\AppData\Local\tts\tts_models--multilingual--multi-dataset--xtts_v2"
SAMPLE_RATE = 24000
LANGUAGE    = "en"

# XTTS v2 hard internal limit is ~250 chars.
# We split at 200 to stay comfortably under it and avoid the warning entirely.
XTTS_CHAR_LIMIT = 200

# Silence gap between sub-segments of the same logical segment (ms)
SUBSEG_GAP_MS = 80

# ─── Tone Profiles ────────────────────────────────────────────────────────
# VOICE IDENTITY RULE: All XTTS inference parameters (temperature,
# length_penalty, repetition_penalty, top_k, top_p) are IDENTICAL across
# every tone. Only speed varies — it's a post-hoc time-stretch applied
# AFTER the model generates audio, so it cannot alter voice character.
#
# This ensures a character ALWAYS sounds like the same person,
# regardless of which tone the parser assigns.
# ──────────────────────────────────────────────────────────────────────────
TONE_PROFILES = {
    "neutral":     {"speed": 1.1, "temperature": 0.65, "length_penalty": 1.0, "repetition_penalty": 3.0, "top_k": 50, "top_p": 0.85},
    "calm":        {"speed": 1.1, "temperature": 0.45, "length_penalty": 1.0, "repetition_penalty": 2.0, "top_k": 50, "top_p": 0.80},
    "tense":       {"speed": 1.1, "temperature": 0.75, "length_penalty": 1.0, "repetition_penalty": 4.5, "top_k": 50, "top_p": 0.92},
    "whisper":     {"speed": 1.1, "temperature": 0.35, "length_penalty": 1.0, "repetition_penalty": 2.0, "top_k": 50, "top_p": 0.80},
    "angry":       {"speed": 1.1, "temperature": 0.85, "length_penalty": 1.0, "repetition_penalty": 5.0, "top_k": 50, "top_p": 0.95},
    "sad":         {"speed": 1.1, "temperature": 0.40, "length_penalty": 1.0, "repetition_penalty": 2.5, "top_k": 50, "top_p": 0.82},
    "excited":     {"speed": 1.1, "temperature": 0.80, "length_penalty": 1.0, "repetition_penalty": 5.0, "top_k": 50, "top_p": 0.95},
    "cold":        {"speed": 1.1, "temperature": 0.30, "length_penalty": 1.0, "repetition_penalty": 2.0, "top_k": 50, "top_p": 0.80},
    "fearful":     {"speed": 1.1, "temperature": 0.75, "length_penalty": 1.0, "repetition_penalty": 4.5, "top_k": 50, "top_p": 0.92},
    "sarcastic":   {"speed": 1.1, "temperature": 0.70, "length_penalty": 1.0, "repetition_penalty": 4.0, "top_k": 50, "top_p": 0.90},
    "pleading":    {"speed": 1.1, "temperature": 0.50, "length_penalty": 1.0, "repetition_penalty": 2.5, "top_k": 50, "top_p": 0.82},
    "commanding":  {"speed": 1.1, "temperature": 0.60, "length_penalty": 1.0, "repetition_penalty": 4.0, "top_k": 50, "top_p": 0.88},
    "gentle":      {"speed": 1.1, "temperature": 0.35, "length_penalty": 1.0, "repetition_penalty": 2.0, "top_k": 50, "top_p": 0.80},
    "mocking":     {"speed": 1.1, "temperature": 0.75, "length_penalty": 1.0, "repetition_penalty": 4.5, "top_k": 50, "top_p": 0.92},
    "sorrowful":   {"speed": 1.1, "temperature": 0.35, "length_penalty": 1.0, "repetition_penalty": 2.5, "top_k": 50, "top_p": 0.82},
    "triumphant":  {"speed": 1.1, "temperature": 0.80, "length_penalty": 1.0, "repetition_penalty": 5.0, "top_k": 50, "top_p": 0.95},
}


def clean_text_for_tts(text: str) -> str:
    """Normalize text for natural TTS output (stutters, encoding, punctuation)."""
    # Remove square brackets — XTTS treats them as special tokens, causing
    # mispronunciation of the first word (e.g. "[The host..." → "theaah")
    text = text.replace("[", "").replace("]", "")
    # Stutters: "w-who" → "who", "s-stop" → "stop", "th-the" → "the"
    text = re.sub(r"\b([a-zA-Z]{1,3})-(?=\1)", "", text, flags=re.IGNORECASE)
    # Repeated word stutters: "I-I" → "I", "no-no" → "no"
    text = re.sub(r"\b(\w+)(?:-\1)+\b", r"\1", text, flags=re.IGNORECASE)
    text = text.encode("ascii", "ignore").decode("ascii")
    text = re.sub(r" +", " ", text)
    text = re.sub(r"([.!?,]){3,}", r"\1\1", text)
    return text.strip()


def load_tts():
    use_cuda = torch.cuda.is_available()
    print(f"Loading XTTS v2 on {'cuda' if use_cuda else 'cpu'}...")

    config = XttsConfig()
    config.load_json(os.path.join(MODEL_DIR, "config.json"))

    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=MODEL_DIR, eval=True)

    if use_cuda:
        model.cuda()

    print("Warming up...")
    gpt_cond_latent   = model.speaker_manager.speakers["Ana Florence"]["gpt_cond_latent"]
    speaker_embedding = model.speaker_manager.speakers["Ana Florence"]["speaker_embedding"]
    model.inference(
        text="Warmup.",
        language=LANGUAGE,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.7,
    )
    if use_cuda:
        torch.cuda.synchronize()

    print("TTS ready.")
    return model, config


# ═══════════════════════════════════════════════════════════════════════════
#  TEXT SPLITTING
# ═══════════════════════════════════════════════════════════════════════════

def _split_for_xtts(text: str) -> list[str]:
    """
    Split text into chunks that each stay under XTTS_CHAR_LIMIT characters.

    Strategy:
    1. Try splitting at sentence boundaries first (.  !  ?)
    2. Fall back to clause boundaries (, ;  :  —) if sentences are too long
    3. Hard-split at word boundary as last resort

    Each chunk is stripped and must be non-empty.
    """
    if len(text) <= XTTS_CHAR_LIMIT:
        return [text]

    chunks = []

    # Split at sentence endings, keeping the punctuation with the preceding chunk
    sentences = re.split(r'(?<=[.!?])\s+', text)

    current = ""
    for sent in sentences:
        if not sent.strip():
            continue

        # If a single sentence is still too long, split it at clause boundaries
        if len(sent) > XTTS_CHAR_LIMIT:
            clauses = re.split(r'(?<=[,;:\—])\s+', sent)
            for clause in clauses:
                if len(current) + len(clause) + 1 <= XTTS_CHAR_LIMIT:
                    current = (current + " " + clause).strip()
                else:
                    if current:
                        chunks.append(current)
                    # If a single clause is still over the limit, hard-split at words
                    if len(clause) > XTTS_CHAR_LIMIT:
                        words = clause.split()
                        current = ""
                        for word in words:
                            if len(current) + len(word) + 1 <= XTTS_CHAR_LIMIT:
                                current = (current + " " + word).strip()
                            else:
                                if current:
                                    chunks.append(current)
                                current = word
                    else:
                        current = clause
        else:
            if len(current) + len(sent) + 1 <= XTTS_CHAR_LIMIT:
                current = (current + " " + sent).strip()
            else:
                if current:
                    chunks.append(current)
                current = sent

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]


# ═══════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ═══════════════════════════════════════════════════════════════════════════

def _trim_trailing_noise(wav: np.ndarray, text: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Trim trailing gibberish/hallucinated audio from XTTS output.

    Uses two strategies:
    1. Duration cap — estimate max expected duration from text length
       and truncate anything beyond that.
    2. Energy fade — detect trailing silence/low-energy tail and cut it.

    XTTS hallucinations often have similar energy to real speech,
    so duration-based capping is the primary defense.
    """
    if len(wav) < sr * 0.3:
        return wav

    # ── Strategy 1: Duration cap ───────────────────────────────
    # Average English speech: ~4-5 characters per second.
    # Allow generous headroom (3 chars/sec) for slow/dramatic speech.
    word_count = len(text.split())
    char_count = len(text)
    # Use both word and char estimates, take the more generous one
    max_dur_by_chars = max(1.5, char_count / 3.0)
    max_dur_by_words = max(1.5, word_count / 1.8)  # ~108 wpm (very slow)
    max_duration = max(max_dur_by_chars, max_dur_by_words)
    max_samples  = int(max_duration * sr)

    if len(wav) > max_samples:
        # Apply a short fade-out (50ms) at the cut point
        fade_len = int(sr * 0.05)
        wav = wav[:max_samples].copy()
        if len(wav) > fade_len:
            fade = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
            wav[-fade_len:] *= fade
        return wav

    # ── Strategy 2: Trim trailing silence ──────────────────────
    # Walk backwards, find last frame above silence threshold
    window = int(sr * 0.05)  # 50ms frames
    hop    = window // 2

    energies = []
    for i in range(0, len(wav) - window, hop):
        rms = np.sqrt(np.mean(wav[i : i + window] ** 2))
        energies.append(rms)

    if not energies:
        return wav

    peak = max(energies)
    if peak < 1e-6:
        return wav

    threshold = peak * 0.02
    silence_run = 0
    min_silence = int(0.3 * sr / hop)  # 300ms sustained silence

    for i in range(len(energies) - 1, -1, -1):
        if energies[i] > threshold:
            if silence_run > min_silence:
                end_sample = min(len(wav), (i + 2) * hop + int(sr * 0.1))
                return wav[:end_sample]
            break
        silence_run += 1

    return wav


def _infer(
    model, config, text: str, speaker_name: str,
    temperature: float = 0.65,
    length_penalty: float = 1.0,
    repetition_penalty: float = 2.0,
    top_k: int = 50,
    top_p: float = 0.85,
) -> np.ndarray:
    gpt_cond_latent   = model.speaker_manager.speakers[speaker_name]["gpt_cond_latent"]
    speaker_embedding = model.speaker_manager.speakers[speaker_name]["speaker_embedding"]

    out = model.inference(
        text=text,
        language=LANGUAGE,
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=temperature,
        length_penalty=length_penalty,
        repetition_penalty=repetition_penalty,
        top_k=top_k,
        top_p=top_p,
        enable_text_splitting=False,
    )
    wav = np.array(out["wav"], dtype=np.float32)
    return _trim_trailing_noise(wav, text)


def render_segment(model, config, text: str, speaker: str, tone: str) -> np.ndarray:
    text      = clean_text_for_tts(text)
    profile   = TONE_PROFILES.get(tone, TONE_PROFILES["neutral"])
    chunks    = _split_for_xtts(text)
    gap       = np.zeros(int(SAMPLE_RATE * SUBSEG_GAP_MS / 1000), dtype=np.float32)

    if len(chunks) > 1:
        print(f"      ↳ split into {len(chunks)} sub-segments (text: {len(text)} chars)")

    parts = []
    for chunk in chunks:
        wav = _infer(
            model, config, chunk, speaker,
            temperature=profile["temperature"],
            length_penalty=profile["length_penalty"],
            repetition_penalty=profile["repetition_penalty"],
            top_k=profile["top_k"],
            top_p=profile["top_p"],
        )
        parts.append(wav)
        parts.append(gap)

    # Remove trailing gap
    if parts:
        parts.pop()

    combined = np.concatenate(parts) if len(parts) > 1 else parts[0]

    # Apply speed profile
    if profile["speed"] != 1.0:
        target_len = int(len(combined) / profile["speed"])
        combined   = sps.resample(combined, target_len).astype(np.float32)

    return combined