"""
assembler.py

Audio assembler — concatenates rendered WAV segments into a single chapter
audio file. Inserts context-aware silence pauses between segments based on
transition type (e.g. dialogue→narration gets a longer pause than
dialogue→dialogue).

Functions:
    get_pause(type_a, type_b)         — Look up pause duration (ms) for a segment transition.
    assemble(segments_with_audio, output_path) — Concatenate segments with pauses, export WAV.
"""

import numpy as np
import io
import soundfile as sf
from pydub import AudioSegment

SAMPLE_RATE = 24000

# Pause durations in milliseconds by transition type
PAUSE_RULES = {
    ("dialogue",   "dialogue"):   300,
    ("dialogue",   "narration"):  450,
    ("dialogue",   "action"):     200,
    ("narration",  "dialogue"):   400,
    ("narration",  "narration"):  350,
    ("action",     "dialogue"):   200,
    ("thought",    "narration"):  400,
    "default":                    400,
}


def get_pause(type_a: str, type_b: str) -> int:
    return PAUSE_RULES.get((type_a, type_b), PAUSE_RULES["default"])


def assemble(segments_with_audio: list[dict], output_path: str):
    """
    segments_with_audio: list of dicts with keys:
        index, type, wav (np.ndarray)
    """
    parts = []

    # ── 500ms silence at the start of each chapter ────────────
    lead_in_samples = int(SAMPLE_RATE * 500 / 1000)
    parts.append(np.zeros(lead_in_samples, dtype=np.float32))

    for i, seg in enumerate(segments_with_audio):
        parts.append(seg["wav"])

        # Add pause between segments
        if i < len(segments_with_audio) - 1:
            next_type = segments_with_audio[i+1]["type"]
            pause_ms  = get_pause(seg["type"], next_type)
            pause_samples = int(SAMPLE_RATE * pause_ms / 1000)
            parts.append(np.zeros(pause_samples, dtype=np.float32))

    combined = np.concatenate(parts)
    pcm = (combined * 32767).clip(-32768, 32767).astype(np.int16)

    buf = io.BytesIO()
    sf.write(buf, pcm, SAMPLE_RATE, format="wav", subtype="PCM_16")
    buf.seek(0)

    audio = AudioSegment.from_wav(buf)
    audio.export(output_path, format="wav")
    print(f"Exported: {output_path}")