"""
inspect_parser.py

Diagnostic tool — shows the complete raw Groq response for a chapter.
Useful for debugging JSON errors, checking segment quality, and
understanding token distribution.

Functions:
    _get_client()                     — Round-robin through Groq API keys.
    call_groq_raw(text, known_characters) — Send chapter to Groq, return full inspection dict.
    parse_segments_safely(raw)        — Try to parse raw response into segments with repair.
    analyze_segments(segments, source_text) — Build per-segment and aggregate stats.
    _bar(label, value, total, width)  — Render a text-based progress bar.
    print_report(...)                 — Print formatted inspection report.
    print_raw(result)                 — Print the raw Groq response.
    save_report(chapter_file, result, segments, stats) — Save report to JSON.
    main()                            — CLI entry point.

Usage:
    python inspect_parser.py novels/shadow_slave/input/chapter_2.txt
    python inspect_parser.py input/chapter_4.txt --save
    python inspect_parser.py input/chapter_4.txt --raw-only
"""

import sys
import os
import json
import time
import re
import argparse
from dotenv import load_dotenv

load_dotenv()

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from registry import load_registry, get_known_characters

from groq import Groq

SYSTEM_PROMPT = """You are a verbatim audiobook transcriber. Your ONLY job is to label every sentence of the input text with speaker and tone metadata for TTS rendering.

CRITICAL RULE: You must include EVERY sentence from the input, without exception. Do not skip, summarize, condense, or paraphrase any content. Every sentence that exists in the input must appear in your output with its text preserved exactly word-for-word. Retain 100% of the text. Do not change any word or its position.

OUTPUT: Return ONLY a valid JSON array. No markdown, no explanation.

SCHEMA: Each segment needs exactly:
{"index": int, "speaker": str, "type": str, "tone": str, "text": str}

SPEAKERS:
- narrator: all narration, description, action beats, and attribution tags (e.g. "he said", "she thought")
- character name in lowercase_underscore for dialogue AND thoughts
- never use pronouns (he/she) as speaker names
- unknown: if speaker truly cannot be inferred

TYPES: dialogue | narration | action | thought

TONES: neutral | calm | tense | whisper | angry | sad | excited | cold | fearful | sarcastic | pleading | commanding | gentle | mocking | sorrowful | triumphant
- narration/action default: neutral
- whispered/murmured → whisper or calm
- snapped/shouted → angry
- ellipsis → tense or sad
- em-dash cut-off → tense
- trembling/shaking voice → fearful
- dry/ironic delivery → sarcastic
- begging/desperate → pleading
- orders/authority → commanding
- soft/warm/soothing → gentle
- contemptuous/sneering → mocking
- deep grief/near tears → sorrowful
- victorious/booming → triumphant
- pick least extreme tone when uncertain

RULES:
- Strip quotes from dialogue text
- PRESERVE all text exactly word-for-word — never rephrase, shorten, or merge meaning from multiple sentences
- Escape any double quotes inside text values with a backslash: \\"Blackout Day\\" not "Blackout Day"
- Merge interrupted dialogue: "I won't," she said, "do it" → one segment
- Attribution tag (e.g. "he said", "she thought") becomes SEPARATE narration segment

THOUGHTS (CRITICAL — speaker is the CHARACTER, not narrator):
When text in quotes represents a character's inner thoughts (indicated by words like
"thought", "wondered", "asked herself", "exclaimed in his heart", "said to himself"),
the speaker MUST be the CHARACTER who is thinking, NOT the narrator.

  Example input:  "Oh no, what do I do?" thought Idan, panicking.
  Correct output: segment 1 → speaker: idan, type: thought, text: "Oh no, what do I do?"
                  segment 2 → speaker: narrator, type: narration, text: "thought Idan, panicking."

  Example input:  "H-husband! How is this possible?" Arabel thought, looking at Idan.
  Correct output: segment 1 → speaker: arabel_morgan, type: thought, text: "H-husband! How is this possible?"
                  segment 2 → speaker: narrator, type: narration, text: "Arabel thought, looking at Idan."

  WRONG: speaker: narrator, type: thought ← NEVER do this for quoted thoughts

MIXED PARAGRAPHS (CRITICAL — DO NOT SKIP):
When a paragraph contains dialogue FOLLOWED or PRECEDED by narration,
you MUST create SEPARATE segments for EACH part. Example:

  Input:  "W-who are you?" Arabel spoke first, addressing Idan. Unlike him,
          who was still reeling from the shock, she had grown up in an
          influential family and was used to the pressure.

  Output: segment 1 → speaker: arabel, type: dialogue, text: "W-who are you?"
          segment 2 → speaker: narrator, type: narration, text: "Arabel spoke
          first, addressing Idan. Unlike him, who was still reeling from the
          shock, she had grown up in an influential family and was used to
          the pressure."

NEVER drop the narration around dialogue. Every sentence must appear.
"""

_API_KEYS = [
    os.environ.get(f"GROQ_API_KEY_{i}", "")
    for i in range(1, 7)
]
MODEL = "llama-3.3-70b-versatile"

_clients = [Groq(api_key=k) for k in _API_KEYS if k]
print(f"  Groq API keys loaded: {len(_clients)}")
_client_index = 0


def _get_client() -> Groq:
    global _client_index
    client = _clients[_client_index % len(_clients)]
    _client_index += 1
    return client


# ═══════════════════════════════════════════════════════════════════════════
#  RAW GROQ CALL (no validation, returns everything)
# ═══════════════════════════════════════════════════════════════════════════

def call_groq_raw(text: str, known_characters: list[str]) -> dict:
    """
    Send chapter to Groq and return a full inspection dict.
    Automatically retries with the next API key on rate limit errors.
    """
    character_list = ", ".join(known_characters) if known_characters else "none yet"

    user_message = (
        f"Parse the following text exactly as given. Include every sentence verbatim.\n\n"
        f"KNOWN CHARACTERS (use these exact name keys):\n{character_list}\n\n"
        f"TEXT:\n{text}"
    )

    last_error = None
    for attempt in range(len(_clients)):
        client  = _get_client()
        key_num = ((_client_index - 1) % len(_clients)) + 1
        print(f"  Using API key {key_num} (attempt {attempt + 1}/{len(_clients)})")

        try:
            t0 = time.perf_counter()
            response = client.chat.completions.create(
                model=MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user",   "content": user_message},
                ],
                temperature=0.1,
                max_tokens=8192,
            )
            elapsed = time.perf_counter() - t0

            raw           = response.choices[0].message.content
            finish_reason = response.choices[0].finish_reason
            usage         = response.usage

            return {
                "raw":           raw,
                "finish_reason": finish_reason,
                "elapsed_s":     round(elapsed, 2),
                "input_tokens":  usage.prompt_tokens      if usage else 0,
                "output_tokens": usage.completion_tokens  if usage else 0,
                "total_tokens":  usage.total_tokens       if usage else 0,
            }

        except Exception as e:
            if "rate_limit" in str(e).lower() or "429" in str(e):
                print(f"  ⚠ Key {key_num} rate-limited, trying next...")
                last_error = e
                continue
            raise

    raise RuntimeError(f"All {len(_clients)} API keys are rate-limited: {last_error}")


# ═══════════════════════════════════════════════════════════════════════════
#  ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def parse_segments_safely(raw: str) -> tuple[list[dict], str]:
    """
    Try to parse the raw response into segments.
    Returns (segments, status) where status is 'ok', 'repaired', or 'failed'.
    """
    cleaned = re.sub(r"```json|```", "", raw).strip()
    start   = cleaned.find("[")
    end     = cleaned.rfind("]")

    if start == -1 or end == -1:
        return [], "no_array"

    json_str = cleaned[start:end + 1]

    # Attempt 1: direct parse
    try:
        return json.loads(json_str), "ok"
    except json.JSONDecodeError:
        pass

    # Attempt 2: repair
    repaired = re.sub(r',\s*([}\]])', r'\1', json_str)
    if not repaired.rstrip().endswith("]"):
        last = repaired.rfind("}")
        if last != -1:
            repaired = repaired[:last + 1] + "\n]"
    try:
        return json.loads(repaired), "repaired"
    except json.JSONDecodeError:
        pass

    # Attempt 3: extract individual objects
    matches   = re.findall(r'\{[^{}]*\}', json_str, re.DOTALL)
    recovered = []
    for m in matches:
        try:
            obj = json.loads(m)
            if isinstance(obj, dict) and "text" in obj:
                recovered.append(obj)
        except json.JSONDecodeError:
            continue

    if recovered:
        return recovered, "partial"

    return [], "failed"


def analyze_segments(segments: list[dict], source_text: str) -> dict:
    """Build per-segment and aggregate stats."""
    speaker_counts = {}
    type_counts    = {}
    tone_counts    = {}
    word_counts    = {}
    long_segments  = []     # over 200 chars (XTTS warning zone)
    json_issues    = []     # segments with potential quote problems

    for seg in segments:
        sp = seg.get("speaker", "unknown")
        tp = seg.get("type",    "unknown")
        tn = seg.get("tone",    "unknown")
        tx = seg.get("text",    "")

        speaker_counts[sp] = speaker_counts.get(sp, 0) + 1
        type_counts[tp]    = type_counts.get(tp, 0) + 1
        tone_counts[tn]    = tone_counts.get(tn, 0) + 1
        word_counts[sp]    = word_counts.get(sp, 0) + len(tx.split())

        if len(tx) > 200:
            long_segments.append({
                "index":  seg.get("index"),
                "chars":  len(tx),
                "speaker": sp,
                "preview": tx[:80]
            })

        if '"' in tx:
            json_issues.append({
                "index":   seg.get("index"),
                "speaker": sp,
                "preview": tx[:80]
            })

    # Coverage
    src_words = set(source_text.lower().split())
    seg_words = set()
    for seg in segments:
        seg_words.update(seg.get("text", "").lower().split())
    coverage = round(len(src_words & seg_words) / len(src_words) * 100, 1) if src_words else 0

    return {
        "total_segments":  len(segments),
        "total_seg_words": sum(len(s.get("text","").split()) for s in segments),
        "source_words":    len(source_text.split()),
        "coverage_pct":    coverage,
        "speaker_counts":  dict(sorted(speaker_counts.items(), key=lambda x: -x[1])),
        "type_counts":     type_counts,
        "tone_counts":     dict(sorted(tone_counts.items(), key=lambda x: -x[1])),
        "word_counts":     dict(sorted(word_counts.items(), key=lambda x: -x[1])),
        "long_segments":   long_segments,
        "json_issues":     json_issues,
    }


# ═══════════════════════════════════════════════════════════════════════════
#  REPORT PRINTER
# ═══════════════════════════════════════════════════════════════════════════

W = 62   # report width

def _bar(label: str, value: int, total: int, width: int = 20) -> str:
    filled = int(value / total * width) if total else 0
    return f"  {label:<22} {'█' * filled}{'░' * (width - filled)}  {value}"


def print_report(
    chapter_file: str,
    source_text: str,
    result: dict,
    segments: list[dict],
    parse_status: str,
    stats: dict
):
    print(f"\n{'═' * W}")
    print(f"  GROQ INSPECTION REPORT")
    print(f"{'═' * W}")
    print(f"  File          : {os.path.basename(chapter_file)}")
    print(f"  Model         : {MODEL}")
    print(f"  Finish reason : {result['finish_reason']}"
          + (" ⚠ TRUNCATED" if result['finish_reason'] == 'length' else " ✓"))
    print(f"  Response time : {result['elapsed_s']}s")

    print(f"\n── Token Breakdown {'─' * (W - 19)}")
    print(f"  Input tokens  : {result['input_tokens']:>6,}")
    print(f"  Output tokens : {result['output_tokens']:>6,}")
    print(f"  Total tokens  : {result['total_tokens']:>6,}")

    input_breakdown_pct = result['input_tokens'] and {
        "System prompt": round(480 / result['input_tokens'] * 100),
        "Known chars":   round(80  / result['input_tokens'] * 100),
        "Chapter text":  round((result['input_tokens'] - 560) / result['input_tokens'] * 100),
    }
    if input_breakdown_pct:
        print(f"\n  Input token breakdown (estimate):")
        for k, v in input_breakdown_pct.items():
            print(f"    {k:<18}: ~{v}%")

    print(f"\n── Parse Status {'─' * (W - 16)}")
    status_icon = {"ok": "✓", "repaired": "⚠ repaired", "partial": "✗ partial", "failed": "✗ FAILED", "no_array": "✗ NO ARRAY"}
    print(f"  Status        : {status_icon.get(parse_status, parse_status)}")
    print(f"  Segments      : {stats['total_segments']}")
    print(f"  Coverage      : {stats['coverage_pct']}%  "
          f"({stats['total_seg_words']} / {stats['source_words']} words)")

    if parse_status == "repaired":
        print(f"  ⚠ JSON was malformed but successfully repaired")
    if parse_status == "partial":
        print(f"  ✗ Some segments could not be recovered — check raw output")

    print(f"\n── Speaker Distribution {'─' * (W - 24)}")
    total_segs = stats['total_segments'] or 1
    for sp, count in stats['speaker_counts'].items():
        words = stats['word_counts'].get(sp, 0)
        print(_bar(sp, count, total_segs) + f"  ({words} words)")

    print(f"\n── Type Distribution {'─' * (W - 21)}")
    for tp, count in stats['type_counts'].items():
        print(_bar(tp, count, total_segs))

    print(f"\n── Tone Distribution {'─' * (W - 21)}")
    for tn, count in stats['tone_counts'].items():
        print(_bar(tn, count, total_segs))

    if stats['long_segments']:
        print(f"\n── ⚠ Long Segments (>200 chars, XTTS will split) {'─' * (W - 50)}")
        for ls in stats['long_segments']:
            print(f"  [{ls['index']:03d}] {ls['chars']} chars | {ls['speaker']} | {ls['preview']}...")

    if stats['json_issues']:
        print(f"\n── ⚠ Segments with double quotes (JSON risk) {'─' * (W - 45)}")
        for ji in stats['json_issues']:
            print(f"  [{ji['index']:03d}] {ji['speaker']} | {ji['preview']}")

    print(f"\n── Segment Sequence (full) {'─' * (W - 27)}")
    print(f"  {'#':<5} {'Speaker':<18} {'Type':<12} {'Tone':<10} {'Words':<6}  Text preview")
    print(f"  {'─'*5} {'─'*18} {'─'*12} {'─'*10} {'─'*6}  {'─'*30}")
    for seg in segments:
        idx     = seg.get('index', '?')
        sp      = seg.get('speaker', '?')[:17]
        tp      = seg.get('type',   '?')[:11]
        tn      = seg.get('tone',   '?')[:9]
        tx      = seg.get('text',   '')
        wc      = len(tx.split())
        preview = tx[:45].replace('\n', ' ')
        print(f"  {str(idx):<5} {sp:<18} {tp:<12} {tn:<10} {wc:<6}  {preview}")

    print(f"\n{'═' * W}")


def print_raw(result: dict):
    print(f"\n{'═' * W}")
    print(f"  RAW GROQ RESPONSE")
    print(f"{'═' * W}")
    print(result['raw'])
    print(f"\n{'═' * W}")


# ═══════════════════════════════════════════════════════════════════════════
#  SAVE REPORT
# ═══════════════════════════════════════════════════════════════════════════

def save_report(chapter_file: str, result: dict, segments: list, stats: dict):
    chapter_name = os.path.splitext(os.path.basename(chapter_file))[0]
    out_path     = f"data/{chapter_name}_inspection.json"
    os.makedirs("data", exist_ok=True)

    report = {
        "chapter":       chapter_file,
        "model":         MODEL,
        "finish_reason": result['finish_reason'],
        "elapsed_s":     result['elapsed_s'],
        "tokens": {
            "input":  result['input_tokens'],
            "output": result['output_tokens'],
            "total":  result['total_tokens'],
        },
        "stats":    stats,
        "segments": segments,
        "raw":      result['raw'],
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Full report saved: {out_path}")


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Inspect the full Groq parser response for a chapter"
    )
    parser.add_argument(
        "chapter_file",
        help="Path to chapter txt file"
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save full report + raw JSON to data/{chapter}_inspection.json"
    )
    parser.add_argument(
        "--raw-only",
        action="store_true",
        help="Print only the raw Groq response, no analysis"
    )
    args = parser.parse_args()

    if not os.path.exists(args.chapter_file):
        print(f"✗ File not found: {args.chapter_file}")
        sys.exit(1)

    with open(args.chapter_file, "r", encoding="utf-8") as f:
        source_text = f.read()

    # Try to load registry for known characters
    try:
        novel_dir = os.path.dirname(os.path.dirname(args.chapter_file))
        registry  = load_registry(novel_dir)
        known     = get_known_characters(registry)
    except Exception:
        known = []

    print(f"\n  Sending {os.path.basename(args.chapter_file)} to Groq...")
    print(f"  Words: {len(source_text.split()):,}  |  Known characters: {known or 'none'}")

    result = call_groq_raw(source_text, known)

    if args.raw_only:
        print_raw(result)
        return

    segments, parse_status = parse_segments_safely(result['raw'])
    stats                  = analyze_segments(segments, source_text)

    print_report(args.chapter_file, source_text, result, segments, parse_status, stats)

    if args.save:
        save_report(args.chapter_file, result, segments, stats)


if __name__ == "__main__":
    main()
