"""
parser.py

LLM-based chapter parser — sends novel chapter text to Groq (LLaMA 3.3 70B)
and receives structured JSON segments (dialogue, narration, action, thought).
Includes coverage checking, narration injection, and JSON repair logic to
ensure 100% text retention.

Functions:
    get_token_report()                — Return cumulative token usage stats.
    reset_token_tracker()             — Zero out token counters.
    _get_client()                     — Round-robin through multiple Groq API keys.
    parse_chapter(text, known_characters) — Main entry: parse a full chapter into segments.
    _inject_missing_narration(source, segments) — Deterministically insert narration the LLM missed.
    _normalize(text)                  — Lowercase word-set for coverage comparison.
    _measure_coverage(source, segments)  — Word-level coverage ratio (0.0–1.0).
    _find_missing_paragraphs(source, segments) — Identify paragraphs with sentences not in any segment.
    _merge_repair(original, repair, source)  — Insert repair segments in source-text order.
    _build_message(text, character_list) — Build the Groq prompt message.
    _call_groq(client, user_message, label, retries) — Single Groq API call with retry logic.
    _continue_until_complete(client, original_user_message, partial_raw, label) — Resume truncated responses.
    _merge_partial(accumulated, continuation_text) — Merge continuation JSON into accumulated output.
    _track(usage, label)              — Accumulate token counts.
    _repair_json(raw)                 — Fix common JSON issues from LLM output.
    _extract_and_validate(raw, index_offset) — Parse raw JSON into validated segments.
    _extract_partial_segments(json_str) — Regex-based last-resort segment recovery.
    _validate_segments(segments, index_offset) — Sanitise and normalise segment fields.
    _normalize_probe(text)            — Build a short fingerprint for paragraph-boundary detection.
    _merge_short_segments(segments, source_text) — Merge consecutive short same-speaker segments.
"""

import json
import re
import os
import time
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

_token_tracker = {
    "total_input":  0,
    "total_output": 0,
    "total_calls":  0,
}


def get_token_report() -> dict:
    total = _token_tracker["total_input"] + _token_tracker["total_output"]
    return {
        "calls":        _token_tracker["total_calls"],
        "input_tokens": _token_tracker["total_input"],
        "output_tokens":_token_tracker["total_output"],
        "total_tokens": total,
    }


def reset_token_tracker():
    _token_tracker["total_input"]  = 0
    _token_tracker["total_output"] = 0
    _token_tracker["total_calls"]  = 0


# ── API Key Config (up to 6 keys) ────────────────────────────────────────────
_API_KEYS = [
    os.environ.get(f"GROQ_API_KEY_{i}", "")
    for i in range(1, 7)
]
MODEL = "llama-3.3-70b-versatile"

clients = [Groq(api_key=k) for k in _API_KEYS if k]
print(f"  Groq API keys loaded: {len(clients)}")

_client_index = 0


def _get_client() -> Groq:
    global _client_index
    client = clients[_client_index % len(clients)]
    _client_index += 1
    return client
# ────────────────────────────────────────────────────────────────────────────


# ── Coverage config ───────────────────────────────────────────────────────────
COVERAGE_THRESHOLD = 0.92   # trigger repair if word coverage falls below this
PARA_MIN_WORDS     = 0      # ignore paragraphs shorter than this (titles, etc.)
PARA_MISS_RATIO    = 0.50   # a paragraph is "missing" if <50% of its words are covered
# ── Merge config ──────────────────────────────────────────────────────────────
MERGE_CAP          = 400    # stop merging into a segment once it exceeds this length
# ─────────────────────────────────────────────────────────────────────────────


SYSTEM_PROMPT = """You are a verbatim audiobook transcriber. Your ONLY job is to label every sentence of the input text with speaker and tone metadata for TTS rendering.

CRITICAL RULE: You must include EVERY sentence from the input, without exception. Do not skip, summarize, condense, or paraphrase any content. Every sentence that exists in the input must appear in your output with its text preserved exactly word-for-word. Retain 100% of the text. Do not change any word or its position.

OUTPUT: Return ONLY a valid JSON array. No markdown, no explanation.

SCHEMA: Each segment needs exactly:
{"index": int, "speaker": str, "type": str, "tone": str, "text": str}

SPEAKERS:
- narrator: all narration, description, action beats, and attribution tags (e.g. "he said", "she thought")
- character name in lowercase_underscore for dialogue AND thoughts
- never use pronouns (he/she) as speaker names
- if the speaker is unknown, unnamed, or cannot be identified → use narrator

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

GROUP / SIMULTANEOUS DIALOGUE (CRITICAL):
When multiple characters speak the SAME line together (indicated by "said together",
"in unison", "chorused", "exclaimed together", or similar group attribution), produce
ONLY ONE dialogue segment using the FIRST named character as speaker. The attribution
line becomes a SEPARATE narration segment with speaker: narrator.

  Example input:  "We must leave now!" — Nemo, Eulalia and Milica said together.
  Correct output: segment 1 → speaker: nemo, type: dialogue, text: "We must leave now!"
                  segment 2 → speaker: narrator, type: narration, text: "— Nemo, Eulalia and Milica said together."

  WRONG: Creating separate dialogue segments with the same text for each speaker — NEVER do this.

NARRATION vs DIALOGUE (CRITICAL):
Any sentence WITHOUT quoted text (inside "" marks) is ALWAYS type: narration with
speaker: narrator — even if it starts with or mentions a character's name.
Only type: dialogue and type: thought may have a non-narrator speaker.

  Example input:  Eulalia joined the conversation, sharing the information she had recently received.
  Correct output: segment 1 → speaker: narrator, type: narration, text: "Eulalia joined the conversation, sharing the information she had recently received."

  WRONG: speaker: eulalia, type: narration ← NEVER assign a character as speaker for narration or action segments.
"""


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def parse_chapter(text: str, known_characters: list[str]) -> list[dict]:
    character_list = ", ".join(known_characters) if known_characters else "none yet"

    # ── Pass 1: full chapter ─────────────────────────────────────────────
    key_num = (_client_index % len(clients)) + 1
    print(f"  Sending to Groq parser (key {key_num})...")
    t0     = time.perf_counter()
    client = _get_client()
    raw    = _call_groq(client, _build_message(text, character_list), label="pass-1")

    segments = _extract_and_validate(raw, index_offset=0)
    segments = _dedup_consecutive_dialogue(segments)

    # ── Coverage check ───────────────────────────────────────────────────
    coverage = _measure_coverage(text, segments)
    print(f"  Coverage: {coverage*100:.1f}%  ({len(segments)} segments, "
          f"{sum(len(s['text'].split()) for s in segments)} / {len(text.split())} words)")

    # ── Pass 2: targeted repair (only if needed) ─────────────────────────
    if coverage < COVERAGE_THRESHOLD:
        missing_paras = _find_missing_paragraphs(text, segments)

        if missing_paras:
            print(f"  ⚠ Coverage below {COVERAGE_THRESHOLD*100:.0f}% — "
                  f"repairing {len(missing_paras)} paragraph(s)...")

            repair_text  = "\n\n".join(missing_paras)
            repair_words = len(repair_text.split())
            key_num      = (_client_index % len(clients)) + 1
            print(f"  Repair call: {repair_words} words → key {key_num}")

            repair_client = _get_client()
            repair_raw    = _call_groq(
                repair_client,
                _build_message(repair_text, character_list),
                label="repair"
            )

            repair_segs = _extract_and_validate(
                repair_raw,
                index_offset=len(segments)
            )
            repair_segs = _dedup_consecutive_dialogue(repair_segs)

            segments = _merge_repair(segments, repair_segs, text)

            new_coverage = _measure_coverage(text, segments)
            print(f"  Coverage after repair: {new_coverage*100:.1f}%  "
                  f"({len(segments)} segments)")
        else:
            print(f"  No missing paragraphs identified — skipping repair.")

    # ── Inject missing narration (deterministic) ─────────────────────────
    segments = _inject_missing_narration(text, segments)

    # ── Final pass: merge + re-index ─────────────────────────────────────
    segments = _merge_short_segments(segments, text)
    for i, seg in enumerate(segments):
        seg["index"] = i

    elapsed = time.perf_counter() - t0
    report  = get_token_report()

    print(f"  Groq total time: {elapsed:.1f}s")
    print(f"  ┌─ Token Report ───────────────────────")
    print(f"  │  Input tokens  : {report['input_tokens']}")
    print(f"  │  Output tokens : {report['output_tokens']}")
    print(f"  │  Total tokens  : {report['total_tokens']}")
    print(f"  │  API calls     : {report['calls']}")
    print(f"  └──────────────────────────────────────")

    return segments


# ═══════════════════════════════════════════════════════════════════════════
#  DETERMINISTIC NARRATION INJECTION
# ═══════════════════════════════════════════════════════════════════════════

def _inject_missing_narration(source: str, segments: list[dict]) -> list[dict]:
    """
    Walk through segments top-to-bottom. For each dialogue segment,
    find it in the source text, extract the narration that follows
    the closing quote, and check if subsequent segments captured it.
    If not, inject it right after the dialogue — no reshuffling.
    """
    result = []
    search_from = 0
    source_lower = source.lower()

    for seg_idx, seg in enumerate(segments):
        result.append(seg)

        # Only check dialogue segments
        if seg.get("type") != "dialogue":
            continue

        dialogue_text = seg["text"].strip()
        if len(dialogue_text) < 5:
            continue

        # Find this dialogue in the source (always search forward)
        probe = dialogue_text[:25].lower()
        pos = source_lower.find(probe, search_from)
        if pos == -1:
            continue

        # Find the closing quote after the dialogue content
        close_quote = source.find('"', pos + len(dialogue_text))
        if close_quote == -1:
            search_from = pos + len(probe)
            continue

        # ── Split-dialogue check ──────────────────────────────────
        # If the next segment is same-speaker dialogue, check whether
        # it's inside the SAME quoted string (LLM split a long quote)
        # or a SEPARATE quoted string (with narration between them).
        if (seg_idx + 1 < len(segments)
                and segments[seg_idx + 1].get("type") == "dialogue"
                and segments[seg_idx + 1].get("speaker") == seg.get("speaker")):
            next_probe = segments[seg_idx + 1]["text"][:25].lower()
            # Check if the next dialogue's text appears BEFORE the closing "
            between = source_lower[pos + len(dialogue_text):close_quote]
            if next_probe in between:
                # Same quoted string → skip, check after the last part
                search_from = close_quote
                continue
            # Otherwise, separate quoted strings → proceed with after-text check

        after_pos = close_quote + 1

        # Extract text until next opening quote or paragraph break
        remaining = source[after_pos:]
        next_quote = remaining.find('"')
        next_para  = remaining.find('\n\n')
        if next_quote == -1: next_quote = len(remaining)
        if next_para  == -1: next_para  = len(remaining)

        end = min(next_quote, next_para)
        after_text = remaining[:end].strip()

        # Clean leading connectors (comma, em-dash, colon, etc.)
        after_text = re.sub(r'^[\s,.\-\u2014:;]+', '', after_text).strip()

        if len(after_text.split()) < 3:
            search_from = after_pos
            continue

        # Check if ANY other segment already captured this narration
        after_clean = re.sub(r"[^\w\s]", "", after_text.lower())
        check_words = " ".join(after_clean.split()[:5])

        if len(check_words) < 8:
            search_from = after_pos
            continue

        captured = False
        for nearby in segments:
            if nearby is seg:
                continue
            nearby_clean = re.sub(r"[^\w\s]", "", nearby["text"].lower())
            if check_words in nearby_clean:
                captured = True
                break

        if not captured:
            result.append({
                "speaker": "narrator",
                "type":    "narration",
                "tone":    "neutral",
                "text":    after_text,
                "index":   0,
            })

        search_from = after_pos

    injected = len(result) - len(segments)
    if injected > 0:
        print(f"  \u26a1 Injected {injected} missing narration segment(s)")
        for i, s in enumerate(result):
            s["index"] = i

    return result


# ═══════════════════════════════════════════════════════════════════════════
#  COVERAGE ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def _normalize(text: str) -> set[str]:
    """Lowercase word set, punctuation stripped."""
    text = text.lower()
    text = re.sub(r"[^\w\s]", "", text)
    return set(text.split())


def _measure_coverage(source: str, segments: list[dict]) -> float:
    """
    What fraction of the source chapter's words appear in any segment?
    Returns 0.0–1.0.
    """
    src_words  = _normalize(source)
    seg_words  = set()
    for s in segments:
        seg_words.update(_normalize(s["text"]))

    if not src_words:
        return 1.0
    return len(src_words & seg_words) / len(src_words)


def _find_missing_paragraphs(source: str, segments: list[dict]) -> list[str]:
    """
    Return paragraphs from source that have sentences not found in any segment.
    Uses sentence-level substring matching instead of word-set overlap.
    This catches partial paragraph loss (e.g. narration after dialogue dropped).
    """
    paragraphs = [p.strip() for p in re.split(r'\n\n+', source) if p.strip()]

    # Build a single lowercase string of all segment text for substring search
    all_seg_text = " ".join(s["text"] for s in segments).lower()
    # Also strip punctuation for fuzzy matching
    all_seg_text_clean = re.sub(r"[^\w\s]", "", all_seg_text)

    missing = []
    for para in paragraphs:
        words = para.split()
        if len(words) < PARA_MIN_WORDS:
            continue

        # Split paragraph into sentences
        sentences = re.split(r'(?<=[.!?])\s+', para)

        # Count how many sentences are missing from segments
        total_sentences = 0
        missing_sentences = 0
        for sent in sentences:
            sent = sent.strip()
            if len(sent.split()) < 3:  # skip very short fragments
                continue
            total_sentences += 1

            # Try both raw and cleaned matching
            sent_lower = sent.lower()
            sent_clean = re.sub(r"[^\w\s]", "", sent_lower)

            # Check if a meaningful substring (first 40 chars) appears
            probe = sent_clean[:50].strip()
            if probe and probe not in all_seg_text_clean:
                missing_sentences += 1

        # If any sentence is missing, the paragraph needs repair
        if total_sentences > 0 and missing_sentences > 0:
            missing.append(para)

    return missing


def _merge_repair(
    original: list[dict],
    repair: list[dict],
    source: str
) -> list[dict]:
    """
    Insert repair segments into the original list in source-text order.

    Strategy: build a position map by finding where each segment's text
    first appears in the source, then sort all segments by that position.
    Segments with no match keep their current relative order at the end.
    """
    source_lower = source.lower()

    def source_position(seg: dict) -> int:
        snippet = seg["text"][:40].lower()
        snippet = re.sub(r"[^\w\s]", "", snippet).strip()
        # Try progressively shorter matches
        for length in (40, 30, 20, 10):
            probe = snippet[:length]
            if not probe:
                continue
            pos = source_lower.find(probe)
            if pos != -1:
                return pos
        return len(source_lower)  # unknown — append at end

    combined = original + repair
    combined.sort(key=source_position)

    for i, seg in enumerate(combined):
        seg["index"] = i

    return combined


# ═══════════════════════════════════════════════════════════════════════════
#  GROQ CALLS
# ═══════════════════════════════════════════════════════════════════════════

def _build_message(text: str, character_list: str) -> str:
    return (
        f"Parse the following text exactly as given. "
        f"Include every sentence verbatim.\n\n"
        f"KNOWN CHARACTERS (use these exact name keys):\n{character_list}\n\n"
        f"TEXT:\n{text}"
    )


def _parse_retry_after(error_msg: str) -> float:
    """
    Extract the retry-after delay (in seconds) from a Groq rate-limit error.

    Groq errors typically contain phrases like:
        "Please try again in 12.5s"
        "Please try again in 1m30s"
        "retry after 45.2s"
    Returns the parsed seconds, or 30.0 as a safe default.
    """
    msg = str(error_msg)

    # Match patterns like "in 1m30s", "in 45.2s", "in 2m"
    m = re.search(r'in\s+(?:(\d+)m)?(\d+(?:\.\d+)?)s', msg)
    if m:
        minutes = int(m.group(1)) if m.group(1) else 0
        seconds = float(m.group(2))
        return minutes * 60 + seconds

    # Match "in 2m" without seconds
    m = re.search(r'in\s+(\d+)m(?!\d)', msg)
    if m:
        return int(m.group(1)) * 60

    # Match bare seconds like "after 12.5s" or "after 12s"
    m = re.search(r'after\s+(\d+(?:\.\d+)?)s', msg)
    if m:
        return float(m.group(1))

    return 30.0  # safe default


def _call_groq(client: Groq, user_message: str, label: str = "", retries: int = 3) -> str:
    """
    Call Groq with smart key rotation.

    Rate-limit strategy:
        On 429, immediately try the next API key (no wait).
        Only after ALL keys have been rate-limited in one round,
        parse the retry-after time from each error and wait the
        MINIMUM of all keys' retry times.  Up to `retries` full rounds.

    Server/connection errors (502, 503, timeout, connection):
        Retry on the same key with exponential backoff.
        Up to `retries` attempts per key.
    """
    global _client_index
    num_keys = len(clients)

    for round_num in range(retries):
        # ── Try every key once per round ──────────────────────────
        keys_tried     = 0
        rate_limited   = 0
        retry_times    = []   # per-key retry-after seconds

        while keys_tried < num_keys:
            cur_key = (_client_index % num_keys) + 1

            try:
                response = client.chat.completions.create(
                    model=MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user",   "content": user_message},
                    ],
                    temperature=0.1,
                    max_tokens=8192,
                )

                _track(response.usage, label)

                raw           = response.choices[0].message.content.strip()
                finish_reason = response.choices[0].finish_reason

                if finish_reason == "length":
                    print(f"  ⚠ [{label}] Output truncated — running continuation...")
                    raw = _continue_until_complete(client, user_message, raw, label)

                return raw

            except Exception as e:
                err = str(e).lower()
                print(f"  ⚠ Groq error [{label}] key {cur_key}: {type(e).__name__}: {e}")

                if "rate_limit" in err:
                    keys_tried   += 1
                    rate_limited += 1
                    wait_time = _parse_retry_after(str(e))
                    retry_times.append(wait_time)
                    # Switch to next key immediately — no wait
                    _client_index += 1
                    client = clients[_client_index % num_keys]
                    next_key = (_client_index % num_keys) + 1
                    print(f"  Rate limit on key {cur_key} (retry in {wait_time:.0f}s) — switching to key {next_key}")

                elif "503" in err or "502" in err or "timeout" in err or "connection" in err:
                    if "503" in err:
                        err_type = "503 Service Unavailable"
                    elif "502" in err:
                        err_type = "502 Bad Gateway"
                    elif "timeout" in err:
                        err_type = "Timeout"
                    else:
                        err_type = "Connection error"
                    wait = 10 * (round_num + 1)
                    print(f"  {err_type} (key {cur_key}) — retrying in {wait}s...")
                    time.sleep(wait)
                    keys_tried += 1  # count as tried so we don't loop forever

                else:
                    raise

        # All keys exhausted this round
        if rate_limited == num_keys and retry_times:
            wait = max(1, min(retry_times))  # smallest wait across all keys
            print(f"  All {num_keys} keys rate-limited (round {round_num+1}/{retries}) "
                  f"— waiting {wait:.0f}s (shortest key cooldown)...")
            time.sleep(wait)

    raise RuntimeError(f"Groq [{label}] failed after {retries} retries.")


MAX_CONTINUATIONS = 3


def _continue_until_complete(
    client: Groq,
    original_user_message: str,
    partial_raw: str,
    label: str
) -> str:
    accumulated = partial_raw

    for pass_num in range(1, MAX_CONTINUATIONS + 1):
        print(f"  → Continuation pass {pass_num}/{MAX_CONTINUATIONS} [{label}]...")

        continuation = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system",    "content": SYSTEM_PROMPT},
                {"role": "user",      "content": original_user_message},
                {"role": "assistant", "content": accumulated},
                {"role": "user",      "content":
                    "Your response was cut off. Continue the JSON array exactly from where "
                    "you stopped. Do not repeat any segments already written. "
                    "Continue from the last complete segment and finish the array with ]."
                }
            ],
            temperature=0.1,
            max_tokens=8192,
        )

        _track(continuation.usage, f"cont-{label}-{pass_num}")

        continuation_text = continuation.choices[0].message.content.strip()
        finish_reason     = continuation.choices[0].finish_reason
        accumulated       = _merge_partial(accumulated, continuation_text)

        if finish_reason != "length":
            print(f"  ✓ Continuation complete [{label}] after {pass_num} pass(es).")
            return accumulated

    print(f"  ✗ Max continuations reached [{label}]. Force-closing array.")
    accumulated = accumulated.rstrip().rstrip(",") + "\n]"
    return "[" + accumulated.strip().lstrip("[")


def _merge_partial(accumulated: str, continuation_text: str) -> str:
    last_complete = accumulated.rfind("}")
    if last_complete == -1:
        raise ValueError("No complete JSON segment found in accumulated output.")

    clean_accumulated = accumulated[:last_complete + 1]

    first_brace = continuation_text.find("{")
    if first_brace == -1:
        clean_continuation = continuation_text.strip().lstrip(",").strip()
    else:
        clean_continuation = continuation_text[first_brace:]

    merged = clean_accumulated + ",\n" + clean_continuation

    if not merged.strip().endswith("]"):
        merged = merged.rstrip().rstrip(",") + "\n]"

    return "[" + merged.strip().lstrip("[")


def _track(usage, label: str = ""):
    if not usage:
        return
    _token_tracker["total_input"]  += usage.prompt_tokens
    _token_tracker["total_output"] += usage.completion_tokens
    _token_tracker["total_calls"]  += 1
    tag = f"  [{label}] " if label else "  "
    print(
        f"{tag}Tokens — in: {usage.prompt_tokens} | "
        f"out: {usage.completion_tokens} | "
        f"total: {usage.prompt_tokens + usage.completion_tokens}"
    )


# ═══════════════════════════════════════════════════════════════════════════
#  JSON PARSING + VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def _repair_json(raw: str) -> str:
    """
    Attempt to repair common JSON issues from LLM output:
    1. Unescaped double quotes inside string values
    2. Truncated array — force-close if missing ]
    3. Trailing comma before closing bracket
    """
    # Step 1 — fix unescaped double quotes inside "text" values.
    # Targets the text field specifically since that's where prose comes in.
    # Pattern: finds "text": "..." and re-escapes any unescaped internal quotes.
    def fix_text_field(match):
        prefix = match.group(1)   # "text": "
        value  = match.group(2)   # the content
        suffix = match.group(3)   # closing "
        # Escape any double quotes inside that aren't already escaped
        value = re.sub(r'(?<!\\)"', r'\\"', value)
        return prefix + value + suffix

    # Match "text": "..." non-greedily across the value
    raw = re.sub(
        r'("text":\s*")(.*?)("(?:\s*[,}]))',
        fix_text_field,
        raw,
        flags=re.DOTALL
    )

    # Step 2 — remove trailing comma before ] or }
    raw = re.sub(r',\s*([}\]])', r'\1', raw)

    # Step 3 — if array is not closed, close it
    if not raw.rstrip().endswith("]"):
        # Find the last complete segment (last })
        last_brace = raw.rfind("}")
        if last_brace != -1:
            raw = raw[:last_brace + 1] + "\n]"

    return raw


def _extract_and_validate(raw: str, index_offset: int = 0) -> list[dict]:
    raw   = re.sub(r"```json|```", "", raw).strip()
    start = raw.find("[")
    end   = raw.rfind("]")

    if start == -1 or end == -1:
        raise ValueError(f"Parser did not return a JSON array.\nRaw:\n{raw[:500]}")

    json_str = raw[start:end + 1]

    # First attempt — parse as-is
    try:
        segments = json.loads(json_str)
    except json.JSONDecodeError as e:
        print(f"  Warning: JSON parse failed ({e}), attempting repair...")

        # Second attempt — after repair
        repaired = _repair_json(json_str)
        try:
            segments = json.loads(repaired)
            print(f"  Repair succeeded.")
        except json.JSONDecodeError as e2:
            # Third attempt — extract only complete segments using regex
            print(f"  Repair failed ({e2}), extracting partial segments...")
            segments = _extract_partial_segments(json_str)
            if not segments:
                raise ValueError(
                    f"JSON parse failed and no segments could be recovered.\n"
                    f"Original error: {e}\nRaw:\n{raw[start:start + 500]}"
                )
            print(f"  Recovered {len(segments)} segments from partial JSON.")

    if not isinstance(segments, list):
        raise ValueError(f"Expected JSON array, got: {type(segments)}")

    return _validate_segments(segments, index_offset)


def _extract_partial_segments(json_str: str) -> list[dict]:
    """
    Last-resort recovery: use regex to extract every complete
    {...} object from a broken JSON array individually.
    Each object is parsed independently so one bad segment
    doesn't poison the rest.
    """
    # Match complete JSON objects at the top level
    pattern = re.compile(r'\{[^{}]*\}', re.DOTALL)
    matches = pattern.findall(json_str)

    recovered = []
    for i, match in enumerate(matches):
        try:
            obj = json.loads(match)
            if isinstance(obj, dict) and "text" in obj:
                recovered.append(obj)
        except json.JSONDecodeError:
            print(f"  Skipping unrecoverable segment {i}")
            continue

    return recovered


VALID_TONES   = {
    "neutral", "calm", "tense", "whisper", "angry", "sad", "excited", "cold",
    "fearful", "sarcastic", "pleading", "commanding", "gentle", "mocking",
    "sorrowful", "triumphant",
}
VALID_TYPES   = {"dialogue", "narration", "action", "thought"}
PRONOUN_NAMES = {"he", "she", "they", "it", "him", "her", "them", "his", "hers"}


def _validate_segments(segments: list, index_offset: int = 0) -> list:
    cleaned = []
    for seg in segments:
        seg.setdefault("speaker", "narrator")
        seg.setdefault("type",    "narration")
        seg.setdefault("tone",    "neutral")
        seg.setdefault("text",    "")

        if seg["speaker"].lower() in PRONOUN_NAMES:
            seg["speaker"] = "narrator"

        seg["speaker"] = seg["speaker"].lower().replace(" ", "_").strip()

        if seg["tone"] not in VALID_TONES:
            seg["tone"] = "neutral"

        if seg["type"] not in VALID_TYPES:
            seg["type"] = "narration"

        # Narration and action segments must always be spoken by the narrator.
        # Only dialogue and thought may have a character as speaker.
        if seg["type"] in ("narration", "action") and seg["speaker"] != "narrator":
            seg["speaker"] = "narrator"

        if not seg["text"].strip():
            continue

        seg["index"] = index_offset + len(cleaned)
        cleaned.append(seg)

    return cleaned


def _normalize_probe(text: str) -> str:
    """Return first ~40 chars, lowercased and punctuation-stripped, as a paragraph fingerprint."""
    cleaned = re.sub(r"[^\w\s]", "", text.lower()).strip()
    return cleaned[:40]


def _word_overlap(text_a: str, text_b: str) -> float:
    """Return what fraction of the shorter text's words appear in the other.

    Uses the size of the smaller word-set as the denominator so that a line
    that is literally repeated (possibly with trivial additions) scores near
    1.0, making it suitable for detecting near-duplicate group-dialogue lines.
    """
    words_a = set(re.sub(r"[^\w\s]", "", text_a.lower()).split())
    words_b = set(re.sub(r"[^\w\s]", "", text_b.lower()).split())
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / min(len(words_a), len(words_b))


def _dedup_consecutive_dialogue(segments: list[dict]) -> list[dict]:
    """
    Collapse runs of consecutive dialogue segments that have identical or
    near-identical text (≥95 % word overlap) into a single segment, keeping
    the first speaker.  This eliminates duplicated audio caused by the LLM
    emitting one dialogue segment per speaker when a group speaks in unison.
    """
    if not segments:
        return segments

    result: list[dict] = []
    for seg in segments:
        if (
            seg.get("type") == "dialogue"
            and result
            and result[-1].get("type") == "dialogue"
            and _word_overlap(result[-1]["text"], seg["text"]) >= 0.95
        ):
            # Skip — this is a near-duplicate of the previous dialogue segment.
            continue
        result.append(seg)

    return result


def _merge_short_segments(segments: list[dict], source_text: str) -> list[dict]:
    """
    Merge consecutive same-speaker/tone/type segments where the incoming
    segment text is under 120 chars, subject to:
      - Chapter heading protection: the first segment (index 0) is never
        merged into — it always remains its own segment.
      - MERGE_CAP: once the accumulated segment exceeds MERGE_CAP chars,
        stop merging UNLESS the incoming segment continues the same paragraph
        (i.e. does not start a new paragraph boundary in source_text).
    """
    # Build a set of paragraph-start fingerprints from the source text
    para_starts: set[str] = set()
    for para in re.split(r"\n\n+", source_text):
        para = para.strip()
        if para:
            para_starts.add(_normalize_probe(para))

    merged: list[dict] = []
    for seg in segments:
        if not merged:
            # Always start a fresh list with the first segment (chapter heading protection)
            merged.append(seg)
            continue

        # Rule 1: chapter heading protection — the heading is always merged[-1] when
        #         len(merged) == 1, so the very next segment must start its own entry.
        if len(merged) == 1:
            merged.append(seg)
            continue

        # Rule 2: incoming segment is too large to merge
        if len(seg["text"]) >= 120:
            merged.append(seg)
            continue

        # Rule 3: speaker / tone / type must match
        if (
            merged[-1]["speaker"] != seg["speaker"]
            or merged[-1]["tone"]  != seg["tone"]
            or merged[-1]["type"]  != seg["type"]
        ):
            merged.append(seg)
            continue

        # Rule 4: under MERGE_CAP — always merge
        if len(merged[-1]["text"]) < MERGE_CAP:
            merged[-1]["text"] += " " + seg["text"].strip()
            continue

        # Rule 5: over MERGE_CAP — check paragraph boundary
        probe = _normalize_probe(seg["text"])
        if probe in para_starts:
            # Incoming segment starts a new paragraph → clean break
            merged.append(seg)
        else:
            # Mid-paragraph continuation → keep merging to avoid cutting mid-para
            merged[-1]["text"] += " " + seg["text"].strip()

    for i, seg in enumerate(merged):
        seg["index"] = i

    return merged