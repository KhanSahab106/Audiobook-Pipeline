"""
novel_manager/update_novel.py

Arc update — run every 50–100 chapters or after a major arc concludes.

HOW IT WORKS (patch-based architecture):
    Instead of asking Gemini to rewrite the full novel.md (which causes it
    to silently drop characters), we ask Gemini to output ONLY THE CHANGES
    it wants to make, then Python applies those changes surgically to the
    existing file.  Characters can NEVER be pruned by this process because
    Gemini never writes the full character list — it only writes updates.

    Gemini outputs a structured markdown block with:
      - SECTION_UPDATE blocks for Overview/World/Factions/ChapterMap
      - CHARACTER_UPDATE blocks for existing characters (arc_notes + field changes)
      - NEW_CHARACTER blocks for brand-new speakers
      - DORMANT/REACTIVATE markers

    Python merges these into the existing novel.md.

Functions:
    build_prompt(current_novel_md, chapters_text, ch_start, ch_end)
    apply_patches(current_md, patch_text, ch_start, ch_end)
    sync_speakers_json(novel_dir, old_md, new_md, ch_start, ch_end)
    main()

Usage:
    python novel_manager/update_novel.py novels/shadow_slave
    python novel_manager/update_novel.py novels/shadow_slave --chapters 101-200
"""

import sys
import os
import re
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from novel_manager.gemini_client import call_gemini
from novel_manager.novel_utils   import (
    get_chapters_in_range, load_chapters_text,
    read_novel_md, write_novel_md, update_meta_field,
    get_last_updated_chapter, get_all_chapters,
    extract_character_keys, extract_dormant_characters,
    extract_newly_dormant, extract_reactivated, parse_chapter_range,
    _parse_character_entries
)


# ═══════════════════════════════════════════════════════════════════════════
#  PROMPT
# ═══════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """\
You are the analyst for an AI audiobook production pipeline. Your job is to
update a character/world database (novel.md) after new chapters have been read.

CRITICAL: You do NOT rewrite novel.md. You only output the CHANGES you want
to make. A separate program applies your changes to the existing file. Nothing
will be deleted unless you explicitly use the PRUNE command with justification.

━━━ WHAT YOU WILL RECEIVE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
1. CHARACTER INDEX — a compact list of existing characters (key, status,
   last_updated_chapter, role tier). Use this to identify who already exists.
2. CURRENT novel.md — the full existing document for reference.
3. NEW CHAPTERS — raw chapter text to analyze.

━━━ WHAT YOU OUTPUT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Output ONLY a structured patch block. Nothing else. No explanations outside
the patch block. The patch block starts with <<<PATCH>>> and ends with
<<<END_PATCH>>>.

Inside the patch block, use these commands:

─── CHARACTER COMMANDS ──────────────────────────────────────────

UPDATE_CHARACTER: existing_key
  Add ONLY fields that changed or new arc_notes lines. Do NOT repeat
  existing arc_notes. Format:
    arc_notes_append:
      Ch N: [what happened]
      Ch N: [what happened]
    status: active          ← only if changing
    confidence: complete    ← only if upgrading
    role: major supporting  ← only if changing
    last_updated_chapter: N ← only if they had dialogue in new chapters

NEW_CHARACTER: key_name
  Full character entry for a brand-new character with dialogue in new chapters.
  Format (all fields required):
    - confidence: sparse
    - status: active
    - introduced_chapter: N
    - role: [protagonist/antagonist/major supporting/supporting/minor]
    - gender: [male/female/unknown]
    - age: [age or unknown]
    - personality: [2-3 sentences]
    - voice_style: [how they sound]
    - speech_patterns: [notable quirks or unknown]
    - arc_notes:
      Ch N: [first appearance]
    - relationship_to_protagonist: [one sentence]
    - casting_note: [voice quality hints]
    - last_updated_chapter: N

MARK_DORMANT: key_name
  Mark an existing character as dormant. Only use when they exceed their tier's
  threshold. Include the arc_notes line to add:
    arc_notes_append:
      Ch N: marked dormant (no dialogue since Ch X)

REACTIVATE: key_name
  Reactivate a dormant character who has dialogue in new chapters.
    arc_notes_append:
      Ch N: reactivated

PRUNE: key_name [EXTREME LAST RESORT]
  Move character to ## Pruned Characters. Only valid when ALL of these are true:
    - They were ALREADY status: dormant in the current novel.md
    - They have been silent for 2× their tier threshold (protagonist=never)
    - They have no ongoing plotlines and are clearly gone from the story
  If unsure, use MARK_DORMANT instead. Do NOT prune active characters.

─── SECTION COMMANDS ────────────────────────────────────────────

UPDATE_SECTION: Overview
  [Additional text to APPEND to the Overview section — do not rewrite it]

UPDATE_SECTION: World & Tone Notes
  [Additional text to APPEND]

UPDATE_SECTION: Factions
  [New factions or updates to append]

APPEND_CHAPTER_MAP:
  Ch N: [summary]
  Ch N: [summary]
  ...

━━━ RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

NEW CHARACTERS — scan every chapter carefully:
  - Create NEW_CHARACTER for any named individual with quoted dialogue
    who is NOT in the Character Index.
  - Do NOT create entries for unnamed/unknown speakers ("a soldier", "the guard")
  - Do NOT create entries for doppelgangers of existing characters
    (e.g. Idan's doppelganger). Only create if it's an independent named character.
  - In 50 chapters, expect 3–8 new speakers. Finding 0–1 is suspicious — recheck.

EXISTING CHARACTERS — update only characters with actual activity:
  - Only add arc_notes for chapters where they have DIALOGUE (quoted speech)
    or significant plot events directly involving them.
  - Only update last_updated_chapter when they have actual DIALOGUE.
  - Skip characters with no activity in new chapters — do NOT output any
    command for them (they stay unchanged).

DORMANCY — apply these silence thresholds:
  protagonist            → NEVER dormant
  deuteragonist/antagonist (S-Tier) → 200 silent chapters
  major supporting (A-Tier) → 150 silent chapters
  supporting (B-Tier)    → 100 silent chapters
  minor (C-Tier) / unspecified → 50/30 silent chapters
  Count from their last_updated_chapter to the END of the new batch.

NARRATOR AND SYSTEM:
  - NEVER mark narrator or system as dormant.
  - Do not output UPDATE_CHARACTER for narrator or system unless they have
    genuinely new arc_notes to add.

━━━ OUTPUT FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

<<<PATCH>>>
UPDATE_CHARACTER: character_key
  arc_notes_append:
    Ch 155: [what happened]
  last_updated_chapter: 155

NEW_CHARACTER: new_character_key
  - confidence: sparse
  - status: active
  - introduced_chapter: 157
  - role: supporting
  - gender: female
  - age: unknown
  - personality: [description]
  - voice_style: [description]
  - speech_patterns: unknown
  - arc_notes:
    Ch 157: [first appearance and key events]
  - relationship_to_protagonist: [description]
  - casting_note: [voice quality]
  - last_updated_chapter: 157

MARK_DORMANT: some_character
  arc_notes_append:
    Ch 200: marked dormant (no dialogue since Ch 120)

APPEND_CHAPTER_MAP:
  Ch 151: [summary]
  Ch 152: [summary]
<<<END_PATCH>>>
"""


def _build_character_index(novel_md: str) -> str:
    """Build a compact index of all existing characters for the prompt."""
    entries = _parse_character_entries(novel_md)
    lines = ["key | status | role | last_updated_chapter",
             "─────────────────────────────────────────"]
    for key, info in sorted(entries.items()):
        luc = info.get("last_updated_chapter") or "?"
        lines.append(f"{key:35s} | {info['status']:7s} | {info['role']:20s} | {luc}")
    return "\n".join(lines)


def build_prompt(
    current_novel_md: str,
    chapters_text: str,
    ch_start: int,
    ch_end: int
) -> str:
    char_index = _build_character_index(current_novel_md)
    return f"""{SYSTEM_PROMPT}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHARACTER INDEX (existing characters — do NOT recreate these):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{char_index}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT novel.md (for reference — you do NOT rewrite this):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{current_novel_md}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEW CHAPTERS ({ch_start}–{ch_end}) — analyze these:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chapters_text}
"""


# ═══════════════════════════════════════════════════════════════════════════
#  PATCH PARSER + APPLICATOR
# ═══════════════════════════════════════════════════════════════════════════

def _extract_patch_block(raw: str) -> str:
    """Extract content between <<<PATCH>>> and <<<END_PATCH>>>."""
    m = re.search(r"<<<PATCH>>>(.*?)<<<END_PATCH>>>", raw, re.DOTALL)
    if not m:
        raise ValueError(
            "Gemini response did not contain a <<<PATCH>>>...<<<END_PATCH>>> block.\n"
            f"Raw response (first 500 chars):\n{raw[:500]}"
        )
    return m.group(1).strip()


def _split_patch_commands(patch_text: str) -> list[tuple[str, str, str]]:
    """
    Parse patch text into list of (command, argument, body) tuples.

    Commands: UPDATE_CHARACTER, NEW_CHARACTER, MARK_DORMANT, REACTIVATE,
              PRUNE, UPDATE_SECTION, APPEND_CHAPTER_MAP
    """
    # Split on command lines (start of line, all-caps command)
    command_pattern = re.compile(
        r'^(UPDATE_CHARACTER|NEW_CHARACTER|MARK_DORMANT|REACTIVATE|PRUNE'
        r'|UPDATE_SECTION|APPEND_CHAPTER_MAP)(?::\s*(.+))?$',
        re.MULTILINE
    )

    commands = []
    positions = [(m.start(), m.group(1), m.group(2) or "") for m in command_pattern.finditer(patch_text)]

    for i, (pos, cmd, arg) in enumerate(positions):
        end = positions[i + 1][0] if i + 1 < len(positions) else len(patch_text)
        body_start = patch_text.index('\n', pos) + 1 if '\n' in patch_text[pos:end] else end
        body = patch_text[body_start:end].strip()
        commands.append((cmd, arg.strip(), body))

    return commands


def _parse_arc_notes_append(body: str) -> str:
    """Extract arc_notes lines from an arc_notes_append: block."""
    m = re.search(r"arc_notes_append:\s*\n(.*?)(?=\n\S|\Z)", body, re.DOTALL)
    if not m:
        return ""
    lines = m.group(1)
    # Normalize indentation — ensure each line starts with "  "
    normalized = []
    for line in lines.split("\n"):
        stripped = line.strip()
        if stripped:
            normalized.append(f"  {stripped}")
    return "\n".join(normalized)


def _get_field_updates(body: str) -> dict[str, str]:
    """
    Extract simple field: value pairs from body (excluding arc_notes_append block).

    We skip lines that are part of the arc_notes_append block (indented Ch N: lines)
    but continue scanning after the block so fields that appear AFTER arc_notes
    (like last_updated_chapter) are still captured.
    """
    fields = {}
    in_arc_block = False
    for line in body.split("\n"):
        if re.match(r"\s*arc_notes_append:\s*$", line):
            in_arc_block = True
            continue
        if in_arc_block:
            # Arc notes lines are indented (start with whitespace then Ch or similar)
            if re.match(r"\s+\S", line):
                continue   # still inside arc block, skip
            else:
                in_arc_block = False  # back to top-level fields
        m = re.match(r"\s*([a-z_]+):\s*(.+)", line)
        if m and m.group(1) not in ("arc_notes_append",):
            fields[m.group(1)] = m.group(2).strip()
    return fields


def _apply_update_character(novel_md: str, key: str, body: str) -> str:
    """Apply UPDATE_CHARACTER patch to novel.md."""
    # Find the character's section
    pattern = re.compile(rf'^### {re.escape(key)}\s*\n', re.MULTILINE)
    m = pattern.search(novel_md)
    if not m:
        print(f"  [!] UPDATE_CHARACTER '{key}' -- not found in novel.md, skipping")
        return novel_md

    # Find the end of this character's block (next ### or next ## or EOF)
    block_start = m.start()
    next_block = re.search(r'^(?:###|##) ', novel_md[m.end():], re.MULTILINE)
    block_end = m.end() + next_block.start() if next_block else len(novel_md)
    char_block = novel_md[block_start:block_end]

    # 1. Append arc_notes lines
    arc_append = _parse_arc_notes_append(body)
    if arc_append:
        # Find last_updated_chapter line or end of arc_notes block
        luc_m = re.search(r'^- last_updated_chapter:', char_block, re.MULTILINE)
        if luc_m:
            insert_pos = block_start + luc_m.start()
            char_block = char_block[:luc_m.start()] + arc_append + "\n" + char_block[luc_m.start():]
        else:
            char_block = char_block.rstrip() + "\n" + arc_append + "\n"

    # 2. Apply field updates
    field_updates = _get_field_updates(body)
    for field, value in field_updates.items():
        field_pat = re.compile(rf'^(- {re.escape(field)}:)\s*.+$', re.MULTILINE)
        if field_pat.search(char_block):
            char_block = field_pat.sub(rf'\1 {value}', char_block)
        # If field doesn't exist, we skip (don't add random fields)

    novel_md = novel_md[:block_start] + char_block + novel_md[block_end:]
    return novel_md


def _normalize_new_character_body(body: str) -> str:
    """
    Normalize field indentation in a NEW_CHARACTER body.

    Gemini writes fields as '  - field: value' (leading spaces before the dash).
    novel.md parsers (novel_utils._parse_character_entries, cast_voices) require
    top-level fields to start at column 0: '- field: value'.

    Arc notes continuation lines like '    Ch N: ...' have their indent
    reduced proportionally so they stay indented relative to the '- arc_notes:' line.

    Example input:
        - confidence: sparse
        - status: active
        - arc_notes:
          Ch 155: first appearance
        - last_updated_chapter: 155

    Example output (same — already correct if no leading spaces)
    Or if Gemini added extra leading spaces, they get stripped.
    """
    lines = body.split("\n")
    # Detect the common leading whitespace on field lines (lines starting with - after stripping)
    field_lines = [l for l in lines if l.lstrip().startswith("-")]
    if not field_lines:
        return body

    # Find minimum indent on field lines (the amount to strip from all lines)
    min_indent = min(len(l) - len(l.lstrip()) for l in field_lines)
    if min_indent == 0:
        return body  # already normalized

    normalized = []
    for line in lines:
        if len(line) >= min_indent:
            normalized.append(line[min_indent:])
        else:
            normalized.append(line.lstrip())
    return "\n".join(normalized)


def _apply_new_character(novel_md: str, key: str, body: str) -> str:
    """Append a new character to the ## Characters section."""
    existing_keys = extract_character_keys(novel_md)
    if key in existing_keys:
        print(f"  [!] NEW_CHARACTER '{key}' already exists -- converting to UPDATE")
        return _apply_update_character(novel_md, key, body)

    # Normalize indentation so parsers can read the fields
    body = _normalize_new_character_body(body)
    new_entry = f"\n### {key}\n{body}\n"

    # Find the ## Characters section, then find the next ## section after it.
    # Insert new entry just before that next section (= end of Characters block).
    chars_match = re.search(r'^## Characters\s*\n', novel_md, re.MULTILINE)
    if chars_match:
        after_chars = novel_md[chars_match.end():]
        next_section = re.search(r'^## ', after_chars, re.MULTILINE)
        if next_section:
            insert_at = chars_match.end() + next_section.start()
        else:
            insert_at = len(novel_md)
        novel_md = novel_md[:insert_at].rstrip() + new_entry + "\n" + novel_md[insert_at:]
    else:
        # No Characters section found — append at end
        novel_md = novel_md.rstrip() + new_entry

    print(f"  + NEW_CHARACTER: {key}")
    return novel_md



def _apply_mark_dormant(novel_md: str, key: str, body: str) -> str:
    """Mark a character as dormant and append arc_notes."""
    novel_md = _apply_update_character(novel_md, key, body)

    # Change status: active → dormant
    pattern = re.compile(rf'(### {re.escape(key)}\s*\n.*?)(- status:\s*active)', re.DOTALL)
    def replace_status(m):
        return m.group(1) + "- status: dormant"

    # Scoped replacement (only within that character's block)
    block_pat = re.compile(rf'^### {re.escape(key)}\s*\n', re.MULTILINE)
    bm = block_pat.search(novel_md)
    if bm:
        next_block = re.search(r'^(?:###|##) ', novel_md[bm.end():], re.MULTILINE)
        block_end = bm.end() + next_block.start() if next_block else len(novel_md)
        char_block = novel_md[bm.start():block_end]
        char_block = re.sub(r'^- status:\s*\w+', '- status: dormant', char_block, flags=re.MULTILINE)
        novel_md = novel_md[:bm.start()] + char_block + novel_md[block_end:]

    print(f"  -> MARK_DORMANT: {key}")
    return novel_md


def _apply_reactivate(novel_md: str, key: str, body: str) -> str:
    """Reactivate a dormant character and append arc_notes."""
    novel_md = _apply_update_character(novel_md, key, body)

    block_pat = re.compile(rf'^### {re.escape(key)}\s*\n', re.MULTILINE)
    bm = block_pat.search(novel_md)
    if bm:
        next_block = re.search(r'^(?:###|##) ', novel_md[bm.end():], re.MULTILINE)
        block_end = bm.end() + next_block.start() if next_block else len(novel_md)
        char_block = novel_md[bm.start():block_end]
        char_block = re.sub(r'^- status:\s*\w+', '- status: active', char_block, flags=re.MULTILINE)
        novel_md = novel_md[:bm.start()] + char_block + novel_md[block_end:]

    print(f"  ^ REACTIVATE: {key}")
    return novel_md


def _apply_prune(novel_md: str, key: str, old_entries: dict) -> tuple[str, bool]:
    """
    Move a character to ## Pruned Characters.
    Returns (updated_md, was_pruned).
    Refuses to prune active characters.
    """
    old_entry = old_entries.get(key, {})
    old_status = old_entry.get("status", "active")

    if old_status == "active":
        print(f"  [REJECTED] PRUNE '{key}' -- character was active. Keeping.")
        return novel_md, False

    # Remove from ## Characters
    block_pat = re.compile(rf'^### {re.escape(key)}\s*\n', re.MULTILINE)
    bm = block_pat.search(novel_md)
    if not bm:
        return novel_md, False

    next_block = re.search(r'^(?:###|##) ', novel_md[bm.end():], re.MULTILINE)
    block_end = bm.end() + next_block.start() if next_block else len(novel_md)
    novel_md = novel_md[:bm.start()] + novel_md[block_end:]

    # Add to ## Pruned Characters section
    pruned_section = re.search(r'^## Pruned Characters\s*\n', novel_md, re.MULTILINE)
    if pruned_section:
        insert_at = pruned_section.end()
        novel_md = novel_md[:insert_at] + f"- {key}\n" + novel_md[insert_at:]
    else:
        novel_md = novel_md.rstrip() + f"\n\n## Pruned Characters\n- {key}\n"

    print(f"  [PRUNED] {key}")
    return novel_md, True


def _apply_append_chapter_map(novel_md: str, body: str) -> str:
    """Append new chapter map entries to the Chapter Map section."""
    new_lines = body.strip()
    if not new_lines:
        return novel_md

    map_match = re.search(r'^## Chapter Map\s*\n', novel_md, re.MULTILINE)
    if map_match:
        # Find the end of this section
        next_section = re.search(r'^## ', novel_md[map_match.end():], re.MULTILINE)
        if next_section:
            insert_at = map_match.end() + next_section.start()
        else:
            insert_at = len(novel_md)
        novel_md = novel_md[:insert_at].rstrip() + "\n" + new_lines + "\n\n" + novel_md[insert_at:]
    else:
        novel_md = novel_md.rstrip() + f"\n\n## Chapter Map\n{new_lines}\n"

    return novel_md


def _apply_update_section(novel_md: str, section_name: str, body: str) -> str:
    """Append content to a named ## section."""
    if not body.strip():
        return novel_md

    section_pat = re.compile(rf'^## {re.escape(section_name)}\s*\n', re.MULTILINE)
    sm = section_pat.search(novel_md)
    if sm:
        next_section = re.search(r'^## ', novel_md[sm.end():], re.MULTILINE)
        insert_at = sm.end() + next_section.start() if next_section else len(novel_md)
        novel_md = novel_md[:insert_at].rstrip() + "\n\n" + body.strip() + "\n\n" + novel_md[insert_at:]
    # If section doesn't exist, skip (don't create arbitrary sections)
    return novel_md


def apply_patches(current_md: str, patch_text: str, ch_start: int, ch_end: int) -> tuple[str, list[str]]:
    """
    Apply all patch commands to current_md.

    Returns (updated_md, list_of_rejected_prunes).
    """
    commands = _split_patch_commands(patch_text)
    novel_md = current_md
    rejected_prunes = []
    old_entries = _parse_character_entries(current_md)

    for cmd, arg, body in commands:
        if cmd == "UPDATE_CHARACTER":
            novel_md = _apply_update_character(novel_md, arg, body)

        elif cmd == "NEW_CHARACTER":
            novel_md = _apply_new_character(novel_md, arg, body)

        elif cmd == "MARK_DORMANT":
            novel_md = _apply_mark_dormant(novel_md, arg, body)

        elif cmd == "REACTIVATE":
            novel_md = _apply_reactivate(novel_md, arg, body)

        elif cmd == "PRUNE":
            novel_md, was_pruned = _apply_prune(novel_md, arg, old_entries)
            if not was_pruned:
                rejected_prunes.append(arg)

        elif cmd == "APPEND_CHAPTER_MAP":
            novel_md = _apply_append_chapter_map(novel_md, body)

        elif cmd == "UPDATE_SECTION":
            novel_md = _apply_update_section(novel_md, arg, body)

        else:
            print(f"  [!] Unknown patch command: {cmd}")

    return novel_md, rejected_prunes


# ═══════════════════════════════════════════════════════════════════════════
#  SPEAKERS.JSON SYNC
# ═══════════════════════════════════════════════════════════════════════════

def sync_speakers_json(novel_dir: str, old_md: str, new_md: str, ch_start: int = None, ch_end: int = None):
    """
    Sync speakers.json after a novel.md update.

    In the patch-based model, Python (apply_patches) is the sole gatekeeper for
    all character removals — Gemini never deletes entries directly. So any key
    present in old_md but missing from new_md here is a DELIBERATE prune that
    already passed _apply_prune's active-character safety check. No second
    validation is needed in this function.

    Handles three cases:
    1. Newly dormant characters — free voice slot, save to dormant_voices.
    2. Reactivated characters   — restore from dormant_voices (or warn).
    3. Fully pruned characters  — remove from both characters and dormant_voices.
    """
    speakers_path = os.path.join(novel_dir, "data", "speakers.json")
    if not os.path.exists(speakers_path):
        print(f"  [!] speakers.json not found -- skipping voice sync")
        return

    with open(speakers_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    characters     = registry.get("characters", {})
    dormant_voices = registry.setdefault("dormant_voices", {})

    # ── 0. Catch-up: characters already dormant but still in characters ──────
    all_dormant_in_new = extract_dormant_characters(new_md) - {"narrator", "system", "unknown"}
    freed = []
    for key in all_dormant_in_new:
        if key in characters and key not in dormant_voices:
            entry = characters[key]
            dormant_voices[key] = {
                "xtts_speaker":          entry.get("xtts_speaker", "?"),
                "gender":                entry.get("gender", "unknown"),
                "dormant_since_chapter": entry.get("last_updated_chapter"),
            }
            del characters[key]
            freed.append((key, dormant_voices[key]["xtts_speaker"]))

    # ── 1. Newly dormant ────────────────────────────────────────────────────
    newly_dormant = extract_newly_dormant(old_md, new_md) - {"narrator", "system", "unknown"}
    for key in newly_dormant:
        if key in characters:
            entry = characters[key]
            dormant_voices[key] = {
                "xtts_speaker":          entry.get("xtts_speaker", "?"),
                "gender":                entry.get("gender", "unknown"),
                "dormant_since_chapter": entry.get("last_updated_chapter"),
            }
            del characters[key]
            freed.append((key, dormant_voices[key]["xtts_speaker"]))

    # ── 2. Reactivated ──────────────────────────────────────────────────────
    reactivated = extract_reactivated(old_md, new_md) - {"narrator", "system", "unknown"}
    restored = []
    needs_cast = []
    for key in reactivated:
        if key in dormant_voices:
            saved = dormant_voices.pop(key)
            voice = saved["xtts_speaker"]
            used  = {v["xtts_speaker"] for v in characters.values()}
            if voice not in used:
                characters[key] = {
                    "xtts_speaker": voice,
                    "gender":       saved.get("gender", "unknown"),
                    "cast_by":      "dormancy_restore",
                }
                restored.append((key, voice))
            else:
                needs_cast.append(key)
        else:
            needs_cast.append(key)

    # ── 3. Fully pruned (Python-controlled, already validated by _apply_prune) ─
    # Any key in old_md but NOT in new_md was explicitly removed by apply_patches.
    # No further safety checks needed — _apply_prune already rejected invalid prunes.
    old_keys = extract_character_keys(old_md)
    new_keys = extract_character_keys(new_md)
    pruned   = old_keys - new_keys - {"narrator", "system", "unknown"}

    removed = []
    for key in pruned:
        if key in characters:
            del characters[key]
            removed.append(key)
        if key in dormant_voices:
            del dormant_voices[key]
            if key not in removed:
                removed.append(key)

    # ── Persist if anything changed ─────────────────────────────────────────
    if freed or restored or removed or needs_cast:
        with open(speakers_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=True)

    if freed:
        print(f"\n  {'-' * 50}")
        print(f"  -> Freed {len(freed)} voice slot(s) for dormant character(s):")
        for key, voice in sorted(freed):
            print(f"      {key:30s} -> dormant_voices (was: {voice})")
        print(f"  {'-' * 50}")

    if restored:
        print(f"\n  {'-' * 50}")
        print(f"  [OK] Restored {len(restored)} reactivated character(s):")
        for key, voice in sorted(restored):
            print(f"      {key:30s} -> {voice}")
        print(f"  {'-' * 50}")

    if needs_cast:
        print(f"\n  {'-' * 50}")
        print(f"  [!] {len(needs_cast)} reactivated character(s) need voice assignment:")
        for key in sorted(needs_cast):
            print(f"      {key}")
        print(f"    Run cast_voices.py to assign voices.")
        print(f"  {'-' * 50}")

    if removed:
        print(f"\n  {'-' * 50}")
        print(f"  [DEL] Removed {len(removed)} fully pruned character(s):")
        for key in sorted(removed):
            print(f"      {key}")
        print(f"  {'-' * 50}")


def prune_speakers_json(novel_dir: str, old_md: str, new_md: str, ch_start: int = None, ch_end: int = None):
    """Backward-compatible wrapper — delegates to sync_speakers_json."""
    sync_speakers_json(novel_dir, old_md, new_md, ch_start=ch_start, ch_end=ch_end)


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Update novel.md with new chapters (patch-based)"
    )
    parser.add_argument(
        "novel_dir",
        help="Path to novel directory, e.g. novels/shadow_slave"
    )
    parser.add_argument(
        "--chapters",
        default=None,
        help=(
            "Chapter range to process, e.g. 101-200 or 101- (all from 101).\n"
            "If omitted, automatically uses all chapters after last_updated_chapter."
        )
    )
    args = parser.parse_args()

    novel_dir  = os.path.normpath(args.novel_dir)
    novel_name = os.path.basename(novel_dir)
    last_done  = get_last_updated_chapter(novel_dir)

    # Auto-detect range if not specified
    if args.chapters:
        start, end = parse_chapter_range(args.chapters)
    else:
        start = last_done + 1
        end   = None
        print(f"  Auto-detected: processing from chapter {start} onwards")

    chapters = get_chapters_in_range(novel_dir, start, end)

    if not chapters:
        print(f"\n  Nothing to update.")
        print(f"  last_updated_chapter is {last_done}.")
        print(f"  No chapter files found after chapter {last_done} in input/")
        sys.exit(0)

    ch_nums     = [n for n, _ in chapters]
    total_done  = len(get_all_chapters(novel_dir))

    print(f"\n{'=' * 55}")
    print(f"  Novel Manager -- Arc Update (patch mode)")
    print(f"  Novel       : {novel_name}")
    print(f"  New chapters: {ch_nums[0]}-{ch_nums[-1]}  ({len(chapters)} files)")
    print(f"  Previously  : through chapter {last_done}")
    print(f"  Words       : {sum(len(open(f, encoding='utf-8').read().split()) for _, f in chapters):,}")
    print(f"{'=' * 55}")

    current_md    = read_novel_md(novel_dir)
    chapters_text = load_chapters_text(chapters)
    prompt        = build_prompt(current_md, chapters_text, ch_nums[0], ch_nums[-1])

    print(f"\n  Sending to Gemini...")
    raw_response = call_gemini(prompt, label="update")

    # Extract and apply the patch
    print(f"\n  Applying patches...")
    try:
        patch_text = _extract_patch_block(raw_response)
    except ValueError as e:
        print(f"\n  [!] {e}")
        print(f"  Saving raw Gemini response for inspection...")
        debug_path = os.path.join(novel_dir, "data", "last_update_raw.txt")
        with open(debug_path, "w", encoding="utf-8") as f:
            f.write(raw_response)
        print(f"  Raw response saved to: {debug_path}")
        sys.exit(1)

    updated_md, rejected_prunes = apply_patches(current_md, patch_text, ch_nums[0], ch_nums[-1])

    # Update meta fields
    updated_md = update_meta_field(updated_md, "last_updated_chapter", str(ch_nums[-1]))
    updated_md = update_meta_field(updated_md, "total_chapters_processed", str(total_done))

    write_novel_md(novel_dir, updated_md)

    # Report rejected prunes (now much rarer since we control pruning)
    if rejected_prunes:
        print(f"\n  {'-' * 50}")
        print(f"  [!] Rejected {len(rejected_prunes)} invalid PRUNE command(s):")
        for key in sorted(rejected_prunes):
            print(f"      {key:30s} -- was status: active")
        print(f"  {'-' * 50}")

    # Sync voice slots
    sync_speakers_json(novel_dir, current_md, updated_md, ch_start=ch_nums[0], ch_end=ch_nums[-1])

    print(f"\n  [OK] novel.md updated through chapter {ch_nums[-1]}")
    print(f"  Backup saved in {os.path.join(novel_dir, 'data', 'novel_backups')}/")


if __name__ == "__main__":
    main()
