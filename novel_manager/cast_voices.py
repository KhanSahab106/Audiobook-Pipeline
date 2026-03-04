"""
novel_manager/cast_voices.py

Voice casting system — reads novel.md and voices.md, sends character
profiles to Gemini, and writes optimally matched voices to speakers.json.

Run this AFTER init_novel.py has populated novel.md.

Functions:
    extract_characters_from_novel_md(novel_md) — Parse all character entries from novel.md.
    format_character_profiles(characters) — Format character data for the Gemini prompt.
    load_speakers_json(novel_dir)     — Load existing speakers.json.
    save_speakers_json(novel_dir, registry) — Write speakers.json.
    get_already_assigned_voices(registry) — Return set of voices already in use.
    cast_with_gemini(characters, voices_md, already_used) — Send profiles to Gemini for voice matching.
    validate_assignments(assignments, characters, already_used) — Validate Gemini's voice assignments.
    main()                            — CLI entry point.
Can be re-run anytime novel.md is updated with new characters.

Usage:
    python novel_manager/cast_voices.py novels/shs_and_sws
    python novel_manager/cast_voices.py novels/shs_and_sws --dry-run
    python novel_manager/cast_voices.py novels/shs_and_sws --character idan
"""

import sys
import os
import re
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from novel_manager.gemini_client import call_gemini
from novel_manager.novel_utils   import read_novel_md

# ── Paths ─────────────────────────────────────────────────────────────────────
VOICES_MD_PATH = "voices.md"   # project root — shared across all novels

# Always fixed — never touched by casting
FIXED_ASSIGNMENTS = {
    "narrator": "Ana Florence",
    "system":   "Nova Hogarth",
}

# Voices that cannot be assigned to characters
RESERVED_VOICES = {"Ana Florence", "Nova Hogarth"}
# ─────────────────────────────────────────────────────────────────────────────


CASTING_PROMPT = """\
You are a professional audiobook casting director.
Your job is to assign the best matching TTS voice to each character
based on their personality, gender, age, and speaking style.

━━━ RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

1. Read every character's profile carefully before making any assignments
2. Match voice gender to character gender — never assign a female voice
   to a male character or vice versa. For unknown gender, use your best
   judgement from the name and role.
3. Match voice tone and age to character personality:
   - Young/energetic character → pick a younger-sounding voice
   - Cold/calculating character → pick a cold/precise voice
   - Warm/nurturing character → pick a warm voice
   - etc.
4. Prioritise CONTRAST — characters who share scenes often must have
   voices that sound clearly different from each other. Never assign
   similar-sounding voices to characters who interact frequently.
5. Each voice can only be assigned to ONE character. No duplicates.
6. Do NOT assign Ana Florence or Nova Hogarth — they are reserved.
7. If a character is "unknown" gender, infer from name/role/personality.
8. For minor characters with sparse info, match gender and pick the
   voice least likely to conflict with major characters.

━━━ OUTPUT FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return ONLY a valid JSON object. No explanation, no markdown fences.
Format:
{{
  "character_key": "Voice Name",
  "character_key": "Voice Name",
  ...
}}

Include every character from the list. Use exact voice names as shown
in the voice profiles.

━━━ AVAILABLE VOICES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{voices_md}

━━━ CHARACTERS TO CAST ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{character_profiles}

━━━ ALREADY ASSIGNED (do not reassign these voices) ━━━━━━━━━

{already_assigned}
"""


# ═══════════════════════════════════════════════════════════════════════════
#  NOVEL.MD PARSING
# ═══════════════════════════════════════════════════════════════════════════

def extract_characters_from_novel_md(novel_md: str) -> dict[str, dict]:
    """
    Parse all character entries from novel.md.
    Returns dict of {character_key: {field: value, ...}}
    Skips narrator and system (handled by FIXED_ASSIGNMENTS).
    """
    characters = {}

    # Find the Characters section
    chars_match = re.search(
        r"## Characters\s*\n(.*?)(?=\n## |\Z)",
        novel_md,
        re.DOTALL
    )
    if not chars_match:
        return characters

    chars_section = chars_match.group(1)

    # Split into individual character blocks
    blocks = re.split(r"\n(?=###\s+\S)", chars_section)

    for block in blocks:
        # Get character key from ### heading
        key_match = re.match(r"###\s+(\S+)", block.strip())
        if not key_match:
            continue

        key = key_match.group(1).lower().strip()

        # Skip fixed characters
        if key in FIXED_ASSIGNMENTS:
            continue

        # Parse fields
        fields = {}
        for line in block.split("\n"):
            field_match = re.match(r"\s*-\s+(\w[\w_]*)\s*:\s*(.+)", line)
            if field_match:
                field_name  = field_match.group(1).strip()
                field_value = field_match.group(2).strip()
                fields[field_name] = field_value

        if fields:
            fields["key"] = key
            characters[key] = fields

    return characters


def format_character_profiles(characters: dict[str, dict]) -> str:
    """Format character data into a readable profile block for the prompt."""
    lines = []
    for key, fields in characters.items():
        lines.append(f"### {key}")
        for field in ["gender", "age", "role", "personality",
                      "voice_style", "speech_patterns", "casting_note"]:
            value = fields.get(field, "unknown")
            if value and value.lower() != "unknown":
                lines.append(f"- {field}: {value}")
        lines.append("")
    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════
#  SPEAKERS.JSON
# ═══════════════════════════════════════════════════════════════════════════

def load_speakers_json(novel_dir: str) -> dict:
    path = os.path.join(novel_dir, "data", "speakers.json")
    if os.path.exists(path):
        with open(path, "r") as f:
            return json.load(f)
    return {
        "novel_dir":         novel_dir,
        "characters":        {},
        "female_pool_index": 1,
        "male_pool_index":   0,
    }


def save_speakers_json(novel_dir: str, registry: dict):
    path = os.path.join(novel_dir, "data", "speakers.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(registry, f, indent=2)
    print(f"  Saved: {path}")


def get_already_assigned_voices(registry: dict) -> set[str]:
    """Return set of voices already in use in speakers.json."""
    return {
        v["xtts_speaker"]
        for v in registry.get("characters", {}).values()
        if "xtts_speaker" in v
    }


# ═══════════════════════════════════════════════════════════════════════════
#  GEMINI CASTING CALL
# ═══════════════════════════════════════════════════════════════════════════

def cast_with_gemini(
    characters:    dict[str, dict],
    voices_md:     str,
    already_used:  set[str]
) -> dict[str, str]:
    """
    Send character profiles to Gemini and get back voice assignments.
    Returns {character_key: voice_name}
    """
    already_assigned_str = (
        "\n".join(f"  {v}" for v in sorted(already_used))
        if already_used else "  (none yet)"
    )

    prompt = CASTING_PROMPT.format(
        voices_md            = voices_md,
        character_profiles   = format_character_profiles(characters),
        already_assigned     = already_assigned_str,
    )

    print(f"\n  Sending {len(characters)} characters to Gemini for casting...")
    raw = call_gemini(prompt, label="casting")

    # Strip code fences if present
    raw = re.sub(r"^```(?:json)?\s*\n", "", raw, flags=re.MULTILINE)
    raw = re.sub(r"\n```\s*$",          "", raw, flags=re.MULTILINE)
    raw = raw.strip()

    try:
        assignments = json.loads(raw)
    except json.JSONDecodeError as e:
        # Try to extract JSON object
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            assignments = json.loads(match.group(0))
        else:
            raise ValueError(f"Gemini returned invalid JSON: {e}\nRaw:\n{raw[:500]}")

    return assignments


# ═══════════════════════════════════════════════════════════════════════════
#  VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

# Full valid voice list (must match registry.py)
VALID_VOICES = {
    "Gracie Wise", "Sofia Hellen", "Tanja Adelina", "Barbora MacLean",
    "Szofi Granger", "Claribel Dervla", "Daisy Studious", "Tammie Ema",
    "Alison Dietlinde", "Annmarie Nele", "Brenda Stern", "Gitta Nikolina",
    "Tammy Grit", "Chandra MacFarland", "Camilla Holmström", "Lilya Stainthorpe",
    "Zofija Kendrick", "Narelle Moon", "Rosemary Okafor",
    "Damien Black", "Craig Gutsy", "Torcull Diarmuid", "Ludvig Milivoj",
    "Baldur Sanjin", "Zacharie Aimilios", "Andrew Chipper", "Dionisio Schuyler",
    "Abrahan Mack", "Viktor Menelaos",
}


def validate_assignments(
    assignments:  dict[str, str],
    characters:   dict[str, dict],
    already_used: set[str]
) -> tuple[dict[str, str], list[str]]:
    """
    Validate Gemini's assignments:
    - Voice must be in VALID_VOICES
    - Voice must not be reserved
    - No duplicates within new assignments
    - Warn if gender mismatch detected

    Returns (valid_assignments, warnings)
    """
    valid    = {}
    warnings = []
    used_in_this_batch = set()

    FEMALE_VOICES = {
        "Gracie Wise", "Sofia Hellen", "Tanja Adelina", "Barbora MacLean",
        "Szofi Granger", "Claribel Dervla", "Daisy Studious", "Tammie Ema",
        "Alison Dietlinde", "Annmarie Nele", "Brenda Stern", "Gitta Nikolina",
        "Tammy Grit", "Chandra MacFarland", "Camilla Holmström", "Lilya Stainthorpe",
        "Zofija Kendrick", "Narelle Moon", "Rosemary Okafor",
    }
    MALE_VOICES = {
        "Damien Black", "Craig Gutsy", "Torcull Diarmuid", "Ludvig Milivoj",
        "Baldur Sanjin", "Zacharie Aimilios", "Andrew Chipper", "Dionisio Schuyler",
        "Abrahan Mack", "Viktor Menelaos",
    }

    for char_key, voice in assignments.items():
        # Unknown character
        if char_key not in characters:
            warnings.append(f"  ⚠ '{char_key}' not in character list — skipped")
            continue

        # Invalid voice name
        if voice not in VALID_VOICES:
            warnings.append(f"  ⚠ '{char_key}' → '{voice}' is not a valid voice — skipped")
            continue

        # Reserved voice
        if voice in RESERVED_VOICES:
            warnings.append(f"  ⚠ '{char_key}' → '{voice}' is reserved — skipped")
            continue

        # Already used by another character
        if voice in already_used or voice in used_in_this_batch:
            warnings.append(f"  ⚠ '{char_key}' → '{voice}' already assigned to another character — skipped")
            continue

        # Gender mismatch check
        char_gender = characters[char_key].get("gender", "unknown").lower()
        if char_gender == "male" and voice in FEMALE_VOICES:
            warnings.append(f"  ⚠ '{char_key}' (male) assigned female voice '{voice}' — check this")
        elif char_gender == "female" and voice in MALE_VOICES:
            warnings.append(f"  ⚠ '{char_key}' (female) assigned male voice '{voice}' — check this")

        valid[char_key]              = voice
        used_in_this_batch.add(voice)

    return valid, warnings


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Cast voices for characters using Gemini"
    )
    parser.add_argument(
        "novel_dir",
        help="Path to novel directory, e.g. novels/shs_and_sws"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show assignments without writing speakers.json"
    )
    parser.add_argument(
        "--character",
        default=None,
        help="Cast a single character only, e.g. --character idan"
    )
    parser.add_argument(
        "--recast-all",
        action="store_true",
        help=(
            "Recast ALL characters from scratch, ignoring existing assignments.\n"
            "Use after major novel.md updates. narrator and system are always preserved."
        )
    )
    args = parser.parse_args()

    novel_dir  = os.path.normpath(args.novel_dir)
    novel_name = os.path.basename(novel_dir)

    # Load voices.md
    if not os.path.exists(VOICES_MD_PATH):
        print(f"  ✗ voices.md not found at {VOICES_MD_PATH}")
        print(f"  Make sure you're running from the project root.")
        sys.exit(1)
    with open(VOICES_MD_PATH, "r", encoding="utf-8") as f:
        voices_md = f.read()

    # Load novel.md
    novel_md   = read_novel_md(novel_dir)
    characters = extract_characters_from_novel_md(novel_md)

    if not characters:
        print(f"\n  ✗ No characters found in novel.md")
        print(f"  Run init_novel.py first to populate character entries.")
        sys.exit(1)

    # Load existing speakers.json
    registry     = load_speakers_json(novel_dir)
    already_cast = set(registry.get("characters", {}).keys())
    already_used = get_already_assigned_voices(registry) - RESERVED_VOICES

    # Ensure fixed assignments are always present
    for name, voice in FIXED_ASSIGNMENTS.items():
        registry.setdefault("characters", {})[name] = {"xtts_speaker": voice}

    # Determine which characters need casting
    if args.character:
        # Single character mode
        key = args.character.lower().strip()
        if key not in characters:
            print(f"\n  ✗ '{key}' not found in novel.md")
            sys.exit(1)
        to_cast = {key: characters[key]}
        # Remove their existing assignment so it can be replaced
        if key in registry["characters"] and not args.dry_run:
            old = registry["characters"][key].get("xtts_speaker", "")
            already_used.discard(old)

    elif args.recast_all:
        # Full recast — clear all non-fixed assignments
        to_cast      = characters
        already_used = set()   # treat all as available
        print(f"\n  ⚠ Recasting all {len(to_cast)} characters from scratch")

    else:
        # Default — only cast characters not yet in speakers.json
        to_cast = {
            k: v for k, v in characters.items()
            if k not in already_cast
        }

    if not to_cast:
        print(f"\n  All {len(characters)} characters already cast.")
        print(f"  Use --recast-all to redo all assignments.")
        print(f"  Use --character <key> to recast one character.")
        sys.exit(0)

    print(f"\n{'═' * 55}")
    print(f"  Novel Manager — Voice Casting")
    print(f"  Novel      : {novel_name}")
    print(f"  Characters : {len(to_cast)} to cast")
    if already_used:
        print(f"  Voices in use: {len(already_used)} already assigned")
    print(f"{'═' * 55}")
    print(f"\n  Characters to cast:")
    for key, fields in to_cast.items():
        gender = fields.get("gender", "unknown")
        role   = fields.get("role", "unknown")
        print(f"    {key:<20} [{gender}]  {role}")

    # Get Gemini casting decisions
    raw_assignments = cast_with_gemini(to_cast, voices_md, already_used)

    # Validate
    valid_assignments, warnings = validate_assignments(
        raw_assignments, to_cast, already_used
    )

    # Display results
    print(f"\n{'─' * 55}")
    print(f"  Casting Results:")
    print(f"{'─' * 55}")
    for key, voice in valid_assignments.items():
        gender = to_cast[key].get("gender", "?")
        print(f"  {key:<22} [{gender}]  →  {voice}")

    if warnings:
        print(f"\n  Warnings:")
        for w in warnings:
            print(w)

    skipped = len(to_cast) - len(valid_assignments)
    if skipped:
        print(f"\n  ⚠ {skipped} character(s) skipped due to validation errors")
        print(f"  Run --recast-all or fix novel.md and try again")

    if args.dry_run:
        print(f"\n  Dry run — speakers.json not modified.")
        return

    # Write to speakers.json
    for key, voice in valid_assignments.items():
        gender = to_cast[key].get("gender", "unknown")
        registry["characters"][key] = {
            "xtts_speaker": voice,
            "gender":       gender,
            "cast_by":      "gemini",
        }

    save_speakers_json(novel_dir, registry)

    print(f"\n  ✓ {len(valid_assignments)} characters cast and saved to speakers.json")
    if skipped:
        print(f"  ✗ {skipped} skipped — review warnings above")


if __name__ == "__main__":
    main()
