"""
registry.py

Voice registry — manages speaker-to-voice mappings for the audiobook pipeline.
Assigns XTTS voices to characters based on gender (from novel.md), persists
the assignments in novels/{novel}/data/speakers.json, and resolves speaker
names at render time.

Functions:
    _registry_path(novel_dir)         — Return path to speakers.json for a novel.
    _novel_md_path(novel_dir)         — Return path to novel.md for a novel.
    _read_gender_from_novel_md(...)   — Look up a character's gender in novel.md.
    load_registry(novel_dir)          — Load or initialise the speakers.json registry.
    save_registry(registry, novel_dir)— Persist registry dict to speakers.json.
    _assign_voice(registry, name, gender, novel_dir) — Assign next available voice from the pool.
    resolve_speaker(speaker_name, registry, novel_dir) — Map a speaker key to an XTTS voice name.
    get_known_characters(registry)    — Return list of all character keys in the registry.
    reassign_speaker(novel_dir, character_key, new_voice) — Manually reassign a voice.
"""

import json
import os
import re

# ── Voice pool — split by gender ──────────────────────────────────────────────
# Nova Hogarth is ONLY in FIXED_ASSIGNMENTS (system voice).
# It is deliberately excluded from both pools to prevent duplicate assignment.

FEMALE_POOL = [
    "Ana Florence",       # index 0 — narrator only, never assigned to characters
    "Gracie Wise",        # 1
    "Sofia Hellen",       # 2
    "Tanja Adelina",      # 3
    "Barbora MacLean",    # 4
    "Szofi Granger",      # 5
    "Claribel Dervla",    # 6
    "Daisy Studious",     # 7
    "Tammie Ema",         # 8
    "Alison Dietlinde",   # 9
    "Annmarie Nele",      # 10
    "Brenda Stern",       # 11
    "Gitta Nikolina",     # 12
    "Tammy Grit",         # 13
    "Chandra MacFarland", # 14
    "Camilla Holmström",  # 15
    "Lilya Stainthorpe",  # 16
    "Zofija Kendrick",    # 17
    "Narelle Moon",       # 18
    "Rosemary Okafor",    # 19
]

MALE_POOL = [
    "Damien Black",       # 0
    "Craig Gutsy",        # 1
    "Torcull Diarmuid",   # 2
    "Ludvig Milivoj",     # 3
    "Baldur Sanjin",      # 4
    "Zacharie Aimilios",  # 5
    "Andrew Chipper",     # 6
    "Dionisio Schuyler",  # 7
    "Abrahan Mack",       # 8
    "Viktor Menelaos",    # 9
]

# Always fixed — never come from the pools
FIXED_ASSIGNMENTS = {
    "narrator": "Ana Florence",
    "system":   "Nova Hogarth",
}

PRONOUN_NAMES = {"he", "she", "they", "it", "him", "her", "them", "his", "hers"}


# ═══════════════════════════════════════════════════════════════════════════
#  PATHS
# ═══════════════════════════════════════════════════════════════════════════

def _registry_path(novel_dir: str) -> str:
    return os.path.join(novel_dir, "data", "speakers.json")


def _novel_md_path(novel_dir: str) -> str:
    return os.path.join(novel_dir, "novel.md")


# ═══════════════════════════════════════════════════════════════════════════
#  NOVEL.MD GENDER LOOKUP
# ═══════════════════════════════════════════════════════════════════════════

def _read_gender_from_novel_md(novel_dir: str, character_key: str) -> str:
    """
    Read novel.md and return the gender field for a character.
    Returns "female", "male", or "unknown".

    Looks for a section like:
        ### character_key
        - gender: male
    """
    path = _novel_md_path(novel_dir)
    if not os.path.exists(path):
        return "unknown"

    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()

        # Find the character's section
        pattern = rf"###\s+{re.escape(character_key)}\s*\n(.*?)(?=\n###|\n---|\Z)"
        match   = re.search(pattern, content, re.DOTALL | re.IGNORECASE)

        if not match:
            return "unknown"

        section = match.group(1)
        gender_match = re.search(r"-\s*gender:\s*(\w+)", section, re.IGNORECASE)

        if not gender_match:
            return "unknown"

        gender = gender_match.group(1).lower().strip()

        if gender in ("male", "man", "boy"):
            return "male"
        if gender in ("female", "woman", "girl"):
            return "female"
        return "unknown"

    except Exception:
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════════
#  LOAD / SAVE
# ═══════════════════════════════════════════════════════════════════════════

def load_registry(novel_dir: str) -> dict:
    path = _registry_path(novel_dir)

    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        # Migrate old format (single pool_index) to new split format
        if "pool_index" in data and "female_pool_index" not in data:
            data["female_pool_index"] = 1   # start after Ana Florence
            data["male_pool_index"]   = 0
            data.pop("pool_index", None)
            save_registry(data, novel_dir)
            print("  Registry migrated to gender-aware pool format.")
        # Ensure dormant_voices key exists (backward compatible)
        if "dormant_voices" not in data:
            data["dormant_voices"] = {}
        return data

    # First run — seed with fixed assignments only
    registry = {
        "novel_dir":         novel_dir,
        "characters":        {},
        "dormant_voices":    {},
        "female_pool_index": 1,   # index 0 (Ana Florence) reserved for narrator
        "male_pool_index":   0,
    }
    for name, speaker in FIXED_ASSIGNMENTS.items():
        registry["characters"][name] = {"xtts_speaker": speaker}

    return registry


def save_registry(registry: dict, novel_dir: str):
    path = _registry_path(novel_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(registry, f, indent=2, ensure_ascii=False)


# ═══════════════════════════════════════════════════════════════════════════
#  VOICE ASSIGNMENT
# ═══════════════════════════════════════════════════════════════════════════

def _assign_voice(registry: dict, name: str, gender: str, novel_dir: str) -> str:
    """
    Assign the next available voice from the appropriate gender pool.
    Falls back to the other pool if the correct one is exhausted.
    Falls back to Ana Florence if both are exhausted.
    """
    # Build set of already-used voices to avoid any duplicate
    used = {v["xtts_speaker"] for v in registry["characters"].values()}

    def next_from_pool(pool: list, index_key: str) -> str | None:
        idx = registry[index_key]
        while idx < len(pool):
            candidate = pool[idx]
            registry[index_key] = idx + 1
            if candidate not in used:
                return candidate
            idx += 1
            registry[index_key] = idx
        return None

    if gender == "male":
        voice = next_from_pool(MALE_POOL,   "male_pool_index")
        if voice is None:
            print(f"  Warning: male pool exhausted for '{name}', trying female pool")
            voice = next_from_pool(FEMALE_POOL, "female_pool_index")
    elif gender == "female":
        voice = next_from_pool(FEMALE_POOL, "female_pool_index")
        if voice is None:
            print(f"  Warning: female pool exhausted for '{name}', trying male pool")
            voice = next_from_pool(MALE_POOL, "male_pool_index")
    else:
        # Unknown gender — try male pool first (most unnamed/minor
        # characters in novels skew male), then female
        voice = next_from_pool(MALE_POOL, "male_pool_index")
        if voice is None:
            voice = next_from_pool(FEMALE_POOL, "female_pool_index")

    if voice is None:
        print(f"  Warning: all voice pools exhausted for '{name}' — using Ana Florence")
        voice = "Ana Florence"

    return voice


# ═══════════════════════════════════════════════════════════════════════════
#  SPEAKER RESOLUTION
# ═══════════════════════════════════════════════════════════════════════════

def resolve_speaker(speaker_name: str, registry: dict, novel_dir: str) -> str:
    name = speaker_name.lower().strip()

    if name in PRONOUN_NAMES:
        print(f"  Warning: pronoun '{name}' used as speaker → narrator")
        name = "narrator"

    # Already assigned — return immediately (the common case)
    # This covers both Gemini-cast characters and previously seen runtime characters
    if name in registry["characters"]:
        entry    = registry["characters"][name]
        cast_by  = entry.get("cast_by", "sequential")
        if cast_by == "gemini":
            # Silently return — Gemini cast, no noise needed
            pass
        return entry["xtts_speaker"]

    # ── Dormant character spoke unexpectedly mid-batch ────────────────────
    # Restore their voice slot so the pipeline doesn't fall back to narrator.
    dormant_voices = registry.get("dormant_voices", {})
    if name in dormant_voices:
        saved = dormant_voices.pop(name)
        voice = saved["xtts_speaker"]
        registry["characters"][name] = {
            "xtts_speaker": voice,
            "gender":       saved.get("gender", "unknown"),
            "cast_by":      "dormancy_restore",
        }
        save_registry(registry, novel_dir)
        print(f"  Dormant character '{name}' spoke — reactivated with voice {voice}.")
        print(f"    Run update_novel.py to update novel.md.")
        return voice

    # ── Alias matching ────────────────────────────────────────────────────
    # The parser often outputs short names (e.g. "christopher") while
    # speakers.json stores full keys (e.g. "christopher_hugh").
    # Try to find a unique registry key that starts with the short name.
    candidates = [
        key for key in registry["characters"]
        if key.startswith(name + "_") or key == name
    ]
    if len(candidates) == 1:
        matched_key = candidates[0]
        entry = registry["characters"][matched_key]
        print(f"  Alias: '{name}' → '{matched_key}' → {entry['xtts_speaker']}")
        return entry["xtts_speaker"]

    # Unknown character not in speakers.json — use narrator voice
    # instead of wasting a voice slot on an uncast character.
    # Run cast_voices.py to properly assign voices to new characters.
    narrator_voice = registry["characters"].get("narrator", {}).get("xtts_speaker", "Ana Florence")
    print(f"  Unknown character '{name}' → using narrator voice ({narrator_voice})")
    print(f"    Run cast_voices.py to assign a proper voice if this character is important.")

    return narrator_voice


def get_known_characters(registry: dict) -> list[str]:
    return list(registry["characters"].keys())


# ═══════════════════════════════════════════════════════════════════════════
#  MANUAL REASSIGNMENT UTILITY
# ═══════════════════════════════════════════════════════════════════════════

def reassign_speaker(
    novel_dir: str,
    character_key: str,
    new_voice: str
):
    """
    Manually reassign a voice for a character.
    Run this from a Python shell when you want to fix a bad assignment:

        from registry import reassign_speaker
        reassign_speaker("novels/shadow_slave", "idan", "Damien Black")
    """
    registry = load_registry(novel_dir)

    if character_key not in registry["characters"]:
        print(f"  '{character_key}' not found in registry.")
        return

    old_voice = registry["characters"][character_key]["xtts_speaker"]
    registry["characters"][character_key]["xtts_speaker"] = new_voice
    save_registry(registry, novel_dir)

    print(f"  Reassigned '{character_key}': {old_voice} → {new_voice}")