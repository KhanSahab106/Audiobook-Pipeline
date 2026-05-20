"""
novel_manager/audit_characters.py

Audit novel.md for problematic character entries that waste voice slots:

  1. BATTLE-GROUP NPCs — unnamed or numbered enemies/guards occupying slots
     (e.g. "wormskinner_fighter_1", "guard_2", "unnamed_soldier")
  2. GHOST characters — no actual dialogue evidence in arc_notes
     (arc_notes only contain "mentioned", "present", "observed" etc.)
  3. FABRICATED LUC — last_updated_chapter matches a suspiciously round
     number that looks like Gemini set it to the batch end, not real dialogue

Run in audit mode to review, then re-run with --remove to clean up.

Usage:
    python novel_manager/audit_characters.py novels/shs_and_sws
    python novel_manager/audit_characters.py novels/shs_and_sws --remove battle_group
    python novel_manager/audit_characters.py novels/shs_and_sws --remove ghost
    python novel_manager/audit_characters.py novels/shs_and_sws --remove all
    python novel_manager/audit_characters.py novels/shs_and_sws --remove-key wormskinner_fighter_1
"""

import sys
import os
import re
import json
import argparse
import shutil
from datetime import datetime

# Force UTF-8 output on Windows (PowerShell defaults to cp1252)
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from novel_manager.novel_utils import read_novel_md, write_novel_md

# ── Battle-group name patterns ────────────────────────────────────────────────
# Keys matching these are auto-classified as NPCs — they're role+number combos
# that clearly represent disposable scene characters, not recurring cast.
BATTLE_GROUP_PATTERNS = [
    r"\d+$",                         # ends with a number: fighter_1, guard_2
    r"_(leader|chief|boss)$",        # generic authority: scout_leader
    r"_(fighter|guard|soldier|warrior|scout|lieutenant|subordinate|archer|knight|thug|bandit|mercenary|spy|assassin|minion|grunt|lackey|mook|henchman)s?(_\w+)?$",
    r"^(unnamed|unknown|mystery|random|unnamed_\w+)",
    r"_(left|right|north|south|east|west)$",   # positional: subordinate_left
    r"_(platinum|gold|silver|bronze|diamond|ruby|iron)\b",  # rank+role
    r"^(guard|soldier|fighter|warrior|scout|lieutenant|subordinate|archer|knight|thug|bandit|mercenary)s?(_|$)",
]

# Arc-notes phrases that indicate MENTION only (no real dialogue)
NO_DIALOGUE_PHRASES = [
    "mentioned",
    "observed",
    "present",
    "in attendance",
    "seen",
    "noted",
    "referenced",
    "is noted",
    "appears briefly",
    "briefly seen",
]

# ─────────────────────────────────────────────────────────────────────────────


def _is_battle_group(key: str) -> bool:
    """Return True if key looks like a disposable NPC battle-group character."""
    k = key.lower()
    return any(re.search(p, k) for p in BATTLE_GROUP_PATTERNS)


def _has_real_dialogue(arc_notes: str) -> bool:
    """
    Return True if arc_notes contain evidence of actual dialogue.
    Heuristic: if every arc entry only uses no-dialogue phrases, return False.
    """
    if not arc_notes.strip():
        return False

    # Split into individual chapter entries
    entries = re.split(r'\n\s*Ch \d+:', arc_notes)
    entries = [e.strip().lower() for e in entries if e.strip()]

    if not entries:
        return False

    # If ALL entries are mention-only, there's no real dialogue
    dialogue_count = 0
    for entry in entries:
        is_mention_only = any(phrase in entry for phrase in NO_DIALOGUE_PHRASES)
        if not is_mention_only:
            dialogue_count += 1

    return dialogue_count > 0


def _parse_all_characters(novel_md: str) -> dict[str, dict]:
    """
    Parse every character entry from ## Characters section.
    Returns {key: {status, confidence, role, luc, arc_notes_raw, block_start, block_end}}.
    """
    chars_match = re.search(r'^## Characters\s*\n', novel_md, re.MULTILINE)
    if not chars_match:
        return {}

    after_chars = novel_md[chars_match.end():]
    next_section = re.search(r'^## ', after_chars, re.MULTILINE)
    chars_section_end = (chars_match.end() + next_section.start()) if next_section else len(novel_md)
    chars_section = novel_md[chars_match.end():chars_section_end]

    # Split on ### headings
    heading_pat = re.compile(r'^### (\S+)\s*$', re.MULTILINE)
    headings = [(m.start(), m.group(1)) for m in heading_pat.finditer(chars_section)]

    entries = {}
    for i, (pos, key) in enumerate(headings):
        end = headings[i + 1][0] if i + 1 < len(headings) else len(chars_section)
        body = chars_section[pos:end]

        # Absolute positions in novel_md
        abs_start = chars_match.end() + pos
        abs_end   = chars_match.end() + end

        def _field(pattern, text=body):
            m = re.search(pattern, text, re.IGNORECASE)
            return m.group(1).strip() if m else ""

        # Extract arc_notes block
        arc_m = re.search(r'-\s*arc_notes:\s*\n(.*?)(?=^- \w|\Z)', body, re.MULTILINE | re.DOTALL)
        arc_notes_raw = arc_m.group(1) if arc_m else ""

        luc_str = _field(r'-\s*last_updated_chapter:\s*(\d+)')
        luc = int(luc_str) if luc_str.isdigit() else None

        entries[key] = {
            "status":     _field(r'-\s*status:\s*(\w+)'),
            "confidence": _field(r'-\s*confidence:\s*(\w+)'),
            "role":       _field(r'-\s*role:\s*(.+)'),
            "luc":        luc,
            "introduced": _field(r'-\s*introduced_chapter:\s*(\d+)'),
            "arc_notes":  arc_notes_raw,
            "block_start": abs_start,
            "block_end":   abs_end,
        }

    return entries


def _remove_character_from_novel_md(novel_md: str, key: str) -> str:
    """Remove a character's ### block from novel.md."""
    block_pat = re.compile(rf'^### {re.escape(key)}\s*$', re.MULTILINE)
    m = block_pat.search(novel_md)
    if not m:
        return novel_md

    after = novel_md[m.end():]
    next_block = re.search(r'^(?:###|##) ', after, re.MULTILINE)
    block_end = m.end() + next_block.start() if next_block else len(novel_md)
    novel_md = novel_md[:m.start()] + novel_md[block_end:]
    return novel_md


def _remove_from_speakers_json(novel_dir: str, keys: list[str]) -> int:
    """Remove character keys from speakers.json. Returns count removed."""
    speakers_path = os.path.join(novel_dir, "data", "speakers.json")
    if not os.path.exists(speakers_path):
        return 0

    with open(speakers_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    removed = 0
    chars = registry.get("characters", {})
    for key in keys:
        if key in chars:
            del chars[key]
            removed += 1

    if removed:
        registry["characters"] = chars
        with open(speakers_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=True)

    return removed


def audit(novel_dir: str, luc_suspect: int = None) -> dict[str, list[str]]:
    """
    Run the full audit. Returns categorized character keys.

    luc_suspect: if provided, flag all active characters whose LUC == this value
                 (typically the last batch end chapter, e.g. 350).
    """
    novel_md = read_novel_md(novel_dir)
    entries  = _parse_all_characters(novel_md)

    ALWAYS_KEEP = {"narrator", "system", "unknown"}

    battle_group = []
    ghost        = []
    suspicious   = []

    for key, info in entries.items():
        if key in ALWAYS_KEEP:
            continue

        # Category 1: Battle-group NPC
        if _is_battle_group(key):
            battle_group.append(key)
            continue  # no need to check further

        # Category 2: Ghost — no real dialogue in arc_notes
        if not _has_real_dialogue(info["arc_notes"]):
            ghost.append(key)
            continue

        # Category 3: Suspicious LUC — LUC matches the batch-end and introduced != LUC
        if luc_suspect and info["luc"] == luc_suspect:
            intro = int(info["introduced"]) if info["introduced"].isdigit() else 0
            if info["luc"] != intro:  # not a character first introduced in that chapter
                suspicious.append(key)

    return {
        "battle_group": sorted(battle_group),
        "ghost":        sorted(ghost),
        "suspicious":   sorted(suspicious),
    }


def print_report(novel_dir: str, categories: dict[str, list[str]], entries: dict):
    SYMBOLS = {
        "battle_group": "⚔",
        "ghost":        "👻",
        "suspicious":   "⚠",
    }
    LABELS  = {
        "battle_group": "Battle-group NPCs (disposable, auto-detected by name pattern)",
        "ghost":        "Ghost characters (no dialogue evidence in arc_notes)",
        "suspicious":   "Suspicious LUC (last_updated_chapter looks fabricated)",
    }
    total = sum(len(v) for v in categories.values())

    print(f"\n{'═' * 60}")
    print(f"  Character Audit — {os.path.basename(novel_dir)}")
    print(f"  Total flagged: {total}")
    print(f"{'═' * 60}")

    for cat, keys in categories.items():
        if not keys:
            continue
        print(f"\n  {SYMBOLS[cat]}  {LABELS[cat]} ({len(keys)})")
        print(f"  {'─' * 56}")
        for key in keys:
            info = entries.get(key, {})
            luc  = info.get("luc", "?")
            role = info.get("role", "?")[:25]
            print(f"    {key:35s}  luc={luc:>4}  role={role}")

    print(f"\n{'═' * 60}")
    print(f"  To remove a category:")
    print(f"    python novel_manager/audit_characters.py {novel_dir} --remove battle_group")
    print(f"    python novel_manager/audit_characters.py {novel_dir} --remove ghost")
    print(f"    python novel_manager/audit_characters.py {novel_dir} --remove all")
    print(f"  To remove a single key:")
    print(f"    python novel_manager/audit_characters.py {novel_dir} --remove-key <key>")
    print(f"{'═' * 60}\n")


def remove_characters(novel_dir: str, keys: list[str], dry_run: bool = False) -> None:
    """Remove characters from novel.md and speakers.json."""
    if not keys:
        print("  Nothing to remove.")
        return

    novel_md = read_novel_md(novel_dir)

    # Backup novel.md first
    backup_dir = os.path.join(novel_dir, "data", "novel_backups")
    os.makedirs(backup_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_path = os.path.join(backup_dir, f"novel_pre_audit_{ts}.md")
    shutil.copy2(os.path.join(novel_dir, "novel.md"), backup_path)
    print(f"  Backup saved: {backup_path}")

    removed_md = 0
    for key in keys:
        new_md = _remove_character_from_novel_md(novel_md, key)
        if new_md != novel_md:
            removed_md += 1
            print(f"  - Removed from novel.md: {key}")
            novel_md = new_md
        else:
            print(f"  ! Not found in novel.md: {key}")

    if not dry_run:
        write_novel_md(novel_dir, novel_md)
        removed_json = _remove_from_speakers_json(novel_dir, keys)
        print(f"\n  ✓ Removed {removed_md} character(s) from novel.md")
        print(f"  ✓ Removed {removed_json} voice slot(s) from speakers.json")
    else:
        print(f"\n  [dry-run] Would remove {removed_md} from novel.md")


def main():
    parser = argparse.ArgumentParser(
        description="Audit novel.md for ghost and battle-group characters"
    )
    parser.add_argument("novel_dir", help="Path to novel directory (e.g. novels/shs_and_sws)")
    parser.add_argument(
        "--remove", metavar="CATEGORY",
        choices=["battle_group", "ghost", "all"],
        help="Remove characters in this category: battle_group / ghost / all"
    )
    parser.add_argument(
        "--remove-key", metavar="KEY",
        help="Remove a specific character key from novel.md and speakers.json"
    )
    parser.add_argument(
        "--luc-suspect", type=int, default=None,
        help="Flag active characters whose last_updated_chapter equals this value "
             "(e.g. 350 if all LUCs suspiciously equal 350)"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be removed without writing files")

    args = parser.parse_args()
    novel_dir = args.novel_dir

    # ── Single key removal ───────────────────────────────────────────────────
    if args.remove_key:
        remove_characters(novel_dir, [args.remove_key], dry_run=args.dry_run)
        return

    # ── Audit ────────────────────────────────────────────────────────────────
    novel_md  = read_novel_md(novel_dir)
    entries   = _parse_all_characters(novel_md)
    categories = audit(novel_dir, luc_suspect=args.luc_suspect)
    print_report(novel_dir, categories, entries)

    # ── Bulk removal ─────────────────────────────────────────────────────────
    if args.remove:
        if args.remove == "all":
            to_remove = categories["battle_group"] + categories["ghost"]
        else:
            to_remove = categories[args.remove]

        if not to_remove:
            print(f"  No characters in category '{args.remove}' to remove.")
            return

        print(f"\n  Removing {len(to_remove)} character(s)...")
        remove_characters(novel_dir, to_remove, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
