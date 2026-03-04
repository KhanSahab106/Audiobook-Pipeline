"""
novel_manager/add_character.py

Targeted character entry generator — run when the pipeline output shows
a new speaker key that isn't in novel.md yet and you want to characterize
them before their voice assignment is finalized.

Functions:
    build_prompt(character_key, overview_text, chapters_text, ch_start, ch_end) — Build the Gemini prompt.
    extract_overview(novel_md)        — Pull the Overview section from novel.md.
    character_already_in_novel_md(novel_md, character_key) — Check if character already exists.
    insert_character_entry(novel_md, entry) — Insert new entry into the Characters section.
    parse_chapter_range(arg)          — Parse chapter range argument.
    main()                            — CLI entry point.

Usage:
    python novel_manager/add_character.py novels/shadow_slave --character nephis
    python novel_manager/add_character.py novels/shadow_slave --character nephis --chapters 25-40
"""

import sys
import os
import re
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from novel_manager.gemini_client import call_gemini
from novel_manager.novel_utils   import (
    get_chapters_in_range, get_all_chapters,
    load_chapters_text, read_novel_md, write_novel_md,
    get_last_updated_chapter
)


SYSTEM_PROMPT = """\
You are helping maintain a novel.md file for an AI audiobook production pipeline.
A new character has appeared in the story and needs a character entry.

I will give you:
1. The novel's Overview section (for world context)
2. Chapters where this character appears

Your job is to write ONE character entry for novel.md.

━━━ RULES ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  - Only describe what is shown in the provided chapters
  - "unknown" is always acceptable — never guess
  - If the character has minimal presence, most fields should be "unknown"
    and confidence should be "sparse" — that is correct and expected
  - Keep arc_notes as a running log so future updates can append cleanly
  - casting_note should be practical voice quality hints, not actor names
    (e.g. "cold and precise, Eastern European accent would suit" not "like
    Mads Mikkelsen")

━━━ OUTPUT FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return ONLY the character entry block — nothing else.
No explanation, no surrounding text.

### [character_key]
- confidence: [sparse / partial]
- introduced_chapter: N
- role: [protagonist / antagonist / deuteragonist / major supporting / supporting / minor]
- gender: [male / female / unknown]
- age: [exact or estimate or "unknown"]
- personality: [2–4 sentences, factual only]
- voice_style: [how they speak — pace, volume, emotional register]
- speech_patterns: [notable quirks or "unknown"]
- arc_notes:
  Ch N: [what we learn about them in these chapters]
- relationship_to_protagonist: [1 sentence]
- casting_note: [voice quality hints]
- last_updated_chapter: N
"""


def build_prompt(
    character_key: str,
    overview_text: str,
    chapters_text: str,
    ch_start: int,
    ch_end: int
) -> str:
    return f"""{SYSTEM_PROMPT}

CHARACTER KEY TO ANALYZE: {character_key}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NOVEL OVERVIEW (world/tone context):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{overview_text}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CHAPTERS WHERE {character_key.upper()} APPEARS ({ch_start}–{ch_end}):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chapters_text}
"""


def extract_overview(novel_md: str) -> str:
    """Pull just the Overview section from novel.md"""
    match = re.search(
        r"## Overview\s*\n(.*?)(?=\n---|\n## )",
        novel_md,
        re.DOTALL
    )
    if match:
        return match.group(1).strip()
    return "(Overview section not found in novel.md)"


def character_already_in_novel_md(novel_md: str, character_key: str) -> bool:
    pattern = rf"^###\s+{re.escape(character_key)}\s*$"
    return bool(re.search(pattern, novel_md, re.MULTILINE))


def insert_character_entry(novel_md: str, entry: str) -> str:
    """
    Insert the new character entry at the end of the Characters section,
    just before the Factions section.
    """
    # Strip code fences if present
    entry = re.sub(r"^```(?:markdown)?\s*\n", "", entry, flags=re.MULTILINE)
    entry = re.sub(r"\n```\s*$", "", entry, flags=re.MULTILINE)
    entry = entry.strip()

    # Find insertion point: just before ## Factions
    insert_marker = "\n---\n\n## Factions"
    if insert_marker in novel_md:
        return novel_md.replace(
            insert_marker,
            f"\n\n{entry}{insert_marker}"
        )

    # Fallback: append to end
    return novel_md + f"\n\n{entry}\n"


def parse_chapter_range(arg: str) -> tuple[int, int | None]:
    if "-" in arg:
        parts = arg.split("-", 1)
        start = int(parts[0]) if parts[0] else 1
        end   = int(parts[1]) if parts[1] else None
        return start, end
    return int(arg), int(arg)


def main():
    parser = argparse.ArgumentParser(
        description="Add a new character entry to novel.md"
    )
    parser.add_argument(
        "novel_dir",
        help="Path to novel directory, e.g. novels/shadow_slave"
    )
    parser.add_argument(
        "--character",
        required=True,
        help="Character key as used by the parser, e.g. nephis"
    )
    parser.add_argument(
        "--chapters",
        default=None,
        help=(
            "Chapter range where character appears, e.g. 25-40.\n"
            "If omitted, searches all chapters processed so far."
        )
    )
    args = parser.parse_args()

    novel_dir     = os.path.normpath(args.novel_dir)
    novel_name    = os.path.basename(novel_dir)
    character_key = args.character.lower().strip()

    # Check if already in novel.md
    novel_md = read_novel_md(novel_dir)
    if character_already_in_novel_md(novel_md, character_key):
        print(f"\n  '{character_key}' already has an entry in novel.md.")
        print(f"  Use update_novel.py to update existing entries.")
        sys.exit(0)

    # Determine chapter range
    last_done = get_last_updated_chapter(novel_dir)
    if args.chapters:
        start, end = parse_chapter_range(args.chapters)
    else:
        start, end = 1, last_done if last_done > 0 else None

    chapters = get_chapters_in_range(novel_dir, start, end)

    if not chapters:
        print(f"\n  ✗ No chapters found in range.")
        sys.exit(1)

    ch_nums = [n for n, _ in chapters]

    print(f"\n{'═' * 55}")
    print(f"  Novel Manager — New Character")
    print(f"  Novel     : {novel_name}")
    print(f"  Character : {character_key}")
    print(f"  Chapters  : {ch_nums[0]}–{ch_nums[-1]}  ({len(chapters)} files)")
    print(f"{'═' * 55}")

    overview_text = extract_overview(novel_md)
    chapters_text = load_chapters_text(chapters)
    prompt        = build_prompt(
        character_key,
        overview_text,
        chapters_text,
        ch_nums[0],
        ch_nums[-1]
    )

    print(f"\n  Sending to Gemini...")
    entry = call_gemini(prompt, label=f"char-{character_key}")

    # Preview entry
    print(f"\n  ── Generated entry ──────────────────────────────")
    print(entry)
    print(f"  ─────────────────────────────────────────────────")

    # Confirm before writing
    confirm = input("\n  Insert this entry into novel.md? [y/N]: ").strip().lower()
    if confirm != "y":
        print("  Aborted — novel.md not modified.")

        # Save entry to data/ so it's not lost
        draft_path = os.path.join(
            novel_dir, "data", f"character_draft_{character_key}.md"
        )
        os.makedirs(os.path.dirname(draft_path), exist_ok=True)
        with open(draft_path, "w", encoding="utf-8") as f:
            f.write(entry)
        print(f"  Draft saved to: {draft_path}")
        sys.exit(0)

    updated_md = insert_character_entry(novel_md, entry)
    write_novel_md(novel_dir, updated_md)

    print(f"\n  ✓ '{character_key}' added to novel.md")


if __name__ == "__main__":
    main()
