"""
novel_manager/update_novel.py

Arc update — run every 50–100 chapters or after a major arc concludes.
Automatically detects which chapters are new since the last update
and feeds only those to Gemini along with the current novel.md.

Functions:
    build_prompt(current_novel_md, chapters_text, ch_start, ch_end) — Build the update prompt.
    parse_chapter_range(arg)          — Parse chapter range argument.
    main()                            — CLI entry point.

Usage:
    python novel_manager/update_novel.py novels/shadow_slave
    python novel_manager/update_novel.py novels/shadow_slave --chapters 101-200
    python novel_manager/update_novel.py novels/shadow_slave --chapters 101-
"""

import sys
import os
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from novel_manager.gemini_client import call_gemini
from novel_manager.novel_utils   import (
    get_chapters_in_range, load_chapters_text,
    read_novel_md, write_novel_md, update_meta_field,
    get_last_updated_chapter, get_all_chapters,
    extract_character_keys
)


SYSTEM_PROMPT = """\
You are maintaining a novel.md file for an AI audiobook production pipeline.

I will give you:
1. The CURRENT novel.md (contains all analysis done so far)
2. NEW chapters to incorporate

Your job is to return a fully updated novel.md integrating the new chapters.

━━━ WHAT TO DO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

EXISTING CHARACTERS — update entries where new chapters provide new info:
  - Refine personality, voice_style, speech_patterns if clearer now
  - Append to arc_notes (do NOT overwrite — add new lines):
      Ch N: [what changed / what we learn]
  - Upgrade confidence if justified:
      sparse → partial  (castable, enough data)
      partial → complete (stable, fully characterized)
  - Update last_updated_chapter ONLY when a character has ACTUAL DIALOGUE
    (quoted speech) in a chapter. Mere name mentions do NOT count.
  - If a character's role significantly changes (e.g. rival becomes ally),
    update role field and note it in arc_notes

ALIAS MERGING — if a character gets a new name, title, or alias:
  - Do NOT create a separate character entry.
  - Update the EXISTING entry: add the alias to their personality or
    a new "aliases" line, and note the name change in arc_notes.
  - Example: if "Arabel" starts going by "Ara", keep the arabel_morgan
    entry and add "aliases: Ara" — do NOT create a new ### ara entry.

NEW CHARACTERS — add entries ONLY for characters with on-page dialogue:
  - Needs actual quoted dialogue — not just a name mention or action
  - Do NOT add entries for unknown/unnamed speakers ("a soldier",
    "the guard", "a voice") — the narrator will handle those
  - Start at confidence: sparse unless very substantial presence
  - Use same entry format as existing characters

OVERVIEW — extend if new world-building or tone info is revealed:
  - If tone shifts significantly, note the chapter where it happens

WORLD & TONE NOTES — extend with new lore, power system details, etc.

FACTIONS — add newly introduced factions, update existing ones

CHAPTER MAP — APPEND new chapter entries, never rewrite existing ones

━━━ CHARACTER PRUNING ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

To conserve voice slots, REMOVE character entries that meet ALL of these:
  1. They have had NO ACTUAL DIALOGUE (quoted speech) for 50+ consecutive
     chapters. Being mentioned by name does NOT save them from pruning —
     only dialogue counts.
  2. Their last_updated_chapter is 50+ chapters behind the LATEST chapter.
  3. They are NOT narrator or system.
  4. They were NOT introduced in the current batch of new chapters.

When removing a character:
  - Delete their entire ### entry from ## Characters
  - Do NOT add them elsewhere — they are fully removed
  - List every pruned key in a ## Pruned Characters section at the very
    bottom of the document, one per line, like:
      - character_key_name

If no characters are pruned, omit the ## Pruned Characters section.

━━━ WHAT NOT TO DO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✗ Do not remove or overwrite existing arc_notes
  ✗ Do not downgrade confidence levels
  ✗ Do not alter narrator or system entries
  ✗ Do not speculate about future plot
  ✗ Do not add characters who are only mentioned by name
  ✗ Do not create entries for unknown/unnamed speakers
  ✗ Do not create new entries for aliases — merge into existing entries
  ✗ Do not invent information not in the text
  ✗ Do not prune characters introduced in the current batch

━━━ OUTPUT FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return the complete updated novel.md from top to bottom as clean markdown.
Do not wrap in code fences. Do not add explanations outside the document.
Update last_updated_chapter and total_chapters_processed in the Meta section.

CHARACTER ENTRY FORMAT:
### [lowercase_underscore_key]
- confidence: [sparse / partial / complete]
- introduced_chapter: [original — never change this]
- role:
- gender:
- age:
- personality:
- voice_style:
- speech_patterns:
- arc_notes:
  Ch N: [development]
  Ch N: [next development]
- relationship_to_protagonist:
- casting_note:
- last_updated_chapter: N
"""


def build_prompt(
    current_novel_md: str,
    chapters_text: str,
    ch_start: int,
    ch_end: int
) -> str:
    return f"""{SYSTEM_PROMPT}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
CURRENT novel.md:
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{current_novel_md}

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
NEW CHAPTERS ({ch_start}–{ch_end}):
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
{chapters_text}
"""


def parse_chapter_range(arg: str) -> tuple[int, int | None]:
    if "-" in arg:
        parts = arg.split("-", 1)
        start = int(parts[0]) if parts[0] else 1
        end   = int(parts[1]) if parts[1] else None
        return start, end
    return int(arg), int(arg)


def prune_speakers_json(novel_dir: str, old_md: str, new_md: str):
    """
    Compare character keys before/after the Gemini update.
    Remove pruned characters from speakers.json to free voice slots.
    """
    old_keys = extract_character_keys(old_md)
    new_keys = extract_character_keys(new_md)
    pruned   = old_keys - new_keys - {"narrator", "system"}

    if not pruned:
        return

    speakers_path = os.path.join(novel_dir, "data", "speakers.json")
    if not os.path.exists(speakers_path):
        print(f"  ⚠ speakers.json not found — skipping voice cleanup")
        return

    with open(speakers_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    characters = registry.get("characters", {})
    freed = []
    for key in pruned:
        if key in characters:
            voice = characters[key].get("xtts_speaker", "?")
            del characters[key]
            freed.append((key, voice))

    if freed:
        with open(speakers_path, "w", encoding="utf-8") as f:
            json.dump(registry, f, indent=2, ensure_ascii=False)

        print(f"\n  {'─' * 50}")
        print(f"  �  Pruned {len(freed)} inactive character(s) from speakers.json:")
        for key, voice in sorted(freed):
            print(f"      {key:30s} → freed voice: {voice}")
        print(f"  {'─' * 50}")


def main():
    parser = argparse.ArgumentParser(
        description="Update novel.md with new chapters"
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

    print(f"\n{'═' * 55}")
    print(f"  Novel Manager — Arc Update")
    print(f"  Novel       : {novel_name}")
    print(f"  New chapters: {ch_nums[0]}–{ch_nums[-1]}  ({len(chapters)} files)")
    print(f"  Previously  : through chapter {last_done}")
    print(f"  Words       : {sum(len(open(f, encoding='utf-8').read().split()) for _, f in chapters):,}")
    print(f"{'═' * 55}")

    current_md    = read_novel_md(novel_dir)
    chapters_text = load_chapters_text(chapters)
    prompt        = build_prompt(current_md, chapters_text, ch_nums[0], ch_nums[-1])

    print(f"\n  Sending to Gemini...")
    result = call_gemini(prompt, label="update")

    result = update_meta_field(result, "last_updated_chapter", str(ch_nums[-1]))
    result = update_meta_field(result, "total_chapters_processed", str(total_done))

    write_novel_md(novel_dir, result)

    # ── Prune inactive characters from speakers.json ──────────
    prune_speakers_json(novel_dir, current_md, result)

    print(f"\n  ✓ novel.md updated through chapter {ch_nums[-1]}")
    print(f"  Backup saved in {os.path.join(novel_dir, 'data', 'novel_backups')}/")


if __name__ == "__main__":
    main()
