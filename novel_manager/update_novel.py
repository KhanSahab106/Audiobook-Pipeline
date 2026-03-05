"""
novel_manager/update_novel.py

Arc update — run every 50–100 chapters or after a major arc concludes.
Automatically detects which chapters are new since the last update
and feeds only those to Gemini along with the current novel.md.

Functions:
    build_prompt(current_novel_md, chapters_text, ch_start, ch_end) — Build the update prompt.
    sync_speakers_json(novel_dir, old_md, new_md) — Sync dormant/reactivated/pruned voices.
    prune_speakers_json(novel_dir, old_md, new_md) — Backward-compatible alias for sync_speakers_json.
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
    extract_character_keys, extract_dormant_characters,
    extract_newly_dormant, extract_reactivated, parse_chapter_range
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

━━━ CHARACTER DORMANCY ━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Every character entry MUST have a `- status: active` or `- status: dormant` line.

Mark a character as `status: dormant` when they exceed their tier's inactivity
threshold (chapters since their last_updated_chapter with actual dialogue):

  protagonist            → NEVER dormant
  deuteragonist (S-Tier) → 200 chapters without dialogue
  antagonist (S-Tier)    → 200 chapters without dialogue
  major supporting (A-Tier) → 150 chapters without dialogue
  supporting (B-Tier)    → 100 chapters without dialogue
  minor (C-Tier)         → 50 chapters without dialogue
  unspecified / unknown  → 30 chapters without dialogue

Rules:
  - Use last_updated_chapter (dialogue-only) to count silent chapters
  - Reactivate a character (status: active) if they have dialogue in the
    new chapters — even if they were previously dormant
  - NEVER mark narrator or system as dormant
  - NEVER mark characters introduced in the current batch as dormant
  - When marking dormant, append to arc_notes:
      Ch N: marked dormant (no dialogue since Ch X)
  - When reactivating, append to arc_notes:
      Ch N: reactivated
  - Characters dormant for 2× their tier threshold (truly gone) may be moved
    to a ## Pruned Characters section at the bottom:
      - character_key_name
    Only use this for characters clearly finished with the story.
  - If no characters are pruned, omit ## Pruned Characters entirely.

━━━ WHAT NOT TO DO ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  ✗ Do not remove or overwrite existing arc_notes
  ✗ Do not downgrade confidence levels
  ✗ Do not alter narrator or system entries
  ✗ Do not speculate about future plot
  ✗ Do not add characters who are only mentioned by name
  ✗ Do not create entries for unknown/unnamed speakers
  ✗ Do not create new entries for aliases — merge into existing entries
  ✗ Do not invent information not in the text
  ✗ Do not mark dormant characters introduced in the current batch
  ✗ Do not mark narrator or system as dormant

━━━ OUTPUT FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return the complete updated novel.md from top to bottom as clean markdown.
Do not wrap in code fences. Do not add explanations outside the document.
Update last_updated_chapter and total_chapters_processed in the Meta section.

CHARACTER ENTRY FORMAT:
### [lowercase_underscore_key]
- confidence: [sparse / partial / complete]
- status: [active / dormant]
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


def sync_speakers_json(novel_dir: str, old_md: str, new_md: str):
    """
    Sync speakers.json after a novel.md update.

    Handles three cases:
    1. Newly dormant characters — free voice slot, save to dormant_voices.
    2. Reactivated characters   — restore from dormant_voices (or warn).
    3. Fully pruned characters  — remove from both characters and dormant_voices.
    """
    speakers_path = os.path.join(novel_dir, "data", "speakers.json")
    if not os.path.exists(speakers_path):
        print(f"  ⚠ speakers.json not found — skipping voice sync")
        return

    with open(speakers_path, "r", encoding="utf-8") as f:
        registry = json.load(f)

    characters     = registry.get("characters", {})
    dormant_voices = registry.setdefault("dormant_voices", {})

    # ── 0. Catch-up: characters already dormant but still in characters ──────
    all_dormant_in_new = extract_dormant_characters(new_md) - {"narrator", "system"}
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
    newly_dormant = extract_newly_dormant(old_md, new_md) - {"narrator", "system"}
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
    reactivated = extract_reactivated(old_md, new_md) - {"narrator", "system"}
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

    # ── 3. Fully pruned ─────────────────────────────────────────────────────
    old_keys = extract_character_keys(old_md)
    new_keys = extract_character_keys(new_md)
    pruned   = old_keys - new_keys - {"narrator", "system"}
    removed  = []
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
            json.dump(registry, f, indent=2, ensure_ascii=False)

    if freed:
        print(f"\n  {'─' * 50}")
        print(f"  → Freed {len(freed)} voice slot(s) for dormant character(s):")
        for key, voice in sorted(freed):
            print(f"      {key:30s} → dormant_voices (was: {voice})")
        print(f"  {'─' * 50}")

    if restored:
        print(f"\n  {'─' * 50}")
        print(f"  ✓ Restored {len(restored)} reactivated character(s):")
        for key, voice in sorted(restored):
            print(f"      {key:30s} → {voice}")
        print(f"  {'─' * 50}")

    if needs_cast:
        print(f"\n  {'─' * 50}")
        print(f"  ⚠ {len(needs_cast)} reactivated character(s) need voice assignment:")
        for key in sorted(needs_cast):
            print(f"      {key}")
        print(f"    Run cast_voices.py to assign voices.")
        print(f"  {'─' * 50}")

    if removed:
        print(f"\n  {'─' * 50}")
        print(f"  ✗ Removed {len(removed)} fully pruned character(s):")
        for key in sorted(removed):
            print(f"      {key}")
        print(f"  {'─' * 50}")


def prune_speakers_json(novel_dir: str, old_md: str, new_md: str):
    """Backward-compatible wrapper — delegates to sync_speakers_json."""
    sync_speakers_json(novel_dir, old_md, new_md)


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

    # ── Sync dormant/reactivated/pruned characters in speakers.json ──────────
    sync_speakers_json(novel_dir, current_md, result)

    print(f"\n  ✓ novel.md updated through chapter {ch_nums[-1]}")
    print(f"  Backup saved in {os.path.join(novel_dir, 'data', 'novel_backups')}/")


if __name__ == "__main__":
    main()
