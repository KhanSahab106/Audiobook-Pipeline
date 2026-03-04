"""
novel_manager/init_novel.py

Initial novel.md population — run ONCE before the first batch.

DEFAULT MODE (text):
    Reads local chapter files from input/ and sends them to Gemini.
    Accurate, based on actual chapter content — always works.

Functions:
    _output_format_instructions()     — Return the standard character-entry format for prompts.
    _measure_coverage_quality(novel_md_content) — Count unknown vs filled character fields.
    _has_real_characters(novel_md_content) — Check if any non-narrator characters exist.
    _parse_chapter_arg(arg)           — Parse '10', '1-20', '5-' style chapter arguments.
    _chapter_range_label(arg)         — Human-readable label like '1–100'.
    run_web_mode(novel_dir, novel_name, chapter_range, template) — Research novel online via Gemini + Search.
    run_text_mode(novel_dir, chapters_arg, template) — Analyse local chapter text files via Gemini.
    main()                            — CLI entry point.

WEB MODE (--mode web):
    Gemini searches the internet for character info, plot summaries,
    fandom wikis, and reviews. No chapter text needed.
    If web coverage is poor (too many unknowns), automatically
    falls back to text mode using local chapters.
    Use for popular novels with strong online presence.

Usage:
    python novel_manager/init_novel.py novels/shadow_slave
    python novel_manager/init_novel.py novels/shadow_slave --chapters 1-20
    python novel_manager/init_novel.py novels/shadow_slave --mode web --chapters 1-100
    python novel_manager/init_novel.py novels/shadow_slave --mode web --novel-name "Shadow Slave by Guiltythree"
"""

import sys
import os
import re
import argparse

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from novel_manager.gemini_client import call_gemini, call_gemini_with_search
from novel_manager.novel_utils   import (
    get_chapters_in_range, load_chapters_text,
    read_novel_md, write_novel_md, update_meta_field,
    get_last_updated_chapter
)

# ── Config ────────────────────────────────────────────────────────────────────
DEFAULT_TEXT_CHAPTERS   = 10     # chapters to read if --mode text without --chapters
POOR_COVERAGE_THRESHOLD = 0.40   # if >40% of character fields are "unknown" → poor
# ─────────────────────────────────────────────────────────────────────────────


# ═══════════════════════════════════════════════════════════════════════════
#  PROMPTS
# ═══════════════════════════════════════════════════════════════════════════

def _output_format_instructions() -> str:
    return """\
━━━ OUTPUT FORMAT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Return a complete filled novel.md as clean markdown.
Do NOT wrap it in code fences.
Do NOT add explanations outside the document.
Preserve all comment blocks (<!-- -->) from the template.
Update last_updated_chapter and total_chapters_processed in Meta.

CHARACTER ENTRY FORMAT:
### [lowercase_underscore_key]
- status: [active / dormant]
- confidence: [sparse / partial / complete]
- introduced_chapter: [number or "unknown"]
- role: [protagonist / antagonist / deuteragonist / major supporting / supporting / minor]
- gender: [male / female / unknown]
- age: [exact or estimate or "unknown"]
- personality: [2–4 sentences — how they think, behave, react]
- voice_style: [how they speak — pace, volume, emotional register, warmth/coldness]
- speech_patterns: [notable quirks: formal, blunt, sarcastic, talks little, etc. or "unknown"]
- arc_notes:
  Ch N: [what happens to them / what we learn]
- relationship_to_protagonist: [1 sentence or "IS the protagonist"]
- casting_note: [voice quality hints: deep/soft/rough/warm/cold/young/old/accented]
- last_updated_chapter: [chapter number or "unknown"]

Rules:
- "unknown" is always acceptable — never invent
- narrator and system entries must remain exactly as-is
- Character keys: lowercase with underscores, no spaces
"""


WEB_PROMPT_TEMPLATE = """\
You are an expert literary analyst setting up an AI audiobook production pipeline.

I need you to research the web novel "{novel_name}" and produce a filled novel.md
for voice casting and tone calibration.

Use Google Search to find:
- Official novel pages (WebNovel, Chereads, Scribble Hub, Royal Road, etc.)
- Fandom wikis or character pages
- Reader reviews, chapter discussions, plot summaries
- Any available character descriptions, personality analyses, or fan discussions

Focus your research on chapters {chapter_range}.

━━━ WHAT TO FILL IN ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

OVERVIEW section:
- Genre and subgenre (be specific)
- Setting, world, key locations
- Overall tone of the story
- Narrative perspective
- Distinctive stylistic features (system notifications, game elements, etc.)

WORLD & TONE NOTES:
- Power/magic/game system if present — how it affects character behaviour
- Social class distinctions and their effect on speech
- Whether inhuman voices are needed (AI systems, spirits, gods, etc.)
- Overall speech register (formal, casual, coarse, regional)

CHARACTERS:
Include ALL named characters who appear in chapters {chapter_range}.
For each character set confidence based on how much reliable info you found:
  complete = detailed personality and voice info found
  partial  = some info, enough for casting
  sparse   = name and role only, little else known

FACTIONS:
All major groups or organizations active in chapters {chapter_range}.

CHAPTER MAP:
If chapter summaries are available online, include them.
Otherwise write "unknown" for chapters you have no data for.

━━━ IMPORTANT ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

- If you cannot find reliable information about a character, write "unknown"
  for that field — do not guess or hallucinate
- Mark any field you are uncertain about with "(inferred)" so it can be
  manually reviewed
- narrator and system entries must remain exactly as-is in the template

{output_format}

━━━ CURRENT novel.md TEMPLATE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{novel_md_template}
"""


TEXT_PROMPT_TEMPLATE = """\
You are an expert literary analyst setting up an AI audiobook production pipeline.

I will give you:
1. The current novel.md template (may already have partial data from a web search)
2. Chapter text to analyze

Your job is to fill in or improve the novel.md using the provided chapters.

CRITICAL RULES:
- Only describe what is explicitly shown in the provided chapters
- Do NOT guess or speculate about events not in the text
- Do NOT add characters who are only mentioned by name — need on-page presence
- If a field already has a value from a previous web search, keep it unless
  the chapter text contradicts it
- "unknown" is always acceptable and preferred over guessing
- Never alter narrator or system entries

{output_format}

━━━ CURRENT novel.md (may have partial data) ━━━━━━━━━━━━━━━

{novel_md_template}

━━━ CHAPTERS TO ANALYZE ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

{chapters_text}
"""


# ═══════════════════════════════════════════════════════════════════════════
#  COVERAGE QUALITY CHECK
# ═══════════════════════════════════════════════════════════════════════════

def _measure_coverage_quality(novel_md_content: str) -> tuple[float, int, int]:
    """
    Count how many character field values are "unknown" vs filled.
    Returns (unknown_ratio, unknown_count, total_fields).

    Only looks at character entries (between ### name and next ### or ---).
    """
    # Extract just the Characters section
    chars_match = re.search(
        r"## Characters\s*\n(.*?)(?=\n## |\Z)",
        novel_md_content,
        re.DOTALL
    )
    if not chars_match:
        return 1.0, 0, 0

    chars_section = chars_match.group(1)

    # Count fields of the form "- fieldname: value"
    all_fields     = re.findall(r"^\s*-\s+\w[\w_]*\s*:", chars_section, re.MULTILINE)
    unknown_fields = re.findall(r"^\s*-\s+\w[\w_]*\s*:\s*unknown\s*$", chars_section, re.MULTILINE | re.IGNORECASE)

    # Exclude fixed fields (confidence, introduced_chapter — always set)
    total   = len(all_fields)
    unknown = len(unknown_fields)

    ratio = unknown / total if total > 0 else 1.0
    return ratio, unknown, total


def _has_real_characters(novel_md_content: str) -> bool:
    """Check if any non-narrator/system characters were added."""
    entries = re.findall(r"^###\s+(\S+)", novel_md_content, re.MULTILINE)
    real    = [e for e in entries if e.lower() not in ("narrator", "system")]
    return len(real) > 0


# ═══════════════════════════════════════════════════════════════════════════
#  CHAPTER ARG PARSING
# ═══════════════════════════════════════════════════════════════════════════

def _parse_chapter_arg(arg: str) -> tuple[int, int | None]:
    """Parse '10', '1-20', '1-100', '5-' style arguments."""
    if "-" in arg:
        parts = arg.split("-", 1)
        start = int(parts[0]) if parts[0] else 1
        end   = int(parts[1]) if parts[1] else None
        return start, end
    return 1, int(arg)


def _chapter_range_label(arg: str) -> str:
    """Human-readable label like '1–100' or '1–10'."""
    start, end = _parse_chapter_arg(arg)
    return f"{start}–{end}" if end else f"{start}–latest"


# ═══════════════════════════════════════════════════════════════════════════
#  WEB MODE
# ═══════════════════════════════════════════════════════════════════════════

def run_web_mode(
    novel_dir:    str,
    novel_name:   str,
    chapter_range: str,
    template:     str
) -> tuple[str, bool]:
    """
    Run Gemini with Google Search grounding to research the novel online.

    Returns:
        (result_md, is_good_coverage)
    """
    prompt = WEB_PROMPT_TEMPLATE.format(
        novel_name        = novel_name,
        chapter_range     = chapter_range,
        output_format     = _output_format_instructions(),
        novel_md_template = template,
    )

    print(f"\n  Searching for: \"{novel_name}\" chapters {chapter_range}")
    result, sources = call_gemini_with_search(prompt, label="web-init")

    # Strip code fences if Gemini wrapped the output
    result = re.sub(r"^```(?:markdown)?\s*\n", "", result, flags=re.MULTILINE)
    result = re.sub(r"\n```\s*$",              "", result, flags=re.MULTILINE)
    result = result.strip()

    # Measure coverage quality
    ratio, unknown_count, total_fields = _measure_coverage_quality(result)
    has_chars = _has_real_characters(result)

    print(f"\n  Web coverage: {(1-ratio)*100:.0f}%  "
          f"({total_fields - unknown_count}/{total_fields} fields filled)")

    if not has_chars:
        print(f"  ⚠ No characters found — novel may not have online coverage")
        is_good = False
    elif ratio > POOR_COVERAGE_THRESHOLD:
        print(f"  ⚠ Coverage below threshold ({POOR_COVERAGE_THRESHOLD*100:.0f}%) "
              f"— will supplement with chapter text")
        is_good = False
    else:
        print(f"  ✓ Good coverage — no text fallback needed")
        is_good = True

    return result, is_good


# ═══════════════════════════════════════════════════════════════════════════
#  TEXT MODE
# ═══════════════════════════════════════════════════════════════════════════

def run_text_mode(
    novel_dir:   str,
    chapters_arg: str,
    template:    str
) -> tuple[str, list[int]]:
    """
    Run Gemini on local chapter text files.

    Returns:
        (result_md, chapter_numbers_processed)
    """
    start, end = _parse_chapter_arg(chapters_arg)
    chapters   = get_chapters_in_range(novel_dir, start, end)

    if not chapters:
        print(f"\n  ✗ No chapter files found in range {chapters_arg}")
        print(f"    in {os.path.join(novel_dir, 'input')}/")
        sys.exit(1)

    ch_nums       = [n for n, _ in chapters]
    chapters_text = load_chapters_text(chapters)
    total_words   = sum(len(open(f).read().split()) for _, f in chapters)

    print(f"\n  Reading chapters {ch_nums[0]}–{ch_nums[-1]} "
          f"({len(chapters)} files, {total_words:,} words)")

    prompt = TEXT_PROMPT_TEMPLATE.format(
        output_format     = _output_format_instructions(),
        novel_md_template = template,
        chapters_text     = chapters_text,
    )

    result = call_gemini(prompt, label="text-init")

    result = re.sub(r"^```(?:markdown)?\s*\n", "", result, flags=re.MULTILINE)
    result = re.sub(r"\n```\s*$",              "", result, flags=re.MULTILINE)
    result = result.strip()

    return result, ch_nums


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Initialize novel.md — web search mode by default"
    )
    parser.add_argument(
        "novel_dir",
        help="Path to novel directory, e.g. novels/shadow_slave"
    )
    parser.add_argument(
        "--mode",
        choices=["web", "text"],
        default="text",
        help="'text' = read local chapters (default). 'web' = Gemini searches online."
    )
    parser.add_argument(
        "--novel-name",
        default=None,
        help=(
            "Full novel title for web search, e.g. 'Shadow Slave by Guiltythree'.\n"
            "Defaults to the novel directory name (underscores → spaces)."
        )
    )
    parser.add_argument(
        "--chapters",
        default=None,
        help=(
            "Chapter range context.\n"
            "Web mode: tells Gemini which chapters to focus on, e.g. '1-100'.\n"
            "Text mode: which local files to read, e.g. '10' or '1-20'."
        )
    )
    parser.add_argument(
        "--no-fallback",
        action="store_true",
        help="Disable automatic text fallback when web coverage is poor."
    )
    args = parser.parse_args()

    novel_dir  = os.path.normpath(args.novel_dir)
    novel_name = args.novel_name or os.path.basename(novel_dir).replace("_", " ").title()

    # Guard: don't overwrite an already-initialized novel.md
    last = get_last_updated_chapter(novel_dir)
    if last > 0:
        print(f"\n  ✗ novel.md already initialized (last_updated_chapter: {last})")
        print(f"  Use update_novel.py to add new chapters instead.")
        print(f"  To reinitialize: set last_updated_chapter: 0 in novel.md")
        sys.exit(1)

    template = read_novel_md(novel_dir)

    print(f"\n{'═' * 58}")
    print(f"  Novel Manager — Init")
    print(f"  Novel : {novel_name}")
    print(f"  Mode  : {args.mode}  {'(local chapters)' if args.mode == 'text' else '(web search)'}")
    print(f"{'═' * 58}")

    result        = None
    last_ch_done  = 0

    # ── Text mode (default) ───────────────────────────────────────────────
    if args.mode == "text":
        chapters_arg = args.chapters or str(DEFAULT_TEXT_CHAPTERS)
        result, ch_nums = run_text_mode(novel_dir, chapters_arg, template)
        last_ch_done    = ch_nums[-1]

    # ── Web mode ──────────────────────────────────────────────────────────
    else:
        chapter_range = args.chapters or "1-100"

        result, is_good = run_web_mode(
            novel_dir, novel_name, chapter_range, template
        )

        # Auto fallback to text if coverage is poor
        if not is_good and not args.no_fallback:
            has_local = bool(get_chapters_in_range(novel_dir, 1, None))
            if has_local:
                print(f"\n  Auto-fallback: supplementing with local chapter text...")
                text_chapters = args.chapters or str(DEFAULT_TEXT_CHAPTERS)
                result, ch_nums = run_text_mode(novel_dir, text_chapters, result)
                last_ch_done = ch_nums[-1]
            else:
                print(f"\n  ⚠ No local chapters for fallback.")
                print(f"  Add chapter files to {os.path.join(novel_dir, 'input')}/ "
                      f"or re-run with --mode text")
        elif not is_good and args.no_fallback:
            print(f"\n  ⚠ Poor coverage but --no-fallback set. Saving as-is.")

    if result is None:
        print("\n  ✗ No result to save.")
        sys.exit(1)

    # ── Update meta and write ─────────────────────────────────────────────
    if last_ch_done > 0:
        result = update_meta_field(result, "last_updated_chapter",
                                   str(last_ch_done))
    result = update_meta_field(result, "total_chapters_processed",
                               str(last_ch_done) if last_ch_done else "unknown")

    write_novel_md(novel_dir, result)

    print(f"\n{'═' * 58}")
    print(f"  ✓ novel.md initialized")
    if last_ch_done:
        print(f"  Coverage through chapter: {last_ch_done}")
    print(f"  Next steps:")
    print(f"    python batch.py {novel_dir}")
    print(f"    python novel_manager/update_novel.py {novel_dir}  (after ~50 more chapters)")
    print(f"{'═' * 58}")


if __name__ == "__main__":
    main()