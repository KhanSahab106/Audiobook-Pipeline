"""
setup_novel.py

One-time setup for a new novel. Creates the folder structure and
an empty novel.md template to fill in before the first run.

Usage:
    python setup_novel.py shadow_slave
    python setup_novel.py "another novel name"
"""

import os
import sys
import re

# Resolve the project root regardless of the working directory when the script is run.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

NOVEL_MD_TEMPLATE = """\
# {title}

## Meta
last_updated_chapter: 0
total_chapters_processed: 0

## Overview
<!-- One paragraph describing the novel: genre, setting, tone.
     This helps the voice casting system understand the overall feel. -->


## Characters

### narrator
- status: active
- Role: narrator
- Gender: neutral
- Voice style: clear, measured, slightly dramatic
- Notes: default narration voice

### system
- status: active
- Role: system
- Gender: neutral
- Voice style: flat, mechanical, notification-like
- Notes: system messages, notifications, status windows

<!--
Add one entry per named character below.
The "name_key" must match exactly what the parser uses as the speaker name
(lowercase, underscores for spaces — e.g. "john_doe" not "John Doe").

### name_key
- status: active / dormant
- confidence: sparse / partial / complete
- introduced_chapter: N
- Role: protagonist / antagonist / supporting / minor
- Gender: male / female / unknown
- Age: rough range e.g. "late teens", "mid 30s", "elderly"
- personality:
- voice_style: e.g. "quiet and withdrawn", "loud and aggressive", "cheerful"
- speech_patterns:
- arc_notes:
  Ch N: [development]
- relationship_to_protagonist:
- casting_note:
- last_updated_chapter: N
-->


## World / tone notes
<!-- Optional. Describe the world tone so the voice casting
     system can make better choices for unnamed/minor characters.
     e.g. "dark xianxia fantasy", "light-hearted isekai", "grimdark military" -->


## Factions
<!-- Major groups or organizations -->


## Chapter Map
<!-- Brief chapter summaries for tracking arcs -->
"""


def slugify(name: str) -> str:
    """Convert 'Shadow Slave' → 'shadow_slave'"""
    name = name.lower().strip()
    name = re.sub(r"[^\w\s-]", "", name)
    name = re.sub(r"[\s-]+", "_", name)
    return name


def setup_novel(raw_name: str):
    slug      = slugify(raw_name)
    title     = raw_name.title()
    novel_dir = os.path.join(ROOT, "novels", slug)

    dirs = [
        novel_dir,
        os.path.join(novel_dir, "input"),
        os.path.join(novel_dir, "output"),
        os.path.join(novel_dir, "data"),
    ]

    print(f"\n  Setting up: {title}")
    print(f"  Directory : {novel_dir}\n")

    for d in dirs:
        if os.path.exists(d):
            print(f"  ✓ exists   {d}/")
        else:
            os.makedirs(d)
            print(f"  ✓ created  {d}/")

    # Write novel.md template if it doesn't exist yet
    novel_md = os.path.join(novel_dir, "novel.md")
    if os.path.exists(novel_md):
        print(f"  ✓ exists   {novel_md}  (not overwritten)")
    else:
        with open(novel_md, "w", encoding="utf-8") as f:
            f.write(NOVEL_MD_TEMPLATE.format(title=title))
        print(f"  ✓ created  {novel_md}")

    print(f"""
  Next steps:
    1. Edit {novel_md}
       Add character names, roles, and voice style notes.

    2. Copy chapter .txt files into:
       {os.path.join(novel_dir, 'input')}/
       Name them: chapter_1.txt, chapter_2.txt, ...

    3. Run the batch:
       python batch.py {novel_dir}

    4. Output WAVs will appear in:
       {os.path.join(novel_dir, 'output')}/
""")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python setup_novel.py <novel name>")
        print("  e.g. python setup_novel.py shadow_slave")
        sys.exit(1)

    setup_novel(" ".join(sys.argv[1:]))
