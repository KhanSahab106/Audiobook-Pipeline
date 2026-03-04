"""
novel_manager/novel_utils.py

Shared utilities: reading chapters, parsing novel.md metadata,
writing novel.md safely.

Functions:
    _chapter_number(filepath)         — Extract chapter number from filename.
    get_all_chapters(novel_dir)       — Return sorted (chapter_number, filepath) pairs.
    get_chapters_in_range(novel_dir, start, end) — Filter chapters by range.
    load_chapters_text(chapter_pairs, separator) — Read and concatenate chapter files.
    novel_md_path(novel_dir)          — Return path to novel.md.
    read_novel_md(novel_dir)          — Read novel.md content.
    get_last_updated_chapter(novel_dir) — Parse last_updated_chapter from novel.md meta.
    write_novel_md(novel_dir, content, backup) — Write novel.md with optional backup.
    update_meta_field(content, field, value) — Update a field in the Meta section.
    extract_character_keys(content)       — Return set of all character keys in the Characters section.
    parse_chapter_range(arg)              — Parse "101-200" / "101-" / "101" into (start, end).
    extract_character_last_updated(content) — Return {key: last_updated_chapter} for all characters.
"""

import os
import re
import glob
import shutil
from datetime import date


# ═══════════════════════════════════════════════════════════════════════════
#  CHAPTER READING
# ═══════════════════════════════════════════════════════════════════════════

def _chapter_number(filepath: str) -> int:
    """Extract chapter number from filename like chapter_42.txt → 42"""
    name = os.path.basename(filepath)
    match = re.search(r"(\d+)", name)
    return int(match.group(1)) if match else 0


def get_all_chapters(novel_dir: str) -> list[tuple[int, str]]:
    """
    Return sorted list of (chapter_number, filepath) for all chapters
    in novels/{novel}/input/chapter_*.txt
    """
    pattern = os.path.join(novel_dir, "input", "chapter_*.txt")
    files   = glob.glob(pattern)
    pairs   = [(_chapter_number(f), f) for f in files]
    return sorted(pairs, key=lambda x: x[0])


def get_chapters_in_range(
    novel_dir: str,
    start: int,
    end: int | None = None
) -> list[tuple[int, str]]:
    """
    Return chapters where start <= chapter_number <= end.
    If end is None, returns all chapters from start onwards.
    """
    all_ch = get_all_chapters(novel_dir)
    return [
        (n, f) for n, f in all_ch
        if n >= start and (end is None or n <= end)
    ]


def load_chapters_text(
    chapter_pairs: list[tuple[int, str]],
    separator: bool = True
) -> str:
    """
    Read and concatenate chapter text files into a single string.
    Each chapter is prefixed with a header so the LLM knows chapter boundaries.
    """
    parts = []
    for num, filepath in chapter_pairs:
        try:
            with open(filepath, "r", encoding="utf-8") as f:
                text = f.read().strip()
            header = f"\n\n{'─' * 60}\nCHAPTER {num}\n{'─' * 60}\n"
            parts.append(header + text)
        except Exception as e:
            print(f"  Warning: could not read {filepath}: {e}")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════
#  NOVEL.MD READING / WRITING
# ═══════════════════════════════════════════════════════════════════════════

def novel_md_path(novel_dir: str) -> str:
    return os.path.join(novel_dir, "novel.md")


def read_novel_md(novel_dir: str) -> str:
    path = novel_md_path(novel_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"novel.md not found at {path}\n"
            f"Run: python setup_novel.py {os.path.basename(novel_dir)}"
        )
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def get_last_updated_chapter(novel_dir: str) -> int:
    """Parse last_updated_chapter from novel.md meta section."""
    try:
        content = read_novel_md(novel_dir)
        match   = re.search(r"last_updated_chapter:\s*(\d+)", content)
        return int(match.group(1)) if match else 0
    except FileNotFoundError:
        return 0


def write_novel_md(novel_dir: str, content: str, backup: bool = True):
    """
    Write updated content to novel.md.
    Creates a timestamped backup first if backup=True.
    Strips any markdown code fences the LLM may have wrapped the output in.
    """
    path = novel_md_path(novel_dir)

    # Strip code fences if Gemini wrapped the output
    content = re.sub(r"^```(?:markdown)?\s*\n", "", content, flags=re.MULTILINE)
    content = re.sub(r"\n```\s*$", "", content, flags=re.MULTILINE)
    content = content.strip()

    # Backup existing file
    if backup and os.path.exists(path):
        backup_dir  = os.path.join(novel_dir, "data", "novel_backups")
        os.makedirs(backup_dir, exist_ok=True)
        backup_name = f"novel_{date.today().isoformat()}.md"
        backup_path = os.path.join(backup_dir, backup_name)
        shutil.copy2(path, backup_path)
        print(f"  Backup saved: {backup_path}")

    with open(path, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  novel.md updated: {path}")


def update_meta_field(content: str, field: str, value: str) -> str:
    """Update a field in the Meta section of novel.md content."""
    pattern     = rf"({re.escape(field)}:\s*)(.+)"
    replacement = rf"\g<1>{value}"
    updated     = re.sub(pattern, replacement, content)
    if updated == content:
        # Field not found — insert after the meta header
        updated = content.replace(
            "## Meta\n",
            f"## Meta\n{field}: {value}\n"
        )
    return updated


def extract_character_keys(novel_md_content: str) -> set[str]:
    """
    Parse all character keys (### headings) from the ## Characters section.

    Returns a set like {'idan', 'arabel_morgan', 'narrator', ...}.
    Only captures headings between ## Characters and the next ##-level section.
    """
    # Find the Characters section
    chars_match = re.search(r'^## Characters\s*\n(.*?)(?=^## |\Z)', novel_md_content,
                            flags=re.MULTILINE | re.DOTALL)
    if not chars_match:
        return set()

    chars_section = chars_match.group(1)
    keys = set(re.findall(r'^### (\S+)', chars_section, flags=re.MULTILINE))
    return keys


def _parse_character_entries(novel_md_content: str) -> dict[str, dict]:
    """
    Parse all character entries from the ## Characters section into structured dicts.

    Returns {key: {"status": "active"|"dormant", "confidence": ..., "role": ...}, ...}
    """
    chars_match = re.search(r'^## Characters\s*\n(.*?)(?=^## |\Z)', novel_md_content,
                            flags=re.MULTILINE | re.DOTALL)
    if not chars_match:
        return {}

    chars_section = chars_match.group(1)
    entries = {}

    # Split by ### headings
    blocks = re.split(r'^### (\S+)\s*\n', chars_section, flags=re.MULTILINE)
    # blocks = ['', key1, body1, key2, body2, ...]
    for i in range(1, len(blocks) - 1, 2):
        key  = blocks[i].strip()
        body = blocks[i + 1] if i + 1 < len(blocks) else ""

        entry = {"status": "active", "confidence": "sparse", "role": "minor",
                 "last_updated_chapter": None}

        status_m = re.search(r'-\s*status:\s*(\w+)', body)
        if status_m:
            entry["status"] = status_m.group(1).lower()

        conf_m = re.search(r'-\s*confidence:\s*(\w+)', body)
        if conf_m:
            entry["confidence"] = conf_m.group(1).lower()

        role_m = re.search(r'-\s*role:\s*(.+)', body)
        if role_m:
            entry["role"] = role_m.group(1).strip().lower()

        luc_m = re.search(r'-\s*last_updated_chapter:\s*(\d+)', body)
        if luc_m:
            entry["last_updated_chapter"] = int(luc_m.group(1))

        entries[key] = entry

    return entries


def extract_dormant_characters(novel_md_content: str) -> set[str]:
    """Return the set of character keys with status: dormant."""
    entries = _parse_character_entries(novel_md_content)
    return {k for k, v in entries.items() if v["status"] == "dormant"}


def extract_newly_dormant(old_md: str, new_md: str) -> set[str]:
    """
    Return characters that went from active (or absent) in old_md
    to dormant in new_md.
    """
    old_dormant = extract_dormant_characters(old_md)
    new_dormant = extract_dormant_characters(new_md)
    return new_dormant - old_dormant


def extract_reactivated(old_md: str, new_md: str) -> set[str]:
    """
    Return characters that went from dormant in old_md
    to active (or absent) in new_md.
    """
    old_dormant = extract_dormant_characters(old_md)
    new_dormant = extract_dormant_characters(new_md)
    return old_dormant - new_dormant


def parse_chapter_range(arg: str) -> tuple[int, int | None]:
    """
    Parse a chapter range argument into (start, end).

    Accepts:
        "101"    → (101, 101)
        "101-200" → (101, 200)
        "101-"   → (101, None)   # all from 101 onwards
    """
    if "-" in arg:
        parts = arg.split("-", 1)
        start = int(parts[0]) if parts[0] else 1
        end   = int(parts[1]) if parts[1] else None
        return start, end
    return int(arg), int(arg)


def extract_character_last_updated(novel_md_content: str) -> dict[str, int]:
    """
    Return {character_key: last_updated_chapter_int} for all characters
    that have a last_updated_chapter field.  Characters without the field
    are omitted from the result.
    """
    entries = _parse_character_entries(novel_md_content)
    return {
        k: v["last_updated_chapter"]
        for k, v in entries.items()
        if v["last_updated_chapter"] is not None
    }

