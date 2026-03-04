"""
diagnose_coverage.py

Compares source chapter text against what the parser actually returned,
sentence by sentence, and prints a coverage report.

Functions:
    sentence_tokenize(text)           — Split text into sentences.
    normalize(text)                   — Lowercase and strip punctuation for fuzzy matching.
    words(text)                       — Return word set from text.
    coverage_ratio(source_sentence, segment_texts) — Fraction of source words found in segments.
    main()                            — CLI entry point: parse chapter and print coverage report.

Usage:
    python diagnose_coverage.py input/chapter_4.txt
"""

import sys
import re
import os
import json

# ── Add project root to path ──────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Minimal parser re-run (no TTS) ─────────────────────────────────────────
from parser import parse_chapter, reset_token_tracker
from registry import load_registry, get_known_characters


def sentence_tokenize(text: str) -> list[str]:
    """Split text into sentences. Simple but good enough for coverage check."""
    sentences = re.split(r'(?<=[.!?])["\']?\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def normalize(text: str) -> str:
    """Lowercase, strip punctuation for fuzzy matching."""
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


def words(text: str) -> set[str]:
    return set(normalize(text).split())


def coverage_ratio(source_sentence: str, segment_texts: list[str]) -> float:
    """
    What fraction of the source sentence's words appear in any segment?
    Returns 0.0–1.0.
    """
    src_words = words(source_sentence)
    if not src_words:
        return 1.0

    all_segment_words = set()
    for t in segment_texts:
        all_segment_words.update(words(t))

    matched = src_words & all_segment_words
    return len(matched) / len(src_words)


def main():
    if len(sys.argv) < 2:
        print("Usage: python diagnose_coverage.py input/chapter_4.txt")
        sys.exit(1)

    chapter_file = sys.argv[1]

    with open(chapter_file, "r", encoding="utf-8") as f:
        source_text = f.read()

    print(f"\nSource: {chapter_file}")
    print(f"Source words : {len(source_text.split())}")

    # ── Re-parse (uses your real parser + Groq key) ────────────────────────
    print("\nRe-parsing chapter to check coverage...\n")
    # Derive novel_dir from chapter path: novels/{name}/input/chapter.txt → novels/{name}
    input_dir = os.path.dirname(os.path.abspath(chapter_file))
    novel_dir = os.path.dirname(input_dir) if os.path.basename(input_dir) == "input" else "."
    registry = load_registry(novel_dir)
    known    = get_known_characters(registry)
    reset_token_tracker()

    segments = parse_chapter(source_text, known)
    segment_texts = [s["text"] for s in segments]

    total_segment_words = sum(len(t.split()) for t in segment_texts)
    print(f"\nParsed segments  : {len(segments)}")
    print(f"Segment words    : {total_segment_words}")
    print(f"Source words     : {len(source_text.split())}")
    print(f"Word coverage    : {total_segment_words / len(source_text.split()) * 100:.1f}%")

    # ── Sentence-level gap analysis ────────────────────────────────────────
    source_sentences = sentence_tokenize(source_text)
    COVERAGE_THRESHOLD = 0.50   # sentence is "missing" if <50% of words found in any segment

    missing    = []
    partial    = []
    covered    = []

    for sent in source_sentences:
        if len(sent.split()) < 3:     # skip fragments like "He did." 
            covered.append(sent)
            continue
        ratio = coverage_ratio(sent, segment_texts)
        if ratio < COVERAGE_THRESHOLD:
            missing.append((sent, ratio))
        elif ratio < 0.85:
            partial.append((sent, ratio))
        else:
            covered.append(sent)

    print(f"\n── Sentence Coverage Report ─────────────────────────────")
    print(f"  Total sentences : {len(source_sentences)}")
    print(f"  Covered (≥85%)  : {len(covered)}")
    print(f"  Partial (50-85%): {len(partial)}")
    print(f"  Missing (<50%)  : {len(missing)}")
    print(f"  Coverage rate   : {len(covered) / len(source_sentences) * 100:.1f}%")

    if missing:
        print(f"\n── MISSING sentences ────────────────────────────────────")
        for sent, ratio in missing:
            print(f"  [{ratio*100:4.0f}%] {sent[:120]}")

    if partial:
        print(f"\n── PARTIAL sentences ────────────────────────────────────")
        for sent, ratio in partial:
            print(f"  [{ratio*100:4.0f}%] {sent[:120]}")

    # ── Save report ────────────────────────────────────────────────────────
    report = {
        "chapter":          chapter_file,
        "source_words":     len(source_text.split()),
        "segment_words":    total_segment_words,
        "word_coverage_pct": round(total_segment_words / len(source_text.split()) * 100, 1),
        "total_sentences":  len(source_sentences),
        "covered":          len(covered),
        "partial":          len(partial),
        "missing":          len(missing),
        "missing_sentences": [s for s, _ in missing],
        "partial_sentences": [s for s, _ in partial],
    }

    os.makedirs("data", exist_ok=True)
    report_path = "data/coverage_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n  Full report saved to: {report_path}")


if __name__ == "__main__":
    main()
