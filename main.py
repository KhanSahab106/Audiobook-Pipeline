"""
main.py

Pipeline orchestrator — runs the full audiobook generation pipeline for a
single chapter: read text → parse with LLM → resolve speakers → render
TTS audio → assemble final WAV. Can be called standalone for one chapter
or imported by batch.py for multi-chapter runs.

Functions:
    get_novel_dir(chapter_file)       — Derive the novel root directory from a chapter file path.
    process_chapter(chapter_file, tts, tts_config) — Full pipeline for one chapter; returns (tts, config) for reuse.
    _header(title, width)             — Print a bordered section header.
    _step(label, title)               — Print a numbered step indicator.
    _divider(width)                   — Print a horizontal divider line.
    main()                            — CLI entry point: parse args and run process_chapter.
"""

import time
import os
import sys
import json
from parser    import parse_chapter, get_token_report, reset_token_tracker
from registry  import load_registry, resolve_speaker, get_known_characters, save_registry
from renderer  import load_tts, render_segment
from assembler import assemble
from fallback  import render_with_fallback


# ═══════════════════════════════════════════════════════════════════════════
#  PATH HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def get_novel_dir(chapter_file: str) -> str:
    """
    Derive the novel root directory from a chapter file path.

    Expected structure:
        novels/shadow_slave/input/chapter_1.txt
                            ^^^^^ input dir
        novels/shadow_slave/                    ← novel_dir
                ^^^^^^^^^^^^^

    If the chapter file is NOT inside a novels/ subfolder
    (e.g. legacy path like input/chapter_1.txt), return "." so the
    pipeline still works against the project root.
    """
    abs_path   = os.path.abspath(chapter_file)
    input_dir  = os.path.dirname(abs_path)          # .../shadow_slave/input
    novel_dir  = os.path.dirname(input_dir)          # .../shadow_slave

    # Sanity check — the parent of the input dir should have a data/ and output/ folder
    # or at least be inside a novels/ directory.
    if os.path.basename(input_dir) == "input":
        return novel_dir

    # Fallback for legacy flat structure
    return "."


# ═══════════════════════════════════════════════════════════════════════════
#  CHAPTER PROCESSOR
# ═══════════════════════════════════════════════════════════════════════════

def process_chapter(chapter_file: str, tts=None, tts_config=None):
    """
    Full pipeline for a single chapter file.

    Derives all output paths from the chapter file location:
        novels/{novel}/input/chapter_1.txt
        → output: novels/{novel}/output/chapter_1.wav
        → logs:   novels/{novel}/data/chapter_1_failures.json

    Returns (tts, tts_config) for reuse across chapters in a batch.
    """
    novel_dir    = get_novel_dir(chapter_file)
    chapter_name = os.path.splitext(os.path.basename(chapter_file))[0]
    output_dir   = os.path.join(novel_dir, "output")
    data_dir     = os.path.join(novel_dir, "data")
    output_path  = os.path.join(output_dir, f"{chapter_name}.wav")
    log_path     = os.path.join(data_dir,   f"{chapter_name}_failures.json")

    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(data_dir,   exist_ok=True)

    _header(f"  Chapter : {chapter_name}")
    print(f"  Novel   : {novel_dir}")
    print(f"  Output  : {output_path}")

    # ── Read input ───────────────────────────────────────────
    with open(chapter_file, "r", encoding="utf-8") as f:
        text = f.read()

    word_count = len(text.split())
    print(f"  Words   : {word_count}\n")

    # ── Load registry ────────────────────────────────────────
    registry = load_registry(novel_dir)
    known    = get_known_characters(registry)

    # ── [1/3] Parse ──────────────────────────────────────────
    _step("1/3", "Parsing chapter")
    t0 = time.perf_counter()

    try:
        segments = parse_chapter(text, known)
    except Exception as e:
        print(f"  ✗ Parser failed: {e}")
        print(f"  Chapter {chapter_name} skipped.\n")
        return tts, tts_config

    parse_time     = time.perf_counter() - t0
    speakers_found = set(s["speaker"] for s in segments)

    print(f"  Segments : {len(segments)}")
    print(f"  Speakers : {speakers_found}")
    print(f"  Time     : {parse_time:.1f}s\n")

    if not segments:
        print("  ✗ No segments returned. Skipping chapter.")
        return tts, tts_config

    # ── [2/3] Load TTS ───────────────────────────────────────
    _step("2/3", "TTS model")
    if tts is None or tts_config is None:
        tts, tts_config = load_tts()
    else:
        print("  Using pre-loaded model.\n")

    # ── [3/3] Render ─────────────────────────────────────────
    _step("3/3", f"Rendering {len(segments)} segments")
    t_render       = time.perf_counter()
    failure_log    = []
    audio_segments = []

    for seg in segments:
        if not seg["text"].strip():
            continue

        xtts_speaker = resolve_speaker(seg["speaker"], registry, novel_dir)

        print(
            f"  [{seg['index']:03d}] "
            f"{seg['speaker']:<15} | "
            f"{xtts_speaker:<22} | "
            f"{seg['tone']:<8} | "
            f"{seg['text'][:55]}"
        )

        wav = render_with_fallback(
            tts, tts_config,
            seg, xtts_speaker,
            failure_log
        )

        audio_segments.append({
            "index": seg["index"],
            "type":  seg["type"],
            "wav":   wav,
        })

    render_time = time.perf_counter() - t_render
    print(f"\n  Rendered : {len(audio_segments)}/{len(segments)} segments")
    print(f"  Time     : {render_time:.1f}s\n")

    # ── Write failure log ────────────────────────────────────
    if failure_log:
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(failure_log, f, indent=2)

        placeholders = [e for e in failure_log if e["strategy"] == "placeholder_inserted"]
        print(f"  ⚠ {len(failure_log)} segment(s) used fallback → {log_path}")

        if placeholders:
            print(f"  ✗ {len(placeholders)} placeholder(s) inserted — MANUAL FIX REQUIRED:")
            for p in placeholders:
                print(f"    Segment {p['index']:03d}: {p['text'][:70]}")
        print()

    # ── Assemble + export ────────────────────────────────────
    if not audio_segments:
        print("  ✗ No audio to assemble. Skipping export.")
        return tts, tts_config

    audio_segments.sort(key=lambda x: x["index"])
    print("  Assembling audio...")
    assemble(audio_segments, output_path)

    # ── Token + timing summary ───────────────────────────────
    token_report = get_token_report()
    reset_token_tracker()

    total_time = time.perf_counter() - t0
    rate       = word_count / total_time * 60

    print(f"  Tokens     : {token_report['total_tokens']} "
          f"(in: {token_report['input_tokens']} / "
          f"out: {token_report['output_tokens']})")

    _divider()
    print(f"  ✓ {output_path}")
    print(f"  Words      : {word_count}")
    print(f"  Parse      : {parse_time:.1f}s")
    print(f"  Render     : {render_time:.1f}s")
    print(f"  Total      : {total_time:.1f}s")
    print(f"  Rate       : {rate:.0f} words/min")
    if failure_log:
        print(f"  Fallbacks  : {len(failure_log)}")
    _divider()

    return tts, tts_config


# ═══════════════════════════════════════════════════════════════════════════
#  DISPLAY HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _header(title: str, width: int = 60):
    print(f"\n{'═' * width}")
    print(title)
    print(f"{'═' * width}")


def _step(label: str, title: str):
    print(f"  [{label}] {title}")
    print(f"  {'─' * 50}")


def _divider(width: int = 60):
    print(f"  {'─' * width}")


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  Single chapter : python main.py novels/shadow_slave/input/chapter_1.txt")
        print("  Multi chapter  : python main.py novels/shadow_slave/input/chapter_1.txt chapter_2.txt ...")
        print("  Full batch     : python batch.py novels/shadow_slave")
        sys.exit(1)

    chapter_files = sys.argv[1:]

    missing = [f for f in chapter_files if not os.path.exists(f)]
    if missing:
        print("  ✗ Files not found:")
        for f in missing:
            print(f"    {f}")
        sys.exit(1)

    tts, tts_config = None, None
    t_total         = time.perf_counter()
    failed_chapters = []

    for i, chapter_file in enumerate(chapter_files, 1):
        _header(f"  Chapter {i} of {len(chapter_files)}: {os.path.basename(chapter_file)}")

        try:
            result = process_chapter(chapter_file, tts, tts_config)
            if result:
                tts, tts_config = result
        except Exception as e:
            print(f"\n  ✗ Unexpected failure: {e}")
            print(f"  Skipping to next chapter...\n")
            failed_chapters.append(chapter_file)
            continue

    total = time.perf_counter() - t_total
    _header("  Batch Summary")
    print(f"  Chapters   : {len(chapter_files)}")
    print(f"  Completed  : {len(chapter_files) - len(failed_chapters)}")
    print(f"  Failed     : {len(failed_chapters)}")
    print(f"  Total time : {total / 60:.1f} minutes")

    if failed_chapters:
        print(f"\n  ✗ Failed chapters:")
        for f in failed_chapters:
            print(f"    {f}")

    _divider()


if __name__ == "__main__":
    main()