"""
batch.py

Batch chapter processor — iterates over all chapter_*.txt files in a novel's
input/ folder, runs the full pipeline (parse → render → assemble) for each,
and tracks progress via a checkpoint file so interrupted runs can resume.

After all chapters are processed, automatically combines every 10 consecutive
chapter WAV files into single MP3 files (1-10, 11-20, 21-30, ...).

Functions:
    _checkpoint_path(novel_dir)       — Return path to checkpoint.json.
    load_checkpoint(novel_dir)        — Load set of already-processed chapter paths.
    save_checkpoint(novel_dir, completed) — Persist completed chapter set.
    combine_chapters(novel_dir)       — Merge every 10 chapter WAVs into combined MP3s.
    run_batch(novel_dir)              — Process all remaining chapters with checkpoint recovery.
"""

import os
import sys
import json
import time
import glob
from pydub import AudioSegment
from main import process_chapter


# ═══════════════════════════════════════════════════════════════════════════
#  WINDOWS PRIORITY BOOST — prevents OS from throttling background processes
# ═══════════════════════════════════════════════════════════════════════════

def _set_high_priority():
    """Set this process to HIGH priority so Windows doesn't throttle it in the background."""
    try:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            # 0x0080 = HIGH_PRIORITY_CLASS
            handle = kernel32.GetCurrentProcess()
            kernel32.SetPriorityClass(handle, 0x0080)
            print("  ⚡ Process priority set to HIGH (background throttling disabled)")
    except Exception as e:
        print(f"  ⚠ Could not set high priority: {e}")

_set_high_priority()


# ═══════════════════════════════════════════════════════════════════════════
#  CHECKPOINT
# ═══════════════════════════════════════════════════════════════════════════

def _checkpoint_path(novel_dir: str) -> str:
    return os.path.join(novel_dir, "data", "checkpoint.json")


def load_checkpoint(novel_dir: str) -> set:
    path = _checkpoint_path(novel_dir)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return set(json.load(f)["completed"])
    return set()


def save_checkpoint(novel_dir: str, completed: set):
    path = _checkpoint_path(novel_dir)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump({"completed": list(completed)}, f, indent=2)


# ═══════════════════════════════════════════════════════════════════════════
#  COMBINE CHAPTERS → MP3
# ═══════════════════════════════════════════════════════════════════════════

GROUP_SIZE = 10
INTER_CHAPTER_SILENCE_MS = 2000


def combine_chapters(novel_dir: str):
    """
    Combine every 10 consecutive chapter WAVs into a single MP3.

    Groups:
        chapters  1-10  → chapters_1_to_10.mp3
        chapters 11-20  → chapters_11_to_20.mp3
        chapters 21-30  → chapters_21_to_30.mp3
        ...and so on.

    Output is saved to novels/{novel}/combined/
    """
    output_dir   = os.path.join(novel_dir, "output")
    combined_dir = os.path.join(novel_dir, "combined")

    if not os.path.isdir(output_dir):
        print("  ⚠ No output directory — skipping combine step.")
        return

    # Discover chapter WAVs → {chapter_number: path}
    wav_map = {}
    for fname in os.listdir(output_dir):
        if fname.startswith("chapter_") and fname.endswith(".wav"):
            try:
                num = int(fname.replace("chapter_", "").replace(".wav", ""))
                wav_map[num] = os.path.join(output_dir, fname)
            except ValueError:
                continue

    if not wav_map:
        print("  ⚠ No chapter WAV files found — skipping combine step.")
        return

    os.makedirs(combined_dir, exist_ok=True)

    all_nums = sorted(wav_map.keys())
    max_ch   = max(all_nums)

    print(f"\n{'═' * 55}")
    print(f"  Combining chapters → MP3")
    print(f"{'═' * 55}")
    print(f"  Chapters found : {len(wav_map)} WAV files")
    print(f"  Group size     : {GROUP_SIZE}")
    print(f"  Output dir     : {combined_dir}")

    silence = AudioSegment.silent(duration=INTER_CHAPTER_SILENCE_MS)
    group_start    = 1
    groups_created = 0
    groups_skipped = 0

    while group_start <= max_ch:
        group_end    = group_start + GROUP_SIZE - 1
        chapter_nums = list(range(group_start, group_end + 1))
        present      = [n for n in chapter_nums if n in wav_map]

        if present:
            dest_name = f"chapters_{group_start}_to_{group_end}.mp3"
            dest_path = os.path.join(combined_dir, dest_name)

            # Skip if all chapters in this group are already combined
            if os.path.exists(dest_path) and len(present) == GROUP_SIZE:
                print(f"\n  Group {group_start}–{group_end}: ✓ already exists — skipped")
                groups_skipped += 1
                group_start += GROUP_SIZE
                continue

            print(f"\n  Group {group_start}–{group_end}:")
            combined = AudioSegment.empty()
            loaded   = 0

            for num in chapter_nums:
                if num not in wav_map:
                    print(f"    ⚠ chapter_{num}.wav not found — skipping")
                    continue
                audio = AudioSegment.from_wav(wav_map[num])
                if loaded > 0:
                    combined += silence
                combined += audio
                loaded += 1

            if loaded > 0:
                combined.export(dest_path, format="mp3", bitrate="192k")
                duration_min = len(combined) / 1000 / 60
                print(f"    ✓ {dest_name}  ({loaded} chapters, {duration_min:.1f} min)")
                groups_created += 1

        group_start += GROUP_SIZE

    print(f"\n  {'─' * 50}")
    print(f"  ✓ {groups_created} new MP3(s) created, {groups_skipped} skipped (already exist)")
    print(f"  ✓ Output: {combined_dir}")
    print(f"  {'─' * 50}")


# ═══════════════════════════════════════════════════════════════════════════
#  BATCH RUNNER
# ═══════════════════════════════════════════════════════════════════════════

def run_batch(novel_dir: str, chapter_start: int = None, chapter_end: int = None):
    """
    Process all chapters in novels/{novel}/input/chapter_*.txt

    - Optionally filter to a specific chapter range (chapter_start–chapter_end)
    - Skips already-completed chapters (checkpoint recovery)
    - TTS model loaded once, reused across all chapters
    - Failures logged to novels/{novel}/data/failures.log
    """
    novel_dir   = os.path.normpath(novel_dir)
    input_dir   = os.path.join(novel_dir, "input")
    data_dir    = os.path.join(novel_dir, "data")
    failure_log = os.path.join(data_dir,  "failures.log")

    if not os.path.isdir(novel_dir):
        print(f"✗ Novel directory not found: {novel_dir}")
        print(f"  Run: python setup_novel.py <novel_name>")
        sys.exit(1)

    if not os.path.isdir(input_dir):
        print(f"✗ Input directory not found: {input_dir}")
        print(f"  Add chapter .txt files to {input_dir}/")
        sys.exit(1)

    pattern   = os.path.join(input_dir, "chapter_*.txt")
    all_files = sorted(
        glob.glob(pattern),
        key=lambda f: int(os.path.basename(f).replace("chapter_", "").replace(".txt", ""))
    )

    if not all_files:
        print(f"No chapter files found in: {input_dir}")
        print(f"  Expected filenames like: chapter_1.txt, chapter_2.txt ...")
        return

    # ── Filter to requested chapter range ─────────────────────
    if chapter_start is not None or chapter_end is not None:
        def _ch_num(f):
            return int(os.path.basename(f).replace("chapter_", "").replace(".txt", ""))

        lo = chapter_start if chapter_start is not None else 1
        hi = chapter_end   if chapter_end   is not None else 999999
        all_files = [f for f in all_files if lo <= _ch_num(f) <= hi]

        if not all_files:
            print(f"No chapters found in range {lo}–{hi}.")
            return

    os.makedirs(data_dir, exist_ok=True)

    completed = load_checkpoint(novel_dir)
    remaining = [f for f in all_files if f not in completed]

    novel_name = os.path.basename(novel_dir)
    range_label = ""
    if chapter_start is not None or chapter_end is not None:
        lo = chapter_start if chapter_start is not None else "start"
        hi = chapter_end   if chapter_end   is not None else "end"
        range_label = f"  Range     : {lo}–{hi}\n"

    print(f"\n{'═' * 55}")
    print(f"  Novel     : {novel_name}")
    print(f"  Directory : {novel_dir}")
    if range_label:
        print(range_label, end="")
    print(f"{'═' * 55}")
    print(f"  Total chapters  : {len(all_files)}")
    print(f"  Already done    : {len(completed)}")
    print(f"  Remaining       : {len(remaining)}")
    print(f"  Estimated time  : {len(remaining) * 212 / 3600:.1f} hours")
    print(f"{'=' * 55}")

    if not remaining:
        print("  All chapters already processed.")
        return

    # Load TTS once — reuse across all chapters
    tts, tts_config = None, None
    t_batch_start   = time.perf_counter()

    for i, chapter_file in enumerate(remaining, 1):
        print(f"\n[{i}/{len(remaining)}] {os.path.basename(chapter_file)}")
        t0 = time.perf_counter()

        # ── Retry with escalating delay (10s → 20s → ... → 600s) ─────
        MAX_DELAY   = 600   # 10 minutes
        DELAY_STEP  = 10    # increase by 10s each retry
        delay       = DELAY_STEP
        succeeded   = False

        while not succeeded:
            try:
                result = process_chapter(chapter_file, tts, tts_config)
                if result:
                    tts, tts_config = result

                completed.add(chapter_file)
                save_checkpoint(novel_dir, completed)

                elapsed         = time.perf_counter() - t0
                remaining_count = len(remaining) - i
                eta_hours       = remaining_count * elapsed / 3600
                print(f"  ✓ Done in {elapsed:.0f}s | Remaining ETA: {eta_hours:.1f}h")
                succeeded = True

            except KeyboardInterrupt:
                print("\n  ⚠ Interrupted by user — saving checkpoint and exiting.")
                save_checkpoint(novel_dir, completed)
                sys.exit(0)

            except Exception as e:
                print(f"  ✗ ERROR: {os.path.basename(chapter_file)}")
                print(f"    {type(e).__name__}: {e}")
                with open(failure_log, "a") as log:
                    log.write(f"{chapter_file} | {type(e).__name__}: {e}\n")

                if delay > MAX_DELAY:
                    print(f"  ✗ Giving up on {os.path.basename(chapter_file)} after max retries.")
                    break

                print(f"  ⏳ Retrying in {delay}s...")
                time.sleep(delay)
                delay = min(delay + DELAY_STEP, MAX_DELAY)

    total = time.perf_counter() - t_batch_start
    print(f"\n{'=' * 55}")
    print(f"  Batch complete.")
    print(f"  Processed   : {len(remaining)} chapters")
    print(f"  Total time  : {total / 3600:.2f} hours")
    print(f"  Output      : {os.path.join(novel_dir, 'output')}/")
    if os.path.exists(failure_log):
        print(f"  Failures    : {failure_log}")
    print(f"{'=' * 55}")

    # ── Combine every 10 chapters into MP3 ──────────────────
    combine_chapters(novel_dir)


# ═══════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python batch.py <novel_dir> [start-end]")
        print("")
        print("Examples:")
        print("  python batch.py novels/shadow_slave          # all chapters")
        print("  python batch.py novels/shadow_slave 1-50     # chapters 1 to 50")
        print("  python batch.py novels/shadow_slave 51-100   # chapters 51 to 100")
        print("  python batch.py novels/shadow_slave 25-25    # single chapter 25")
        sys.exit(1)

    novel_dir = sys.argv[1]
    ch_start, ch_end = None, None

    if len(sys.argv) >= 3:
        range_arg = sys.argv[2]
        if "-" in range_arg:
            parts = range_arg.split("-", 1)
            try:
                ch_start = int(parts[0])
                ch_end   = int(parts[1])
            except ValueError:
                print(f"✗ Invalid range: {range_arg}")
                print(f"  Expected format: start-end  (e.g. 1-50)")
                sys.exit(1)
        else:
            print(f"✗ Invalid range: {range_arg}")
            print(f"  Expected format: start-end  (e.g. 1-50)")
            sys.exit(1)

    run_batch(novel_dir, ch_start, ch_end)