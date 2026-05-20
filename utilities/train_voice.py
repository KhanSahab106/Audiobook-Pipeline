"""
train_voice.py

Train a new XTTS v2 voice from audio sample(s) and register it in the pipeline.

Steps:
    1. Compute speaker embedding + GPT conditioning latents from audio file(s)
    2. Add the new voice to the XTTS speakers_xtts.pth file
    3. Optionally append the voice entry to voices.md
    4. Generate a test audio clip so you can preview the voice

Requirements:
    - Audio samples should be 6–30 seconds of clear speech (WAV or MP3)
    - One speaker per sample — no background music or noise
    - Longer/more samples = better voice quality

Usage:
    python train_voice.py "Voice Name" path/to/sample.wav --gender male
    python train_voice.py "Voice Name" path/to/sample1.wav path/to/sample2.wav --gender female
    python train_voice.py "Voice Name" samples/ --gender male  (all .wav/.mp3 in folder)
    python train_voice.py --list  (show all registered voices)
    python train_voice.py --remove "Voice Name"  (remove a voice)
"""

import os
import sys
import glob
import argparse
import torch
import numpy as np
import soundfile as sf
import librosa
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

MODEL_DIR   = r"C:\Users\ABRAR\AppData\Local\tts\tts_models--multilingual--multi-dataset--xtts_v2"
SPEAKERS_FILE = os.path.join(MODEL_DIR, "speakers_xtts.pth")
# Resolve the project root regardless of the working directory when the script is run.
ROOT        = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
VOICES_MD   = os.path.join(ROOT, "voices.md")
TEST_OUTPUT = os.path.join(ROOT, "voice_tests")
SAMPLE_RATE = 24000


def load_model():
    """Load XTTS v2 model."""
    use_cuda = torch.cuda.is_available()
    print(f"  Loading XTTS v2 on {'cuda' if use_cuda else 'cpu'}...")

    config = XttsConfig()
    config.load_json(os.path.join(MODEL_DIR, "config.json"))

    model = Xtts.init_from_config(config)
    model.load_checkpoint(config, checkpoint_dir=MODEL_DIR, eval=True)

    if use_cuda:
        model.cuda()

    return model, config


def collect_audio_files(paths: list[str]) -> list[str]:
    """Expand paths — files stay as-is, directories yield all .wav/.mp3 inside."""
    audio_files = []
    for p in paths:
        if os.path.isdir(p):
            audio_files.extend(glob.glob(os.path.join(p, "**", "*.wav"), recursive=True))
            audio_files.extend(glob.glob(os.path.join(p, "**", "*.mp3"), recursive=True))
        elif os.path.isfile(p):
            audio_files.append(p)
        else:
            print(f"  ⚠ Not found: {p}")
    return sorted(audio_files)


def trim_and_filter_clips(
    audio_files: list[str],
    min_duration: float = 1.5,
    max_clips: int = 50,
    silence_db: float = 30.0,
) -> list[str]:
    """
    Filter out clips that are too short after silence trimming.
    Also cap total clips at max_clips to avoid embedding averaging noise.

    XTTS performance degrades when averaging too many clips — the averaged
    embedding becomes a blurry 'mean' voice rather than a crisp identity.
    50 diverse clips is optimal; 100+ adds diminishing returns and can
    introduce pause artifacts from clips with leading/trailing silence.

    Args:
        audio_files:  List of raw audio paths
        min_duration: Minimum seconds of speech content after trimming
        max_clips:    Cap on how many clips to actually use
        silence_db:   dB below peak to treat as silence (higher = more aggressive trim)

    Returns:
        Filtered, trimmed list (saved as temp .wav files in system temp dir)
    """
    import tempfile
    import shutil

    print(f"\n  Filtering {len(audio_files)} clips (min {min_duration}s, max {max_clips})...")

    good = []
    skipped = 0
    tmp_dir = tempfile.mkdtemp(prefix="xtts_train_")

    for path in audio_files:
        if len(good) >= max_clips:
            break
        try:
            y, sr = librosa.load(path, sr=None, mono=True)
            # Trim leading/trailing silence
            y_trim, _ = librosa.effects.trim(y, top_db=silence_db)
            duration = len(y_trim) / sr

            if duration < min_duration:
                skipped += 1
                continue

            # Save trimmed version to temp dir
            out_path = os.path.join(tmp_dir, f"{len(good):04d}_{os.path.basename(path)}.wav")
            sf.write(out_path, y_trim, sr)
            good.append(out_path)

        except Exception as e:
            print(f"  ⚠ Could not process {os.path.basename(path)}: {e}")
            skipped += 1

    print(f"  ✓ Using {len(good)} clips  ({skipped} skipped — too short or unreadable)")
    if skipped > len(audio_files) * 0.5:
        print(f"  ⚠ More than half your clips were skipped. Consider lowering --min-duration.")

    return good, tmp_dir


def compute_speaker_embedding(model, audio_files: list[str]):
    """
    Compute speaker conditioning latents from audio samples.
    Returns (gpt_cond_latent, speaker_embedding) tensors.
    """
    print(f"\n  Computing speaker embedding from {len(audio_files)} clip(s)...")

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=audio_files
    )

    print(f"  ✓ Embedding computed")
    print(f"    gpt_cond_latent shape : {gpt_cond_latent.shape}")
    print(f"    speaker_embedding shape: {speaker_embedding.shape}")

    return gpt_cond_latent, speaker_embedding


def add_voice_to_model(model, voice_name: str, gpt_cond_latent, speaker_embedding):
    """Register the voice in the model's speaker manager."""
    model.speaker_manager.speakers[voice_name] = {
        "gpt_cond_latent": gpt_cond_latent,
        "speaker_embedding": speaker_embedding,
    }
    print(f"  ✓ Voice '{voice_name}' added to model")


def save_speakers(model):
    """Persist all speakers (including new one) to speakers_xtts.pth."""
    # Build the save dict in the same format XTTS expects
    torch.save(model.speaker_manager.speakers, SPEAKERS_FILE)
    print(f"  ✓ Saved to {SPEAKERS_FILE}")


def generate_test(model, voice_name: str) -> float:
    """Generate a test audio clip. Returns duration in seconds."""
    os.makedirs(TEST_OUTPUT, exist_ok=True)
    safe_name = voice_name.replace(" ", "_").lower()
    out_path  = os.path.join(TEST_OUTPUT, f"test_{safe_name}.wav")

    test_text = (
        "Hello, this is a test of my new voice. "
        "I can speak clearly and naturally, with good pacing and intonation. "
        "How does it sound?"
    )

    print(f"\n  Generating test audio...")
    gpt_cond_latent   = model.speaker_manager.speakers[voice_name]["gpt_cond_latent"]
    speaker_embedding = model.speaker_manager.speakers[voice_name]["speaker_embedding"]

    out = model.inference(
        text=test_text,
        language="en",
        gpt_cond_latent=gpt_cond_latent,
        speaker_embedding=speaker_embedding,
        temperature=0.65,
        length_penalty=1.0,
        repetition_penalty=2.0,
        top_k=50,
        top_p=0.85,
        enable_text_splitting=False,
    )

    wav = np.array(out["wav"], dtype=np.float32)
    sf.write(out_path, wav, SAMPLE_RATE)
    duration = len(wav) / SAMPLE_RATE
    print(f"  ✓ Test saved: {out_path} ({duration:.1f}s)")
    return duration


def append_to_voices_md(voice_name: str, gender: str, age: str,
                         accent: str, tone: str, best_for: str):
    """Append a new voice entry to voices.md in the correct section."""
    import re as _re

    entry = (
        f"\n### {voice_name}\n"
        f"- Gender: {gender}\n"
        f"- Age: {age}\n"
        f"- Accent: {accent}\n"
        f"- Tone: {tone}\n"
        f"- Best for: {best_for}\n"
    )

    with open(VOICES_MD, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if voice already exists
    if f"### {voice_name}" in content:
        print(f"  ⚠ '{voice_name}' already exists in voices.md — skipping append")
        return

    if gender.lower() == "female":
        # Insert just before ## Male Voices (with any amount of preceding ---)
        pattern = _re.compile(r'(?=\n(?:-+\n+)?## Male Voices)', _re.MULTILINE)
        new_content, n = pattern.subn(entry, content, count=1)
        section = "Female Voices"
    else:
        # Insert just before ## Casting guidelines (with any amount of preceding ---)
        pattern = _re.compile(r'(?=\n(?:-+\n+)?## Casting guidelines)', _re.MULTILINE)
        new_content, n = pattern.subn(entry, content, count=1)
        section = "Male Voices"

    if n:
        with open(VOICES_MD, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"  ✓ Added to voices.md under {section}")
    else:
        # Fallback: append at end
        with open(VOICES_MD, "a", encoding="utf-8") as f:
            f.write(entry)
        print(f"  ✓ Added to voices.md (appended at end — section marker not found)")


def list_voices():
    """List all registered XTTS voices."""
    speakers = torch.load(SPEAKERS_FILE, map_location="cpu")
    print(f"\n{'═' * 55}")
    print(f"  Registered XTTS Voices ({len(speakers)})")
    print(f"{'═' * 55}")
    for i, name in enumerate(sorted(speakers.keys()), 1):
        print(f"  {i:3d}. {name}")
    print(f"{'═' * 55}")


def remove_voice(voice_name: str):
    """Remove a voice from speakers_xtts.pth, its test file, and voices.md entry."""
    import re as _re

    # ── 1. Remove from speakers_xtts.pth ─────────────────────
    speakers = torch.load(SPEAKERS_FILE, map_location="cpu")
    if voice_name not in speakers:
        print(f"  ✗ Voice '{voice_name}' not found in speakers file")
        print(f"  Available: {', '.join(sorted(speakers.keys()))}")
        return
    del speakers[voice_name]
    torch.save(speakers, SPEAKERS_FILE)
    print(f"  ✓ Removed from speakers_xtts.pth  ({len(speakers)} voices remain)")

    # ── 2. Delete test audio file ─────────────────────────────
    safe_name  = voice_name.replace(" ", "_").lower()
    test_path  = os.path.join(TEST_OUTPUT, f"test_{safe_name}.wav")
    if os.path.exists(test_path):
        os.remove(test_path)
        print(f"  ✓ Deleted test file: {test_path}")
    else:
        print(f"  – No test file found at: {test_path}")

    # ── 3. Remove entry from voices.md ───────────────────────
    if not os.path.exists(VOICES_MD):
        print(f"  – voices.md not found, skipping")
        return

    with open(VOICES_MD, "r", encoding="utf-8") as f:
        content = f.read()

    # Match the ### heading and everything up to the next ### or ## or EOF
    pattern = _re.compile(
        rf'^### {_re.escape(voice_name)}\s*\n.*?(?=^###\s|^##\s|\Z)',
        flags=_re.MULTILINE | _re.DOTALL
    )
    new_content, n = pattern.subn("", content)

    if n:
        # Clean up any double blank lines left behind
        new_content = _re.sub(r'\n{3,}', '\n\n', new_content)
        with open(VOICES_MD, "w", encoding="utf-8") as f:
            f.write(new_content)
        print(f"  ✓ Removed entry from voices.md")
    else:
        print(f"  – Entry '### {voice_name}' not found in voices.md")


def main():
    parser = argparse.ArgumentParser(
        description="Train and register new XTTS v2 voices"
    )
    parser.add_argument(
        "voice_name", nargs="?",
        help="Name for the new voice (e.g. 'Marcus Steel')"
    )
    parser.add_argument(
        "audio_paths", nargs="*",
        help="Audio file(s) or directory of samples"
    )
    parser.add_argument("--gender", default="unknown",
                        help="Voice gender: male / female / unknown")
    parser.add_argument("--age", default="adult, 30s",
                        help="Voice age description")
    parser.add_argument("--accent", default="neutral",
                        help="Voice accent description")
    parser.add_argument("--tone", default="clear, natural",
                        help="Voice tone description")
    parser.add_argument("--best-for", default="general characters",
                        help="Best character types for this voice")
    parser.add_argument("--no-test", action="store_true",
                        help="Skip generating test audio")
    parser.add_argument("--no-voices-md", action="store_true",
                        help="Skip appending to voices.md")
    parser.add_argument("--list", action="store_true",
                        help="List all registered voices and exit")
    parser.add_argument("--remove", metavar="NAME",
                        help="Remove a voice by name and exit")
    parser.add_argument("--min-duration", type=float, default=1.5, metavar="SECS",
                        help="Minimum clip duration after silence trimming (default: 1.5s)")
    parser.add_argument("--max-clips", type=int, default=50, metavar="N",
                        help="Max number of clips to use for embedding (default: 50)")
    parser.add_argument("--silence-db", type=float, default=30.0, metavar="DB",
                        help="Silence trim aggressiveness in dB (default: 30 — higher trims more)")
    parser.add_argument("--max-test-duration", type=float, default=12.0, metavar="SECS",
                        help="Auto-reject voice if test audio exceeds this duration in seconds (default: 12.0). "
                             "Long test = excessive pauses = bad embedding. Use --max-test-duration 0 to disable.")

    args = parser.parse_args()

    # ── List mode ─────────────────────────────────
    if args.list:
        list_voices()
        return

    # ── Remove mode ───────────────────────────────
    if args.remove:
        remove_voice(args.remove)
        return

    # ── Train mode ────────────────────────────────
    if not args.voice_name:
        parser.print_help()
        sys.exit(1)

    if not args.audio_paths:
        print("✗ Provide at least one audio file or directory")
        sys.exit(1)

    raw_files = collect_audio_files(args.audio_paths)
    if not raw_files:
        print("✗ No audio files found in the provided paths")
        sys.exit(1)

    voice_name = args.voice_name

    print(f"\n{'═' * 55}")
    print(f"  XTTS Voice Trainer")
    print(f"{'═' * 55}")
    print(f"  Voice name   : {voice_name}")
    print(f"  Gender       : {args.gender}")
    print(f"  Raw clips    : {len(raw_files)}")
    print(f"  Min duration : {args.min_duration}s after silence trim")
    print(f"  Max clips    : {args.max_clips}")
    print(f"{'═' * 55}")

    # Filter and trim clips before loading model (fast CPU step)
    audio_files, tmp_dir = trim_and_filter_clips(
        raw_files,
        min_duration=args.min_duration,
        max_clips=args.max_clips,
        silence_db=args.silence_db,
    )
    if not audio_files:
        print("✗ No usable clips after filtering. Lower --min-duration or check your audio.")
        sys.exit(1)

    model, config = load_model()

    # Check if voice already exists
    if voice_name in model.speaker_manager.speakers:
        print(f"\n  ⚠ Voice '{voice_name}' already exists!")
        response = input("  Overwrite? (y/N): ").strip().lower()
        if response != "y":
            print("  Aborted.")
            import shutil; shutil.rmtree(tmp_dir, ignore_errors=True)
            return

    gpt_cond_latent, speaker_embedding = compute_speaker_embedding(model, audio_files)
    add_voice_to_model(model, voice_name, gpt_cond_latent, speaker_embedding)
    save_speakers(model)

    if not args.no_test:
        test_duration = generate_test(model, voice_name)

        # ── Auto-reject if test is too long (indicates pause artifacts) ──
        max_dur = args.max_test_duration
        if max_dur > 0 and test_duration > max_dur:
            print(f"\n  ✗ REJECTED: test duration {test_duration:.1f}s > {max_dur}s limit")
            print(f"  Voice has excessive pauses — removing from all locations.")

            # remove_voice() handles: speakers_xtts.pth + test file + voices.md
            remove_voice(voice_name)

            import shutil
            shutil.rmtree(tmp_dir, ignore_errors=True)

            print(f"  Tip: try fewer clips (--max-clips 20) or longer samples to reduce pauses.")
            sys.exit(1)
    else:
        test_duration = None

    if not args.no_voices_md:
        append_to_voices_md(
            voice_name, args.gender, args.age,
            args.accent, args.tone, args.best_for
        )

    # Clean up temp trimmed files
    import shutil
    shutil.rmtree(tmp_dir, ignore_errors=True)

    print(f"\n{'═' * 55}")
    print(f"  ✓ Voice '{voice_name}' is ready!")
    print(f"  You can now use it in speakers.json for character casting.")
    print(f"{'═' * 55}")


if __name__ == "__main__":
    main()
