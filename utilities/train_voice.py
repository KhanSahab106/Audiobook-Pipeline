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
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

MODEL_DIR   = r"C:\Users\ABRAR\AppData\Local\tts\tts_models--multilingual--multi-dataset--xtts_v2"
SPEAKERS_FILE = os.path.join(MODEL_DIR, "speakers_xtts.pth")
VOICES_MD   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voices.md")
TEST_OUTPUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "voice_tests")
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
            audio_files.extend(glob.glob(os.path.join(p, "*.wav")))
            audio_files.extend(glob.glob(os.path.join(p, "*.mp3")))
        elif os.path.isfile(p):
            audio_files.append(p)
        else:
            print(f"  ⚠ Not found: {p}")
    return sorted(audio_files)


def compute_speaker_embedding(model, audio_files: list[str]):
    """
    Compute speaker conditioning latents from audio samples.
    Returns (gpt_cond_latent, speaker_embedding) tensors.
    """
    print(f"\n  Computing speaker embedding from {len(audio_files)} sample(s)...")
    for f in audio_files:
        print(f"    📁 {os.path.basename(f)}")

    gpt_cond_latent, speaker_embedding = model.get_conditioning_latents(
        audio_path=audio_files
    )

    print(f"  ✓ Embedding computed")
    print(f"    gpt_cond_latent shape: {gpt_cond_latent.shape}")
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


def generate_test(model, voice_name: str):
    """Generate a test audio clip with the new voice."""
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


def append_to_voices_md(voice_name: str, gender: str, age: str,
                         accent: str, tone: str, best_for: str):
    """Append a new voice entry to voices.md."""
    section = "Female Voices" if gender.lower() == "female" else "Male Voices"

    entry = f"""
### {voice_name}
- Gender: {gender}
- Age: {age}
- Accent: {accent}
- Tone: {tone}
- Best for: {best_for}
"""

    with open(VOICES_MD, "r", encoding="utf-8") as f:
        content = f.read()

    # Check if voice already exists
    if f"### {voice_name}" in content:
        print(f"  ⚠ '{voice_name}' already exists in voices.md — skipping append")
        return

    # Find the section and append before the --- separator
    if section == "Male Voices":
        # Append before the casting guidelines section
        marker = "---\n\n## Casting guidelines"
    else:
        marker = "---\n\n## Male Voices"

    if marker in content:
        content = content.replace(marker, entry.rstrip() + "\n\n" + marker)
    else:
        # Fallback: append at end
        content += entry

    with open(VOICES_MD, "w", encoding="utf-8") as f:
        f.write(content)

    print(f"  ✓ Added to voices.md under {section}")


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
    """Remove a voice from speakers_xtts.pth."""
    speakers = torch.load(SPEAKERS_FILE, map_location="cpu")
    if voice_name not in speakers:
        print(f"  ✗ Voice '{voice_name}' not found")
        print(f"  Available: {', '.join(sorted(speakers.keys()))}")
        return
    del speakers[voice_name]
    torch.save(speakers, SPEAKERS_FILE)
    print(f"  ✓ Removed '{voice_name}' from {SPEAKERS_FILE}")
    print(f"  Remaining voices: {len(speakers)}")


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

    audio_files = collect_audio_files(args.audio_paths)
    if not audio_files:
        print("✗ No audio files found in the provided paths")
        sys.exit(1)

    voice_name = args.voice_name

    print(f"\n{'═' * 55}")
    print(f"  XTTS Voice Trainer")
    print(f"{'═' * 55}")
    print(f"  Voice name : {voice_name}")
    print(f"  Gender     : {args.gender}")
    print(f"  Samples    : {len(audio_files)}")
    print(f"{'═' * 55}")

    model, config = load_model()

    # Check if voice already exists
    if voice_name in model.speaker_manager.speakers:
        print(f"\n  ⚠ Voice '{voice_name}' already exists!")
        response = input("  Overwrite? (y/N): ").strip().lower()
        if response != "y":
            print("  Aborted.")
            return

    gpt_cond_latent, speaker_embedding = compute_speaker_embedding(model, audio_files)
    add_voice_to_model(model, voice_name, gpt_cond_latent, speaker_embedding)
    save_speakers(model)

    if not args.no_test:
        generate_test(model, voice_name)

    if not args.no_voices_md:
        append_to_voices_md(
            voice_name, args.gender, args.age,
            args.accent, args.tone, args.best_for
        )

    print(f"\n{'═' * 55}")
    print(f"  ✓ Voice '{voice_name}' is ready!")
    print(f"  You can now use it in speakers.json for character casting.")
    print(f"{'═' * 55}")


if __name__ == "__main__":
    main()
