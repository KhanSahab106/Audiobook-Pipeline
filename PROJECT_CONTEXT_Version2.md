# PROJECT CONTEXT — AI Audiobook Generation Pipeline

> **Last updated**: 2026-03-23
> **Purpose**: Paste this file at the start of ANY new AI chat session to restore full project context.

---

## 1. What This Project Does

Production-grade AI audiobook pipeline that converts web novel `.txt` chapter files into multi-character, multi-emotion WAV/MP3 audio files. Multi-speaker rendering with 56+ trained voices, overflow system for unlimited character support. Runs unattended on Windows with GPU acceleration. Supports batch processing of 150+ chapters/day.

**End-to-end flow:**
1. Scrape chapters from web → `input/chapter_N.txt`
2. LLM parses chapter text into speaker-attributed, tone-tagged segments (JSON)
3. Novel manager (Gemini) maintains character database & casts voices
4. XTTS v2 renders each segment with the correct voice + tone
5. Assembler concatenates segments → single chapter WAV
6. Batch runner processes all chapters with checkpoint recovery
7. Combiner merges every 10 chapters → MP3 blocks

---

## 2. Tech Stack

| Component | Technology |
|-----------|-----------|
| TTS Engine | Coqui XTTS v2 (TTS==0.22.0), loaded via `XttsConfig` + `Xtts` classes directly |
| GPU | RTX 4050 Laptop, CUDA 12.1, ~212s per chapter |
| LLM Parser | Groq API, `llama-3.3-70b-versatile`, 6 API keys round-robin + OpenRouter fallback |
| Novel Manager | Gemini 2.5 Flash (`google-genai` SDK) with Google Search grounding |
| Web Scraper | `curl_cffi` (TLS impersonation) + `cloudscraper` fallback + `requests` fallback |
| Audio | pydub, soundfile, scipy, numpy, librosa |
| Python | 3.10.11, venv |
| OS | Windows |

---

## 3. Environment & Paths

```
Project root:  C:\Users\ABRAR\Desktop\audiobook_pipeline\
Venv:          C:\Users\ABRAR\Desktop\audiobook_pipeline\venv
XTTS model:    C:\Users\ABRAR\AppData\Local\tts\tts_models--multilingual--multi-dataset--xtts_v2
```

### .env keys required:
```
GROQ_API_KEY_1=gsk_...
GROQ_API_KEY_2=gsk_...
GROQ_API_KEY_3=gsk_...
GROQ_API_KEY_4=gsk_...
GROQ_API_KEY_5=gsk_...
GROQ_API_KEY_6=gsk_...
OPENROUTER_API_KEY=sk-or-...   # fallback when all Groq keys rate-limited
GEMINI_API_KEY=AIza...
```

### requirements.txt:
```
librosa, soundfile, scipy, pydub, numpy, torch, TTS, groq, openai,
python-dotenv, google-genai, curl_cffi, cloudscraper
```

### CUDA Notes:
- `CUDA_VISIBLE_DEVICES=0` must be set in terminal before running
- `torch.cuda.is_available() = True` confirmed
- PyTorch CUDA version: 12.1

---

## 4. File Structure

```
audiobook_pipeline/
├── .env
├── .gitignore
├── requirements.txt
├── voices.md                     # Profiles for all 56+ castable XTTS voices (built-in + custom trained)
│
├── ── CORE PIPELINE ──────────────────────────────────────────────────
├── main.py                       # Chapter orchestration, multi-speaker render loop
├── batch.py                      # Checkpoint-aware batch runner + chapter combiner
├── parser.py                     # Groq LLM parser, JSON repair, OpenRouter fallback
├── registry.py                   # Speaker→voice resolution stub
├── renderer.py                   # XTTS v2 inference, tone profiles, speed_variant support
├── assembler.py                  # Audio concat with pause rules
├── fallback.py                   # 4-stage fallback render engine with speed_variant pass-through
│
├── ── NOVEL MANAGER ──────────────────────────────────────────────────
├── novel_manager/
│   ├── __init__.py
│   ├── gemini_client.py          # Gemini wrapper: call_gemini() + call_gemini_with_search()
│   ├── novel_utils.py            # Chapter reading, novel.md I/O, dormancy extraction
│   ├── init_novel.py             # Populate novel.md (text default, --mode web opt-in)
│   ├── update_novel.py           # Patch-based arc update (Gemini outputs changes, Python applies)
│   ├── add_character.py          # Add single new character to novel.md
│   ├── cast_voices.py            # Gemini voice casting + overflow (speed-variant reuse)
│   └── audit_characters.py       # Detect & remove ghost/battle-group characters
│
├── ── UTILITIES ──────────────────────────────────────────────────────
├── setup_novel.py                # Scaffold new novel directory + novel.md template
├── utilities/
│   ├── scrape_chapter.py         # Stealth web scraper with batch support
│   ├── train_voice.py            # Custom voice training → injects into speakers_xtts.pth
│   ├── diagnose_coverage.py      # Parser word coverage checker
│   ├── inspect_parser.py         # Full Groq response debugger
│   └── Speakers.py               # Prints all XTTS v2 speaker names
│
└── novels/
    └── {novel_slug}/
        ├── novel.md              # Living character/world document
        ├── input/chapter_N.txt
        ├── output/chapter_N.wav
        └── data/
            ├── speakers.json     # Voice assignments (cast_by: gemini/overflow/sequential)
            ├── checkpoint.json
            ├── failures.log
            ├── chapter_N_failures.json
            └── novel_backups/
```

---

## 5. XTTS v2 Configuration

**Loading method:** `XttsConfig` + `Xtts.init_from_config` + `load_checkpoint(eval=True)`
**Speaker access:** `model.speaker_manager.speakers[name]["gpt_cond_latent"]` and `["speaker_embedding"]`

### Voice Pool (56+ castable + 3 fixed):

**Fixed (never assigned to characters):**
- `Ana Florence` → narrator always
- `Nova Hogarth` → system always
- `Narelle Moon` → unknown/uncast speakers

**Built-in voices:** 29 XTTS v2 speakers (19 female, 10 male)
**Custom trained voices:** 27+ (female-1 through female-20, male-1 through male-7+)

All voices catalogued in `voices.md` with gender, age, accent, tone, best-use.
`cast_voices.py` reads `voices.md` dynamically — any voice added via `train_voice.py` is auto-recognised.

### Tone Profiles (speed locked to 1.1 globally, pitch disabled):

**CRITICAL FINDING:** Speed variation was causing voice identity bleed across tones. Speed is now **locked to 1.1 for ALL 16 tone profiles**. Emotional variation is achieved solely through `temperature` (0.3–0.85), `repetition_penalty`, and `top_p`. `length_penalty` hard-locked to 1.0.

**However**, the overflow system deliberately uses speed variation (`speed_variant`) as a feature — when applied per-character (not per-tone), it creates distinct voice identities from the same base voice. Speed variant multiplies with tone speed: `final_speed = tone_speed * speed_variant`.

16 valid tones: neutral, calm, tense, whisper, angry, sad, excited, cold, + 8 more expanded tones.

### Renderer behavior:
- **Multi-speaker**: loads `speakers.json`, resolves each segment's speaker to XTTS voice + optional speed_variant
- 200-char limit: auto-splits long segments at sentence boundaries, rejoins with 80ms gap
- `gpt_cond_latent` and `speaker_embedding` held 100% constant across all segments for voice consistency

---

## 6. Groq Parser Configuration

- Model: `llama-3.3-70b-versatile` (staying on this — evaluated llama-4-scout, rejected due to TPM limits and JSON accuracy concerns)
- 6 API keys: dynamic round-robin from `.env`
- **OpenRouter fallback**: when all 6 Groq keys are rate-limited, auto-falls back to `meta-llama/llama-3.3-70b-instruct` via OpenRouter API (no TPM limit, slower). Includes continuation loop for truncated responses.
- `max_tokens`: 8192
- On rate limit: smart per-key tracking, exponential backoff with parsed `retry-after` times
- On transient errors (502/503/timeout): incremental `time.sleep()` with exponential fallback
- On truncation (`finish_reason == "length"`): continuation call via `_continue_truncated()`
- Token tracking per chapter: input, output, total, calls

### Parser Output Format:
JSON array of segments: `{index, speaker, type, tone, text}`
- Types: `dialogue | narration | action | thought`
- Tones: 16 valid emotional states
- Merge threshold: 120 chars (same speaker+tone+type)
- Unknown/unnamed speakers → assigned to `narrator` (saves voice slots)
- Pronoun speaker names resolved to narrator

### JSON Repair (3-stage recovery):
1. Direct `json.loads()` parse
2. `_repair_json()`: fixes unescaped quotes, trailing commas, unclosed arrays, missing index values
3. `_extract_partial_segments()`: per-segment schema-aware extraction via `_fix_segment_object()` — extracts each field individually so unescaped quotes in text don't poison other segments

---

## 7. Fallback Engine (4 stages in fallback.py)

All stages pass `speed_variant` through to maintain character voice identity.

1. Normal render × 3 retries (with character's voice + speed_variant)
2. Cleaned text (strip non-ASCII, interjection normalization, trailing punctuation removal) × 3 retries
3. Split into 150-char parts, render individually
4. Audible placeholder (440Hz tone) + log to `data/{chapter}_failures.json`

---

## 8. Batch Processing (batch.py)

- Reads `novels/{slug}/input/chapter_*.txt` sorted
- Supports range filtering: `python batch.py novels/novel_name 1-50`
- Checkpoint in `data/checkpoint.json` — resumes on crash
- TTS model loaded once, reused across all chapters
- Failed chapters logged to `data/failures.log`
- **Chapter combiner** built-in: groups contiguous WAVs into 10-chapter MP3 blocks (e.g., `chapters_1_to_10.mp3`) with 2000ms inter-chapter silence
- **Windows process priority boost** via `ctypes` to prevent GPU throttling when terminal is in background (critical for overnight runs)

---

## 9. Web Scraper (scrape_chapter.py)

- Batch scraping: `-n` flag for consecutive chapters
- Auto-follows `a#next_chap` "Next Chapter" links
- Passes `Referer` header for natural navigation simulation
- **Anti-detection stack:**
  - Primary: `curl_cffi` with Chrome TLS/JA3 fingerprint impersonation (rotating chrome116→chrome131)
  - Fallback 1: `cloudscraper` (JS challenge solving)
  - Fallback 2: `requests`
- Perfect browser header order: `User-Agent`, `Sec-Ch-Ua`, `Sec-Fetch-Site`, `Accept-Language`
- Rate limiting: 15-25s random delay between chapters, 10-minute cooldown every 10 chapters
- ASCII-safe console output (no Unicode box-drawing chars on Windows CMD)

Usage: `python scrape_chapter.py shs_and_sws <URL> -n 12`

---

## 10. Novel Manager Sub-module

### gemini_client.py
- `call_gemini(prompt, label)`: standard generation
- `call_gemini_with_search(prompt, label)`: Google Search grounding
- SDK: `google-genai` (NOT deprecated `google-generativeai`)
- Client: `genai.Client(api_key=...) + client.models.generate_content()`

### init_novel.py
- Default mode: reads local chapter files → Gemini analysis
- Web mode (`--mode web`): Gemini searches internet, auto-fallback to text if <60% coverage
- Usage: `python novel_manager/init_novel.py novels/slug --chapters 1-10`

### update_novel.py (PATCH-BASED ARCHITECTURE)
- Arc update — auto-detects new chapters since `last_updated_chapter`
- **Gemini outputs ONLY changes** (structured patch block), Python applies them surgically
- Characters can NEVER be silently dropped — Gemini never writes the full character list
- Patch commands: `UPDATE_CHARACTER`, `NEW_CHARACTER`, `MARK_DORMANT`, `REACTIVATE`, `PRUNE`, `UPDATE_SECTION`, `APPEND_CHAPTER_MAP`
- Smart Voice Slot Dormancy: importance-tiered thresholds (Protagonist=NEVER, S-Tier=200ch, A-Tier=150ch, B-Tier=100ch, C-Tier=50ch)
- `sync_speakers_json()`: frees voice slots for dormant characters, restores reactivated ones

### add_character.py
- Targeted single character entry with preview + confirmation
- Usage: `python novel_manager/add_character.py novels/slug --character name`

### cast_voices.py
- Gemini reads character profiles from `novel.md` + voice profiles from `voices.md`
- Returns personality-matched voice assignments
- Validates: correct names, no reserved voices, no duplicates, gender mismatch warnings
- **Overflow system**: when all unique voices are used up, reuses voices with speed modifiers
  - 6 speed offsets: `[0.85, 1.15, 0.88, 1.12, 0.92, 1.08]` per base voice
  - Never reuses protagonist/deuteragonist voices
  - Gender-matched when possible
  - **56 base voices × 6 variants = 336 effective voice slots**
- Usage: `python novel_manager/cast_voices.py novels/slug [--dry-run] [--recast-all] [--character name]`

### audit_characters.py
- Detects problematic character entries that waste voice slots:
  - **Battle-group NPCs**: numbered/generic enemies detected by name pattern
  - **Ghost characters**: no dialogue evidence in arc_notes
  - **Suspicious LUC**: fabricated last_updated_chapter values
- Bulk removal with backup: `python novel_manager/audit_characters.py novels/slug --remove all`

### train_voice.py
- Custom voice training from `.wav`/`.mp3` clips
- Silence trimming + clip filtering (min 1.5s after trim, max 50 clips)
- Calculates embeddings (latent conditioning + speaker values)
- Injects directly into `speakers_xtts.pth`
- Auto-appends entries to `voices.md` (regex-based, section-aware)
- Generates test audio in `voice_tests/`
- **Auto-reject**: if test voice > 12s, removes from model + voices.md + test file

---

## 11. speakers.json Format

```json
{
  "novel_dir": "...",
  "characters": {
    "narrator":  { "xtts_speaker": "Ana Florence" },
    "system":    { "xtts_speaker": "Nova Hogarth" },
    "unknown":   { "xtts_speaker": "Narelle Moon" },
    "idan":      { "xtts_speaker": "Damien Black", "gender": "male", "cast_by": "gemini" },
    "arabel":    { "xtts_speaker": "Gracie Wise", "gender": "female", "cast_by": "gemini" },
    "minor_npc": { "xtts_speaker": "Craig Gutsy", "gender": "male", "cast_by": "overflow",
                   "is_overflow": true, "speed_variant": 0.85 }
  },
  "dormant_voices": {}
}
```

- `cast_by: "gemini"` = optimally cast, don't reassign
- `cast_by: "overflow"` = reused voice with speed modifier (auto-assigned when pool exhausted)
- `is_overflow: true` + `speed_variant: 0.85` = renderer applies speed multiplier on tone speed
- `dormant_voices` = voice slots freed by dormant characters, restored on reactivation

---

## 12. Current Novel Being Processed

- **Novel:** Supreme Husband System and Supreme Wife System by Ongoing Expert
- **Slug:** `shs_and_sws`
- **Path:** `novels/shs_and_sws/`
- **Status:** Mid-batch, stable, no crashes reported

---

## 13. Known Issues & WIP

### Resolved
| Issue | Resolution |
|-------|-----------|
| Voice identity bleed across tones | Speed locked to 1.1 globally, emotion via temperature/top_p only |
| Parser JSON failures from unescaped quotes | 3-stage recovery + per-segment schema extraction (`_fix_segment_object`) |
| `google.generativeai` deprecated | Migrated to `google-genai` SDK |
| XTTS pitch_shift broken (librosa ufunc) | Pitch set to 0 in all profiles |
| Camilla Holmstrom UTF-8 encoding crash | ASCII-folded in voices.md + `ensure_ascii=True` in JSON writes |
| Windows background terminal GPU throttling | `ctypes` OS priority boost in batch.py |
| Male characters getting female voices | Gender-aware pool assignment |
| Voice slot waste on unnamed speakers | Parser assigns unknown speakers to `narrator` |
| Groq rate limit causing crashes | OpenRouter fallback (`meta-llama/llama-3.3-70b-instruct`) auto-activates |
| Gemini silently dropping characters | Patch-based `update_novel.py` — Gemini outputs changes only, Python applies |
| Voice pool exhaustion on 50+ characters | Overflow system: speed-variant voice reuse (336 effective slots) |
| Ghost/NPC characters wasting voice slots | `audit_characters.py` detects and bulk-removes them |
| Custom voices not recognized by caster | `cast_voices.py` dynamically loads from `voices.md` |
| XTTS interjection stretching ("Ahh...") | `clean_text_for_tts()` inserts comma after standalone interjections |
| Trailing punctuation gibberish | `clean_text_for_tts()` strips trailing `?!.` from sentences |

---

## 14. Workflow Commands (Quick Reference)

```bash
# ── New novel setup ──
python setup_novel.py novel_name
# copy chapters to novels/novel_name/input/
python novel_manager/init_novel.py novels/novel_name --chapters 1-10
python novel_manager/cast_voices.py novels/novel_name
python batch.py novels/novel_name

# ── Batch with range ──
python batch.py novels/novel_name 1-50

# ── Ongoing maintenance (every 50-100 chapters) ──
python novel_manager/update_novel.py novels/novel_name
python novel_manager/cast_voices.py novels/novel_name

# ── New character appeared ──
python novel_manager/add_character.py novels/novel_name --character name
python novel_manager/cast_voices.py novels/novel_name --character name

# ── Scraping ──
python utilities/scrape_chapter.py novel_slug <START_URL> -n 12

# ── Custom voice training ──
python utilities/train_voice.py <voice_name> <audio_dir>/

# ── Character audit (find ghost/NPC clutter) ──
python novel_manager/audit_characters.py novels/novel_name --luc-suspect 350
python novel_manager/audit_characters.py novels/novel_name --remove battle_group

# ── Debugging ──
python utilities/inspect_parser.py novels/novel_name/input/chapter_N.txt
python utilities/diagnose_coverage.py novels/novel_name/input/chapter_N.txt
python main.py novels/novel_name/input/chapter_1.txt
```

---

## 15. Current Task / What I'm Working On Now

> [UPDATE THIS SECTION every time you switch AI sessions]

```
Multi-speaker rendering and overflow system implemented and tested.
All characters now use their assigned voices from speakers.json.
Overflow system auto-assigns speed-variant voices when pool is exhausted.
Next: run cast_voices.py and test multi-speaker audio output.
```