# PROJECT CONTEXT — AI Audiobook Generation Pipeline

> **Last updated**: 2026-03-03
> **Purpose**: Paste this file at the start of ANY new AI chat session to restore full project context.

---

## 1. What This Project Does

Production-grade AI audiobook pipeline that converts web novel `.txt` chapter files into multi-character, multi-emotion WAV/MP3 audio files. Runs unattended on Windows with GPU acceleration. Supports batch processing of 150+ chapters/day.

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
| LLM Parser | Groq API, `llama-3.3-70b-versatile`, 6 API keys round-robin |
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
GEMINI_API_KEY=AIza...
```

### requirements.txt:
```
librosa, soundfile, scipy, pydub, numpy, torch, TTS, groq,
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
├── voices.md                     # Profiles for all 29 castable XTTS voices
│
├── ── CORE PIPELINE ──────────────────────────────────────────────────
├── main.py                       # Chapter orchestration, fallback render loop
├── batch.py                      # Checkpoint-aware batch runner + chapter combiner
├── parser.py                     # Groq LLM parser, JSON repair, coverage verify+repair
├── registry.py                   # Speaker→voice resolution, gender-aware, Gemini cast aware
├── renderer.py                   # XTTS v2 inference, tone profiles, 200-char auto-split
├── assembler.py                  # Audio concat with pause rules
├── fallback.py                   # 5-stage fallback render engine
│
├── ── NOVEL MANAGER ──────────────────────────────────────────────────
├── novel_manager/
│   ├── __init__.py
│   ├── gemini_client.py          # Gemini wrapper: call_gemini() + call_gemini_with_search()
│   ├── novel_utils.py            # Chapter reading, novel.md I/O, dormancy extraction
│   ├── init_novel.py             # Populate novel.md (text default, --mode web opt-in)
│   ├── update_novel.py           # Arc update every 50-100 chapters (dormancy WIP)
│   ├── add_character.py          # Add single new character to novel.md
│   └── cast_voices.py            # Gemini personality-based voice casting → speakers.json
│
├── ── UTILITIES ──────────────────────────────────────────────────────
├── setup_novel.py                # Scaffold new novel directory + novel.md template
├── scrape_chapter.py             # Stealth web scraper with batch support
├── train_voice.py                # Custom voice training → injects into speakers_xtts.pth
├── diagnose_coverage.py          # Parser word coverage checker
├── inspect_parser.py             # Full Groq response debugger
├── Speakers.py                   # Prints all XTTS v2 speaker names
│
└── novels/
    └── {novel_slug}/
        ├── novel.md              # Living character/world document
        ├── input/chapter_N.txt
        ├── output/chapter_N.wav
        └── data/
            ├── speakers.json     # Voice assignments (cast_by: gemini or sequential)
            ├── checkpoint.json
            ├── failures.log
            ├── chapter_N_failures.json
            └── novel_backups/
```

---

## 5. XTTS v2 Configuration

**Loading method:** `XttsConfig` + `Xtts.init_from_config` + `load_checkpoint(eval=True)`
**Speaker access:** `model.speaker_manager.speakers[name]["gpt_cond_latent"]` and `["speaker_embedding"]`

### Voice Pool (29 castable + 2 fixed):

**Fixed (never assigned to characters):**
- `Ana Florence` → narrator always
- `Nova Hogarth` → system always

**Female (19 castable):** Gracie Wise, Sofia Hellen, Tanja Adelina, Barbora MacLean, Szofi Granger, Claribel Dervla, Daisy Studious, Tammie Ema, Alison Dietlinde, Annmarie Nele, Brenda Stern, Gitta Nikolina, Tammy Grit, Chandra MacFarland, Camilla Holmström, Lilya Stainthorpe, Zofija Kendrick, Narelle Moon, Rosemary Okafor

**Male (10 castable):** Damien Black, Craig Gutsy, Torcull Diarmuid, Ludvig Milivoj, Baldur Sanjin, Zacharie Aimilios, Andrew Chipper, Dionisio Schuyler, Abrahan Mack, Viktor Menelaos

### Tone Profiles (speed locked to 1.1 globally, pitch disabled):

**CRITICAL FINDING:** Speed variation was causing voice identity bleed across tones. Speed is now **locked to 1.1 for ALL 16 tone profiles**. Emotional variation is achieved solely through `temperature` (0.3–0.85), `repetition_penalty`, and `top_p`. `length_penalty` hard-locked to 1.0.

16 valid tones: neutral, calm, tense, whisper, angry, sad, excited, cold, + 8 more expanded tones.

### Renderer behavior:
- 200-char limit: auto-splits long segments at sentence boundaries, rejoins with 80ms gap
- `gpt_cond_latent` and `speaker_embedding` held 100% constant across all segments for voice consistency

---

## 6. Groq Parser Configuration

- Model: `llama-3.3-70b-versatile` (staying on this — evaluated llama-4-scout, rejected due to TPM limits and JSON accuracy concerns)
- 6 API keys: dynamic round-robin from `.env`
- `max_tokens`: 8192
- On rate limit: smart per-key tracking, exponential backoff with parsed `retry-after` times (e.g., `12.5s`, `1m30s`)
- On transient errors (502/503/timeout): incremental `time.sleep()` with exponential fallback instead of crash
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
2. `_repair_json()`: fixes unescaped quotes, trailing commas, unclosed arrays
3. `_extract_partial_segments()`: regex extracts individual `{..}` objects independently

---

## 7. Fallback Engine (5 stages in fallback.py)

1. Normal render × 3 retries
2. Cleaned text (strip non-ASCII) × 3 retries
3. Fallback speaker Ana Florence × 3 retries
4. Split into 150-char parts, render individually
5. Audible placeholder (440Hz tone) + log to `data/{chapter}_failures.json`

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

### update_novel.py
- Arc update — auto-detects new chapters since `last_updated_chapter`
- Appends to `arc_notes`, never overwrites, upgrades confidence levels
- **WIP: Smart Voice Slot Dormancy** (see Known Issues)

### add_character.py
- Targeted single character entry with preview + confirmation
- Usage: `python novel_manager/add_character.py novels/slug --character name`

### cast_voices.py
- Gemini reads character profiles from `novel.md` + voice profiles from `voices.md`
- Returns personality-matched voice assignments
- Validates: correct names, no reserved voices, no duplicates, gender mismatch warnings
- Usage: `python novel_manager/cast_voices.py novels/slug [--dry-run] [--recast-all] [--character name]`

### train_voice.py (NEW)
- Custom voice training from `.wav`/`.mp3` clips
- Calculates embeddings (latent conditioning + speaker values)
- Injects directly into `speakers_xtts.pth`
- Auto-appends entries to `voices.md`
- Generates test audio in `voice_tests/`

---

## 11. speakers.json Format

```json
{
  "novel_dir": "...",
  "characters": {
    "narrator":  { "xtts_speaker": "Ana Florence" },
    "system":    { "xtts_speaker": "Nova Hogarth" },
    "idan":      { "xtts_speaker": "Damien Black", "gender": "male", "cast_by": "gemini" },
    "arabel":    { "xtts_speaker": "Gracie Wise", "gender": "female", "cast_by": "gemini" }
  },
  "female_pool_index": 1,
  "male_pool_index": 0
}
```

- `cast_by: "gemini"` = optimally cast, don't reassign
- `cast_by: "sequential"` or missing = fallback assignment, should run `cast_voices.py`
- Gender-aware pools: separate `FEMALE_POOL` and `MALE_POOL` with independent indices

---

## 12. Current Novel Being Processed

- **Novel:** Supreme Husband System and Supreme Wife System by Ongoing Expert
- **Slug:** `shs_and_sws`
- **Path:** `novels/shs_and_sws/`
- **Status:** Mid-batch, stable, no crashes reported

---

## 13. Known Issues & WIP

### ✅ Resolved
| Issue | Resolution |
|-------|-----------|
| Voice identity bleed across tones | Speed locked to 1.1 globally, emotion via temperature/top_p only |
| Parser JSON failures from unescaped quotes | `_repair_json()` + 3-stage recovery + system prompt instruction |
| `google.generativeai` deprecated | Migrated to `google-genai` SDK |
| XTTS pitch_shift broken (librosa ufunc) | Pitch set to 0 in all profiles |
| Camilla Holmström UTF-8 encoding crash | Sanitized in speakers.json |
| Windows background terminal GPU throttling | `ctypes` OS priority boost in batch.py |
| Male characters getting female voices | Gender-aware pool assignment |
| Voice slot waste on unnamed speakers | Parser assigns unknown speakers to `narrator` |
| Unicode crash in Windows CMD | ASCII-safe console output in scraper |

### 🔧 Work In Progress
| Issue | Status |
|-------|--------|
| **Smart Voice Slot Dormancy** | Designed but NOT fully wired. `novel_utils.py` has `extract_dormant_characters()`. `update_novel.py` system prompt and `prune_speakers_json()` are **incomplete**. Importance-tiered: Protagonist=NEVER, S-Tier=200ch, B/C/D=100/30ch dormancy delays. |
| Together AI migration | Evaluated as Groq alternative (identical model, expanded limits). Not implemented yet. |

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
python scrape_chapter.py novel_slug <START_URL> -n 12

# ── Custom voice training ──
python train_voice.py <voice_name> <audio_file.wav>

# ── Debugging ──
python inspect_parser.py novels/novel_name/input/chapter_N.txt
python diagnose_coverage.py novels/novel_name/input/chapter_N.txt
python main.py novels/novel_name/input/chapter_1.txt
```

---

## 15. Current Task / What I'm Working On Now

> [UPDATE THIS SECTION every time you switch AI sessions]

```
[Describe what you need help with next here]
```