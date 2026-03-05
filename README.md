# 🎧 AI Audiobook Generation Pipeline

A production-grade, fully automated pipeline that converts web novel `.txt` chapter files into high-quality, multi-character, multi-emotion audiobooks. Runs unattended on Windows with GPU acceleration using **Coqui XTTS v2** for text-to-speech, **Groq (LLaMA 3.3 70B)** for intelligent text parsing, and **Gemini 2.5 Flash** for character management and voice casting.

---

## ✨ What It Does

1. **Scrapes** web novel chapters from any URL and saves them as `.txt` files
2. **Parses** chapter text using an LLM — every sentence is labeled with speaker, type (dialogue/narration/action/thought), and emotional tone
3. **Manages characters** — Gemini maintains a living `novel.md` database of character profiles, personalities, and voice traits
4. **Casts voices** — Gemini personality-matches each character to one of 29 XTTS v2 voices
5. **Renders audio** — XTTS v2 synthesizes each segment with the correct voice and emotion
6. **Assembles chapters** — segments are concatenated into a single WAV per chapter with proper pause rules
7. **Batch processes** — all chapters run unattended with checkpoint recovery; every 10 chapters are merged into an MP3 block

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| TTS Engine | Coqui XTTS v2 (`TTS==0.22.0`) |
| GPU | RTX 4050 Laptop, CUDA 12.1 |
| LLM Parser | Groq API — `llama-3.3-70b-versatile`, 6-key round-robin |
| Novel Manager | Gemini 2.5 Flash (`google-genai` SDK) with Google Search grounding |
| Web Scraper | `curl_cffi` + `cloudscraper` + `requests` (3-tier anti-detection) |
| Audio | `pydub`, `soundfile`, `scipy`, `numpy`, `librosa` |
| Python | 3.10.11 |
| OS | Windows |

---

## 📁 Project Structure

```
audiobook_pipeline/
├── .env                          # API keys
├── requirements.txt
├── voices.md                     # Profiles for all 29 castable XTTS voices
│
├── main.py                       # Single-chapter orchestrator
├── batch.py                      # Checkpoint-aware batch runner + chapter combiner
├── parser.py                     # Groq LLM parser — segments every sentence
├── registry.py                   # Speaker → voice resolution
├── renderer.py                   # XTTS v2 inference + tone profiles
├── assembler.py                  # Audio segment concatenation
├── fallback.py                   # 5-stage fallback render engine
│
├── novel_manager/
│   ├── init_novel.py             # Populate novel.md (run once per novel)
│   ├── update_novel.py           # Arc update every 50–100 chapters
│   ├── add_character.py          # Add a single new character
│   ├── cast_voices.py            # Gemini voice casting → speakers.json
│   ├── gemini_client.py          # Gemini API wrapper
│   └── novel_utils.py            # Shared chapter/novel.md utilities
│
├── utilities/
│   ├── setup_novel.py            # Scaffold a new novel directory
│   ├── scrape_chapter.py         # Web scraper with batch support
│   ├── train_voice.py            # Custom voice training
│   ├── diagnose_coverage.py      # Parser word coverage checker
│   ├── inspect_parser.py         # Full Groq response debugger
│   ├── Speakers.py               # List all XTTS v2 speaker names
│   ├── test_tts.py               # XTTS v2 sanity check
│   ├── network_connection_test.py # Groq API connection test
│   └── gemini.py                 # List available Gemini models
│
└── novels/
    └── {novel_slug}/
        ├── novel.md              # Living character/world document
        ├── input/chapter_N.txt
        ├── output/chapter_N.wav
        └── data/
            ├── speakers.json
            ├── checkpoint.json
            └── failures.log
```

---

## ⚙️ Prerequisites & Installation

### 1. Clone the repository

```bash
git clone https://github.com/KhanSahab106/Audiobook-Pipeline.git
cd Audiobook-Pipeline
```

### 2. Create and activate a virtual environment

```bash
python -m venv venv
venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> **Note:** PyTorch with CUDA must be installed separately. See [pytorch.org](https://pytorch.org/get-started/locally/) and install the CUDA 12.1 build.

### 4. Download the XTTS v2 model

```bash
python -c "from TTS.api import TTS; TTS('tts_models/multilingual/multi-dataset/xtts_v2')"
```

The model downloads to `%LOCALAPPDATA%\tts\tts_models--multilingual--multi-dataset--xtts_v2`.

### 5. Configure API keys

Create a `.env` file in the project root:

```env
GROQ_API_KEY_1=gsk_...
GROQ_API_KEY_2=gsk_...
GROQ_API_KEY_3=gsk_...
GROQ_API_KEY_4=gsk_...
GROQ_API_KEY_5=gsk_...
GROQ_API_KEY_6=gsk_...
GEMINI_API_KEY=AIza...
```

> You can use 1–6 Groq keys. More keys = higher throughput via round-robin.

### 6. Set CUDA environment variable (required for GPU)

Run this in your terminal **before** any pipeline command:

```bash
set CUDA_VISIBLE_DEVICES=0
```

---

## 🚀 Complete Workflow — Start to Finish

---

### STEP 1 — Set Up a New Novel

Scaffold the directory structure and create a blank `novel.md` template:

```bash
python utilities/setup_novel.py novel_name
# Example:
python utilities/setup_novel.py shadow_slave
```

This creates:
```
novels/shadow_slave/
├── input/        ← put your chapter .txt files here
├── output/       ← rendered WAVs go here
├── data/         ← checkpoints, logs, speakers.json
└── novel.md      ← character/world document (blank template)
```

---

### STEP 2 — Get Chapters

#### Option A — Scrape from a web novel site

**Scrape a single chapter:**
```bash
python utilities/scrape_chapter.py shadow_slave https://novelsite.com/chapter-1
```

**Scrape multiple consecutive chapters (e.g. 12 chapters starting from URL):**
```bash
python utilities/scrape_chapter.py shadow_slave https://novelsite.com/chapter-1 -n 12
```

> The scraper auto-follows "Next Chapter" links. Uses Chrome TLS fingerprint impersonation to bypass anti-scraping. Random 15–25s delays between chapters; 10-minute cooldown every 10 chapters.

#### Option B — Copy existing `.txt` files manually

Copy your chapter files into the input folder:
```
novels/shadow_slave/input/chapter_1.txt
novels/shadow_slave/input/chapter_2.txt
...
```

Files must be named `chapter_N.txt` where `N` is the chapter number.

---

### STEP 3 — Initialize the Novel (Run Once)

This analyzes the chapters and populates `novel.md` with character profiles.

**Default (text mode) — uses your local chapter files (recommended, always works):**
```bash
# Analyze the first 10 chapters (default)
python novel_manager/init_novel.py novels/shadow_slave

# Analyze a specific chapter range
python novel_manager/init_novel.py novels/shadow_slave --chapters 1-20

# Analyze chapters 5 onwards
python novel_manager/init_novel.py novels/shadow_slave --chapters 5-
```

**Web mode — Gemini searches the internet (good for popular novels with wikis/fandom pages):**
```bash
# Let Gemini figure out the novel name from your folder
python novel_manager/init_novel.py novels/shadow_slave --mode web

# Provide the exact novel name for better search results
python novel_manager/init_novel.py novels/shadow_slave --mode web --novel-name "Shadow Slave by Guiltythree"

# Web mode with a specific chapter range context
python novel_manager/init_novel.py novels/shadow_slave --mode web --chapters 1-100
```

> **Web mode note:** If web coverage is poor (>40% unknown fields), it automatically falls back to text mode using your local chapters.

---

### STEP 4 — Cast Voices

Gemini reads `novel.md` + `voices.md` and assigns the best-matching XTTS voice to each character. Results are written to `novels/shadow_slave/data/speakers.json`.

**Cast all unassigned characters:**
```bash
python novel_manager/cast_voices.py novels/shadow_slave
```

**Preview the casting without writing to disk:**
```bash
python novel_manager/cast_voices.py novels/shadow_slave --dry-run
```

**Re-cast all characters from scratch:**
```bash
python novel_manager/cast_voices.py novels/shadow_slave --recast-all
```

**Cast a single specific character:**
```bash
python novel_manager/cast_voices.py novels/shadow_slave --character idan
```

---

### STEP 5 — Batch Process (Generate Audio)

#### Process ALL chapters

```bash
python batch.py novels/shadow_slave
```

#### Process a specific range of chapters

```bash
# Chapters 1 through 50
python batch.py novels/shadow_slave 1-50

# Chapters 51 onwards (useful for resuming after downloading more)
python batch.py novels/shadow_slave 51-

# A single chapter via batch
python batch.py novels/shadow_slave 7-7
```

> **Checkpoint recovery:** If the batch crashes, just re-run the same command — it resumes from the last successful chapter.
>
> **MP3 output:** Every 10 consecutive chapters are automatically combined into `chapters_1_to_10.mp3`, `chapters_11_to_20.mp3`, etc.
>
> **Performance:** ~212 seconds per chapter on an RTX 4050. The process priority is automatically boosted to prevent GPU throttling during overnight runs.

---

### STEP 6 — Process a Single Chapter Individually

For testing, debugging, or re-rendering one chapter:

```bash
python main.py novels/shadow_slave/input/chapter_1.txt
```

---

## 🔄 Ongoing Maintenance

### Update the novel after 50–100 new chapters

Arc update — Gemini reads the new chapters and appends to `novel.md` (never overwrites existing data):

**Auto-detect new chapters since last update:**
```bash
python novel_manager/update_novel.py novels/shadow_slave
```

**Specify the new chapter range explicitly:**
```bash
python novel_manager/update_novel.py novels/shadow_slave --chapters 101-200
```

**Update from chapter 101 onwards (open-ended):**
```bash
python novel_manager/update_novel.py novels/shadow_slave --chapters 101-
```

After updating, re-run voice casting to assign voices to any new characters:
```bash
python novel_manager/cast_voices.py novels/shadow_slave
```

---

### Add a new character manually

When a new named speaker appears mid-batch and you want them properly characterized before their voice is finalized:

**Add using chapters where the character first appears:**
```bash
python novel_manager/add_character.py novels/shadow_slave --character nephis
```

**Specify which chapters to analyze for that character:**
```bash
python novel_manager/add_character.py novels/shadow_slave --character nephis --chapters 25-40
```

**Then cast a voice for the new character:**
```bash
python novel_manager/cast_voices.py novels/shadow_slave --character nephis
```

---

## 🎤 Custom Voice Training

Add a completely new voice to XTTS v2 from your own audio samples.

**Requirements:**
- Audio clips: 6–30 seconds of clear, single-speaker speech (WAV or MP3)
- No background music or noise
- Longer/more samples = better quality

**Train from a single audio file:**
```bash
python utilities/train_voice.py "My Custom Voice" path/to/sample.wav --gender male
```

**Train from multiple audio files:**
```bash
python utilities/train_voice.py "My Custom Voice" sample1.wav sample2.wav sample3.wav --gender female
```

**Train from all `.wav`/`.mp3` files in a folder:**
```bash
python utilities/train_voice.py "My Custom Voice" samples/ --gender male
```

**List all registered custom voices:**
```bash
python utilities/train_voice.py --list
```

**Remove a custom voice:**
```bash
python utilities/train_voice.py --remove "My Custom Voice"
```

The trained voice is:
- Injected directly into `speakers_xtts.pth`
- Appended to `voices.md`
- A test audio clip is generated in `voice_tests/`

---

## 🧪 Tests & Diagnostics

### Test XTTS v2 is working end-to-end
```bash
python utilities/test_tts.py
```
Loads the model, retrieves speaker latents for "Ana Florence", runs a minimal inference, and prints the WAV length. Use this to confirm GPU + model are functional before a long batch run.

### Test Groq API connection
```bash
python utilities/network_connection_test.py
```
Sends a "Ping" to the Groq API using `GROQ_API_KEY_1` and prints the response. Confirms your API key is valid and the server is reachable.

### List all XTTS v2 speaker names
```bash
python utilities/Speakers.py
```
Loads the XTTS model and prints every available built-in speaker name. Useful when manually editing `speakers.json`.

### List available Gemini models
```bash
python utilities/gemini.py
```
Queries the Gemini API and prints all models that support content generation.

### Diagnose parser word coverage
```bash
python utilities/diagnose_coverage.py novels/shadow_slave/input/chapter_4.txt
```
Re-parses the chapter and compares the parser's output against the source text sentence-by-sentence. Reports what percentage of words were captured. Use this if you suspect the parser is skipping content.

### Inspect raw Groq parser output
```bash
# Standard inspection report
python utilities/inspect_parser.py novels/shadow_slave/input/chapter_2.txt

# Print only the raw JSON response (no analysis)
python utilities/inspect_parser.py novels/shadow_slave/input/chapter_2.txt --raw-only

# Run inspection and save the full report to a JSON file
python utilities/inspect_parser.py novels/shadow_slave/input/chapter_2.txt --save
```
Shows the complete Groq response, token distribution, segment breakdown, and per-speaker statistics. Essential for debugging JSON parse failures.

---

## 📋 Quick Reference — All Commands

```bash
# ── Environment setup ───────────────────────────────────────────
set CUDA_VISIBLE_DEVICES=0
venv\Scripts\activate

# ── New novel setup ─────────────────────────────────────────────
python utilities/setup_novel.py novel_name

# ── Scraping ────────────────────────────────────────────────────
python utilities/scrape_chapter.py novel_slug <START_URL>          # 1 chapter
python utilities/scrape_chapter.py novel_slug <START_URL> -n 12    # 12 chapters

# ── Novel initialization ─────────────────────────────────────────
python novel_manager/init_novel.py novels/novel_name               # text mode, 10 chaps
python novel_manager/init_novel.py novels/novel_name --chapters 1-20
python novel_manager/init_novel.py novels/novel_name --chapters 5-
python novel_manager/init_novel.py novels/novel_name --mode web
python novel_manager/init_novel.py novels/novel_name --mode web --novel-name "Full Title"

# ── Voice casting ────────────────────────────────────────────────
python novel_manager/cast_voices.py novels/novel_name
python novel_manager/cast_voices.py novels/novel_name --dry-run
python novel_manager/cast_voices.py novels/novel_name --recast-all
python novel_manager/cast_voices.py novels/novel_name --character name

# ── Batch processing ─────────────────────────────────────────────
python batch.py novels/novel_name                   # all chapters
python batch.py novels/novel_name 1-50              # chapters 1–50
python batch.py novels/novel_name 51-               # chapter 51 onwards
python batch.py novels/novel_name 7-7               # single chapter via batch

# ── Single chapter ───────────────────────────────────────────────
python main.py novels/novel_name/input/chapter_1.txt

# ── Ongoing maintenance (every 50–100 chapters) ──────────────────
python novel_manager/update_novel.py novels/novel_name
python novel_manager/update_novel.py novels/novel_name --chapters 101-200
python novel_manager/update_novel.py novels/novel_name --chapters 101-

# ── New character appeared ───────────────────────────────────────
python novel_manager/add_character.py novels/novel_name --character name
python novel_manager/add_character.py novels/novel_name --character name --chapters 25-40
python novel_manager/cast_voices.py novels/novel_name --character name

# ── Custom voice training ────────────────────────────────────────
python utilities/train_voice.py "Voice Name" sample.wav --gender male
python utilities/train_voice.py "Voice Name" sample1.wav sample2.wav --gender female
python utilities/train_voice.py "Voice Name" samples/ --gender male
python utilities/train_voice.py --list
python utilities/train_voice.py --remove "Voice Name"

# ── Tests & diagnostics ──────────────────────────────────────────
python utilities/test_tts.py
python utilities/network_connection_test.py
python utilities/Speakers.py
python utilities/gemini.py
python utilities/diagnose_coverage.py novels/novel_name/input/chapter_N.txt
python utilities/inspect_parser.py novels/novel_name/input/chapter_N.txt
python utilities/inspect_parser.py novels/novel_name/input/chapter_N.txt --raw-only
python utilities/inspect_parser.py novels/novel_name/input/chapter_N.txt --save
```

---

## 🎭 Voice Pool

### Fixed (never reassigned)
| Voice | Role |
|---|---|
| Ana Florence | Narrator — always |
| Nova Hogarth | System messages — always |

### Castable Female Voices (19)
Gracie Wise, Sofia Hellen, Tanja Adelina, Barbora MacLean, Szofi Granger, Claribel Dervla, Daisy Studious, Tammie Ema, Alison Dietlinde, Annmarie Nele, Brenda Stern, Gitta Nikolina, Henriette Usha, Maja Ruusunen, Rosemary Okafor, Tammy Grit, Tina Tina, Uta Obando, Alma Maria

### Castable Male Voices (10)
Damien Black, Craig Gutsy, Torcull Diarmuid, Ludvig Milivoj, Baldur Sanjin, Zacharie Aimilios, Andrew Chipper, Dionisio Schuyler, Abrahan Mack, Viktor Menelaos

---

## 🔧 Fallback Engine

If a segment fails to render, the pipeline automatically retries through 5 escalating stages:

1. Normal render × 3 retries
2. Cleaned text (strip non-ASCII) × 3 retries
3. Fallback to Ana Florence voice × 3 retries
4. Split into 150-char parts, render individually
5. Audible placeholder tone (440Hz) + log to `data/{chapter}_failures.json`

No crash — the batch always continues.

---

## 📊 Supported Emotional Tones

`neutral` · `calm` · `tense` · `whisper` · `angry` · `sad` · `excited` · `cold` · `fearful` · `sarcastic` · `pleading` · `commanding` · `gentle` · `mocking` · `sorrowful` · `triumphant`

> Speed is globally locked to **1.1** across all tone profiles. Emotional variation is achieved purely through temperature and sampling parameters — this prevents voice identity bleed between tones.

---

## 📝 Notes

- **`novel.md`** is a living document. Never delete it. The update workflow always appends — it never overwrites existing character data.
- **`speakers.json`** tracks which voice each character is assigned to. Characters cast by Gemini are marked `"cast_by": "gemini"` and will not be reassigned on subsequent runs.
- **Checkpoint recovery** means you can kill and restart `batch.py` at any time without re-rendering completed chapters.
- All scripts are run from the **project root directory**.