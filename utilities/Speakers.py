"""
Speakers.py

Quick utility script — loads the XTTS v2 model and prints all available
speaker names from its built-in speaker manager. Useful for discovering
voice options. No functions; runs directly when executed.
"""

import os, torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
MODEL_DIR = r'C:\Users\ABRAR\AppData\Local\tts\tts_models--multilingual--multi-dataset--xtts_v2'
config = XttsConfig()
config.load_json(os.path.join(MODEL_DIR, 'config.json'))
model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=MODEL_DIR, eval=True)
print(list(model.speaker_manager.speakers.keys()))