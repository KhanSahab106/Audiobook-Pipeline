"""
test_tts.py

TTS sanity-check script — loads the XTTS v2 model, retrieves speaker
latents for "Ana Florence", and runs a minimal inference to verify the
model works end-to-end. No functions; runs directly when executed.
"""

import os
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts

MODEL_DIR = r"C:\Users\ABRAR\AppData\Local\tts\tts_models--multilingual--multi-dataset--xtts_v2"

config = XttsConfig()
config.load_json(os.path.join(MODEL_DIR, "config.json"))
print("Config loaded")

model = Xtts.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir=MODEL_DIR, eval=True)
model.cuda()
print("Model loaded")

print("Available speakers:", list(model.speaker_manager.speakers.keys())[:5])

gpt_cond_latent = model.speaker_manager.speakers["Ana Florence"]["gpt_cond_latent"]
speaker_embedding = model.speaker_manager.speakers["Ana Florence"]["speaker_embedding"]
print("Speaker latents retrieved")

out = model.inference(
    text="Hello.",
    language="en",
    gpt_cond_latent=gpt_cond_latent,
    speaker_embedding=speaker_embedding,
    temperature=0.7
)
print("Inference ok, wav length:", len(out["wav"]))