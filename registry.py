"""
registry.py

Voice registry stub — simplified for single-narrator pipeline.
Always returns the narrator voice ("Ana Florence").
"""

def load_registry(novel_dir: str) -> dict:
    return {"characters": {"narrator": {"xtts_speaker": "Ana Florence"}}}

def save_registry(registry: dict, novel_dir: str):
    pass

def resolve_speaker(speaker_name: str, registry: dict, novel_dir: str) -> str:
    return "Ana Florence"

def get_known_characters(registry: dict) -> list[str]:
    return ["narrator"]

def reassign_speaker(novel_dir: str, character_key: str, new_voice: str):
    pass