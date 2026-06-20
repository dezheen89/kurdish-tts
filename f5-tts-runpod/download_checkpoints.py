"""Download the Kurdish F5-TTS checkpoints + reference audio from Hugging Face."""
import os
import shutil

from huggingface_hub import hf_hub_download

token = os.environ.get("HF_TOKEN") or None
repo = os.environ.get("HF_REPO_ID", "aranemini/central-kurdish-tts")

files = [
    "vocab.txt",
    "model-audiobook-female.pt",
    "model-audiobook-male.pt",
    "model-studio-male.pt",
    "prompt-audiobook-female.wav",
    "prompt-audiobook-female.txt",
    "prompt-audiobook-male.wav",
    "prompt-audiobook-male.txt",
    "prompt-studio-male.wav",
    "prompt-studio-male.txt",
]

os.makedirs("/app/ckpts", exist_ok=True)

for fname in files:
    print(f"Downloading {fname} ...")
    path = hf_hub_download(repo_id=repo, filename=fname, token=token)
    shutil.copy(path, f"/app/ckpts/{fname}")

print("All files downloaded.")
