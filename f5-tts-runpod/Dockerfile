# F5-TTS Central Kurdish — RunPod Serverless Worker
# Bakes in all 3 fine-tuned voices from aranemini/central-kurdish-tts:
#   audiobook-female, audiobook-male, studio-male
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# --- System deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 python3-pip python3-dev git ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

WORKDIR /app

# --- Upgrade packaging tools (fixes PEP 660 / editable install support) ---
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# --- PyTorch (CUDA 12.1 build) ---
RUN pip install --no-cache-dir torch==2.4.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu121

# --- F5-TTS source ---
COPY F5-TTS-main /app/F5-TTS-main
RUN pip install --no-cache-dir -e /app/F5-TTS-main

# --- RunPod SDK + extras ---
RUN pip install --no-cache-dir runpod soundfile huggingface_hub

# --- Kurdish TTS checkpoints (3 voices, ~5.39 GB each = ~16 GB total) ---
# Repo: https://huggingface.co/aranemini/central-kurdish-tts
ARG HF_REPO_ID=aranemini/central-kurdish-tts
ARG HF_TOKEN=""

RUN mkdir -p /app/ckpts
RUN python -c "\
from huggingface_hub import hf_hub_download; \
import shutil, os; \
token = os.environ.get('HF_TOKEN') or None; \
repo = '${HF_REPO_ID}'; \
files = [ \
    'vocab.txt', \
    'model-audiobook-female.pt', \
    'model-audiobook-male.pt', \
    'model-studio-male.pt', \
    'prompt-audiobook-female.wav', \
    'prompt-audiobook-female.txt', \
    'prompt-audiobook-male.wav', \
    'prompt-audiobook-male.txt', \
    'prompt-studio-male.wav', \
    'prompt-studio-male.txt', \
]; \
for fname in files: \
    print(f'Downloading {fname} ...'); \
    path = hf_hub_download(repo_id=repo, filename=fname, token=token); \
    shutil.copy(path, f'/app/ckpts/{fname}'); \
print('All files downloaded.')"

ENV F5TTS_CKPT_DIR=/app/ckpts

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
