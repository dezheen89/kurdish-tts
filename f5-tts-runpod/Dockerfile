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

# --- Kurdish text preprocessing (G2P) ---
# The Kurdish checkpoints were trained on PHONEMIZED text, not raw Kurdish
# script. asosoft.KurdishG2P() converts raw Kurdish into that phonemic form
# before it ever reaches the model — this is the step that was missing and
# caused generated audio to not match the input text. See handler.py's
# normalize_and_g2p(), which mirrors the model author's own infer.py.
RUN pip install --no-cache-dir asosoft

# --- Kurdish TTS checkpoints (3 voices, ~5.39 GB each = ~16 GB total) ---
# Repo: https://huggingface.co/aranemini/central-kurdish-tts
ARG HF_REPO_ID=aranemini/central-kurdish-tts
ARG HF_TOKEN=""
ENV HF_REPO_ID=${HF_REPO_ID}
ENV HF_TOKEN=${HF_TOKEN}

COPY download_checkpoints.py /app/download_checkpoints.py
RUN python /app/download_checkpoints.py

ENV F5TTS_CKPT_DIR=/app/ckpts

COPY handler.py /app/handler.py

CMD ["python", "-u", "/app/handler.py"]
