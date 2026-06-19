# Kurdish F5-TTS on RunPod Serverless

Text-to-speech for Central Kurdish (Sorani), using F5-TTS + the
`aranemini/central-kurdish-tts` fine-tuned checkpoints (3 voices), deployed
as a RunPod Serverless endpoint, with a simple web frontend to call it.

## Confirmed source files (from huggingface.co/aranemini/central-kurdish-tts)

| File | Size | Used for |
|---|---|---|
| `vocab.txt` | 11.3 kB | shared tokenizer for all voices |
| `F5TTS_v1_Base.yaml` | 2.02 kB | confirms model arch = F5TTS_v1_Base |
| `model-audiobook-female.pt` | 5.39 GB | voice: audiobook-female |
| `model-audiobook-male.pt` | 5.39 GB | voice: audiobook-male |
| `model-studio-male.pt` | 5.39 GB | voice: studio-male |
| `prompt-audiobook-female.wav` + `.txt` | 550 kB | reference clip + transcript |
| `prompt-audiobook-male.wav` + `.txt` | 465 kB | reference clip + transcript |
| `prompt-studio-male.wav` + `.txt` | 556 kB | reference clip + transcript |

All of this is downloaded automatically at **Docker build time** straight from
Hugging Face — you don't need to manually download or upload anything.

> Total checkpoint size: ~16 GB. Build will take a while and the RunPod
> container disk needs to be sized accordingly (30 GB+ recommended).

## File layout

```
f5-tts-runpod/
├── Dockerfile           # builds the RunPod worker image, downloads all 3 voices from HF
├── handler.py            # RunPod serverless handler — multi-voice, hot-swaps weights
├── F5-TTS-main/           # F5-TTS source (already included)
└── frontend/
    └── index.html         # standalone web UI with a voice picker, calls your endpoint
```

## Step 1 — Push to GitHub

```bash
cd f5-tts-runpod
git init
git add .
git commit -m "F5-TTS Kurdish multi-voice RunPod worker"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

This repo does **not** contain the 16 GB of checkpoints — those are pulled
from Hugging Face during the Docker build on RunPod's servers, so the
GitHub push itself stays small and fast.

## Step 2 — Create the RunPod endpoint from GitHub

1. Go to **console.runpod.io/serverless**
2. Click **+ New Endpoint**
3. Choose **GitHub Repo** as the source
4. Connect your GitHub account if not already linked, select your repo + branch
5. RunPod detects the `Dockerfile` automatically
6. If `aranemini/central-kurdish-tts` is gated/private on Hugging Face, add a
   **Build Argument**: `HF_TOKEN` = your Hugging Face access token. (Based on
   the model card you showed, this repo appears public, so this may not be needed.)
7. Set **Container Disk** to **30 GB or more** (16 GB of checkpoints + CUDA + libs)
8. Choose a GPU tier — 16 GB VRAM (e.g. RTX A4000) is enough to run inference;
   you only need more VRAM if you want multiple voices loaded simultaneously
   in memory (current handler swaps weights on the same GPU, so 16 GB is fine)
9. Click **Deploy**

First build will take longer than usual (~16 GB to download + CUDA image) —
expect 10-20+ minutes for the first build. Subsequent deploys from the same
repo reuse cached layers and are faster.

## Step 3 — Test the endpoint

```bash
curl -X POST https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync \
  -H "Authorization: Bearer <YOUR_API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"input": {"text": "سڵاو، باشیت چۆنیت؟", "voice": "audiobook-female"}}'
```

Expected response:
```json
{
  "output": {
    "audio_base64": "UklGRi...",
    "sample_rate": 24000,
    "format": "wav",
    "voice": "audiobook-female"
  }
}
```

Valid `voice` values: `audiobook-female`, `audiobook-male`, `studio-male`
(defaults to `audiobook-female` if omitted).

## Step 4 — Use the frontend

Open `frontend/index.html` in a browser (double-click it, or host it on
GitHub Pages / Netlify / any static host).

In **advanced settings**, paste:
- **RunPod endpoint URL**: `https://api.runpod.ai/v2/<ENDPOINT_ID>/runsync`
- **RunPod API key**: your key

Pick a voice from the dropdown, type Kurdish text, click **دەنگی پێبدە**.

## Notes & gotchas

- **Checkpoint format**: these are raw PyTorch `.pt` files, not `.safetensors`.
  `handler.py` loads them with `torch.load` and checks for common state_dict
  key wrappers (`ema_model_state_dict`, `model_state_dict`, or raw). If loading
  fails with a key-mismatch error, inspect the checkpoint structure:
  ```python
  import torch
  ckpt = torch.load("model-audiobook-female.pt", map_location="cpu")
  print(type(ckpt), ckpt.keys() if isinstance(ckpt, dict) else None)
  ```
  and adjust `_load_voice_weights()` in `handler.py` accordingly.
- **Cold starts**: with 3 voices baked in but only one loaded into the active
  GPU model at a time, switching voice on a cold worker means loading a fresh
  ~5.39 GB checkpoint from disk into GPU memory — expect a few extra seconds
  the first time each voice is used per worker. The handler caches loaded
  state_dicts in CPU memory after first use within the same worker, so
  subsequent switches back to that voice are faster.
- **`/runsync` vs `/run`**: `/runsync` blocks and returns the result directly
  (used above, simplest for a frontend). `/run` returns immediately with a job
  ID you poll — better for long texts that risk timing out `/runsync`.
- **Custom voice per request**: callers can bypass the 3 built-in voices by
  sending their own `ref_audio` (base64 wav) + `ref_text` — the handler
  supports this already; the frontend doesn't expose it yet but can be
  extended with a file upload input.
