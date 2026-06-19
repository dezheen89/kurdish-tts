"""
RunPod Serverless handler for F5-TTS (Central Kurdish) — multi-voice.

Three fine-tuned voices are baked into the image:
    - "audiobook-female"
    - "audiobook-male"
    - "studio-male"

Each voice has its own checkpoint (.pt) + its own reference audio/text
(taken straight from aranemini/central-kurdish-tts), but they all share
the same vocab.txt and F5TTS_v1_Base architecture, so only one vocoder
needs to be loaded — we swap the EMA model weights per request/voice.

Input JSON shape (sent as {"input": {...}} to the RunPod endpoint):
{
    "text": "Kurdish text to synthesize",          # required
    "voice": "audiobook-female",                    # optional, default "audiobook-female"
                                                      # one of: audiobook-female | audiobook-male | studio-male
    "ref_audio": "base64-or-omit",                  # optional, base64 wav — overrides the built-in voice prompt
    "ref_text": "transcript of ref_audio",           # required if ref_audio is provided
    "speed": 1.0,                                    # optional
    "remove_silence": false                          # optional
}

Output JSON shape:
{
    "audio_base64": "...",   # base64-encoded WAV bytes
    "sample_rate": 24000,
    "voice": "audiobook-female"
}
"""

import base64
import os
import tempfile

import runpod
import torch

from f5_tts.api import F5TTS

CKPT_DIR = os.environ.get("F5TTS_CKPT_DIR", "/app/ckpts")
VOCAB_FILE = os.path.join(CKPT_DIR, "vocab.txt")

VOICES = {
    "audiobook-female": {
        "ckpt": os.path.join(CKPT_DIR, "model-audiobook-female.pt"),
        "ref_audio": os.path.join(CKPT_DIR, "prompt-audiobook-female.wav"),
        "ref_text_file": os.path.join(CKPT_DIR, "prompt-audiobook-female.txt"),
    },
    "audiobook-male": {
        "ckpt": os.path.join(CKPT_DIR, "model-audiobook-male.pt"),
        "ref_audio": os.path.join(CKPT_DIR, "prompt-audiobook-male.wav"),
        "ref_text_file": os.path.join(CKPT_DIR, "prompt-audiobook-male.txt"),
    },
    "studio-male": {
        "ckpt": os.path.join(CKPT_DIR, "model-studio-male.pt"),
        "ref_audio": os.path.join(CKPT_DIR, "prompt-studio-male.wav"),
        "ref_text_file": os.path.join(CKPT_DIR, "prompt-studio-male.txt"),
    },
}
DEFAULT_VOICE = "audiobook-female"

# Read each voice's reference transcript once at startup
for _voice, _cfg in VOICES.items():
    if os.path.exists(_cfg["ref_text_file"]):
        with open(_cfg["ref_text_file"], "r", encoding="utf-8") as f:
            _cfg["ref_text"] = f.read().strip()
    else:
        _cfg["ref_text"] = None

print("Loading F5-TTS (vocoder + base model)...")
# Load once with whichever voice is default; we'll hot-swap state_dicts for other voices.
f5tts = F5TTS(
    model="F5TTS_v1_Base",
    ckpt_file=VOICES[DEFAULT_VOICE]["ckpt"],
    vocab_file=VOCAB_FILE,
)
print(f"Loaded default voice: {DEFAULT_VOICE}")

# Cache of loaded state_dicts per voice so repeated calls don't re-read from disk
_state_dict_cache = {}


def _load_voice_weights(voice_name: str):
    """Swap the EMA model's weights to the requested voice's checkpoint."""
    if voice_name == DEFAULT_VOICE and voice_name not in _state_dict_cache:
        # already loaded as part of F5TTS() init — nothing to do
        return

    if voice_name not in _state_dict_cache:
        ckpt_path = VOICES[voice_name]["ckpt"]
        print(f"Loading checkpoint for voice '{voice_name}' from {ckpt_path} ...")
        raw = torch.load(ckpt_path, map_location=f5tts.device)
        # F5-TTS finetuned checkpoints commonly store either the raw state_dict
        # or a dict with an "ema_model_state_dict" / "model_state_dict" key.
        if isinstance(raw, dict) and "ema_model_state_dict" in raw:
            state_dict = raw["ema_model_state_dict"]
        elif isinstance(raw, dict) and "model_state_dict" in raw:
            state_dict = raw["model_state_dict"]
        else:
            state_dict = raw
        _state_dict_cache[voice_name] = state_dict

    f5tts.ema_model.load_state_dict(_state_dict_cache[voice_name], strict=False)
    f5tts.ema_model.eval()


def _wav_bytes_to_base64(wav_path: str) -> str:
    with open(wav_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


def handler(job):
    job_input = job.get("input", {})

    gen_text = job_input.get("text")
    if not gen_text:
        return {"error": "Missing required field: 'text'"}

    voice = job_input.get("voice", DEFAULT_VOICE)
    if voice not in VOICES:
        return {"error": f"Unknown voice '{voice}'. Valid options: {list(VOICES.keys())}"}

    speed = float(job_input.get("speed", 1.0))
    remove_silence = bool(job_input.get("remove_silence", False))

    ref_audio_b64 = job_input.get("ref_audio")
    ref_text = job_input.get("ref_text")

    voice_cfg = VOICES[voice]
    tmp_ref_path = voice_cfg["ref_audio"]
    tmp_file_to_clean = None

    try:
        # Swap in the requested voice's model weights
        _load_voice_weights(voice)

        if ref_audio_b64:
            # caller supplied their own reference clip — overrides the built-in voice prompt
            if not ref_text:
                return {"error": "ref_text is required when ref_audio is provided"}
            audio_bytes = base64.b64decode(ref_audio_b64)
            tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
            tmp.write(audio_bytes)
            tmp.close()
            tmp_ref_path = tmp.name
            tmp_file_to_clean = tmp.name
        else:
            ref_text = ref_text or voice_cfg["ref_text"]
            if not os.path.exists(tmp_ref_path):
                return {"error": f"Built-in reference audio missing for voice '{voice}': {tmp_ref_path}"}
            if not ref_text:
                return {"error": f"No ref_text available for voice '{voice}'"}

        out_tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        out_path = out_tmp.name
        out_tmp.close()

        wav, sr, _ = f5tts.infer(
            ref_file=tmp_ref_path,
            ref_text=ref_text,
            gen_text=gen_text,
            speed=speed,
            remove_silence=remove_silence,
            file_wave=out_path,
            seed=job_input.get("seed"),
        )

        audio_b64 = _wav_bytes_to_base64(out_path)
        os.remove(out_path)

        return {
            "audio_base64": audio_b64,
            "sample_rate": sr,
            "format": "wav",
            "voice": voice,
        }

    except Exception as e:
        return {"error": str(e)}

    finally:
        if tmp_file_to_clean and os.path.exists(tmp_file_to_clean):
            os.remove(tmp_file_to_clean)


runpod.serverless.start({"handler": handler})
