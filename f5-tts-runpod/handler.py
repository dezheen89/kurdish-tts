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

    # Debug mode (no inference run):
    "debug": "inspect_checkpoint",                   # requires "voice" too
    "debug": "inspect_all",                          # inspects all 3 voices
}

Output JSON shape (normal generation):
{
    "audio_base64": "...",   # base64-encoded WAV bytes
    "sample_rate": 24000,
    "format": "wav",
    "voice": "audiobook-female",
    "warning": "..."         # only present if checkpoint keys didn't match cleanly
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


def _load_voice_weights(voice_name: str) -> dict:
    """Swap the EMA model's weights to the requested voice's checkpoint.
    Returns a dict with mismatch info (empty if clean load)."""
    if voice_name == DEFAULT_VOICE and voice_name not in _state_dict_cache:
        # already loaded as part of F5TTS() init — nothing to do
        return {}

    if voice_name not in _state_dict_cache:
        ckpt_path = VOICES[voice_name]["ckpt"]
        print(f"Loading checkpoint for voice '{voice_name}' from {ckpt_path} ...")
        raw = torch.load(ckpt_path, map_location=f5tts.device)
        if isinstance(raw, dict) and "ema_model_state_dict" in raw:
            state_dict = raw["ema_model_state_dict"]
        elif isinstance(raw, dict) and "model_state_dict" in raw:
            state_dict = raw["model_state_dict"]
        else:
            state_dict = raw
        _state_dict_cache[voice_name] = state_dict

    incompatible = f5tts.ema_model.load_state_dict(_state_dict_cache[voice_name], strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))
    f5tts.ema_model.eval()

    if missing or unexpected:
        print(
            f"WARNING: voice '{voice_name}' checkpoint key mismatch — "
            f"{len(missing)} missing, {len(unexpected)} unexpected. "
            f"Sample missing: {missing[:5]} Sample unexpected: {unexpected[:5]}"
        )
        return {"missing_count": len(missing), "unexpected_count": len(unexpected)}
    return {}


def _inspect_checkpoint(voice_name: str) -> dict:
    """Debug helper: report the raw structure of a voice's .pt file without loading it."""
    ckpt_path = VOICES[voice_name]["ckpt"]
    raw = torch.load(ckpt_path, map_location="cpu")
    info = {"voice": voice_name, "ckpt_path": ckpt_path}
    if isinstance(raw, dict):
        info["top_level_keys"] = list(raw.keys())
        for key in ("ema_model_state_dict", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                sub_keys = list(raw[key].keys())
                info[f"{key}_sample"] = sub_keys[:10]
                info[f"{key}_count"] = len(sub_keys)
        if "ema_model_state_dict" not in raw and "model_state_dict" not in raw:
            # raw dict might itself be the state_dict
            sample = list(raw.keys())[:10]
            info["raw_dict_sample_keys"] = sample
    else:
        info["type"] = str(type(raw))
    # Compare against what the live model actually expects
    expected_keys = list(f5tts.ema_model.state_dict().keys())
    info["model_expected_sample"] = expected_keys[:10]
    info["model_expected_count"] = len(expected_keys)
    return info


def _wav_bytes_to_base64(wav_path: str) -> str:
    with open(wav_path, "rb") as f:
        data = f.read()
    return base64.b64encode(data).decode("utf-8")


def handler(job):
    job_input = job.get("input", {})

    # Debug mode: {"input": {"debug": "inspect_checkpoint", "voice": "audiobook-male"}}
    # Returns checkpoint structure info without running inference.
    if job_input.get("debug") == "inspect_checkpoint":
        voice = job_input.get("voice", DEFAULT_VOICE)
        if voice not in VOICES:
            return {"error": f"Unknown voice '{voice}'. Valid options: {list(VOICES.keys())}"}
        try:
            return {"debug_info": _inspect_checkpoint(voice)}
        except Exception as e:
            return {"error": f"inspect_checkpoint failed: {e}"}

    if job_input.get("debug") == "inspect_all":
        results = {}
        for v in VOICES:
            try:
                results[v] = _inspect_checkpoint(v)
            except Exception as e:
                results[v] = {"error": str(e)}
        return {"debug_info": results}

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
        mismatch_info = _load_voice_weights(voice)

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

        result = {
            "audio_base64": audio_b64,
            "sample_rate": sr,
            "format": "wav",
            "voice": voice,
        }
        if mismatch_info:
            result["warning"] = (
                f"checkpoint key mismatch for voice '{voice}': "
                f"{mismatch_info['missing_count']} missing, "
                f"{mismatch_info['unexpected_count']} unexpected — "
                f"audio may not reflect the requested voice correctly"
            )
        return result

    except Exception as e:
        return {"error": str(e)}

    finally:
        if tmp_file_to_clean and os.path.exists(tmp_file_to_clean):
            os.remove(tmp_file_to_clean)


runpod.serverless.start({"handler": handler})
