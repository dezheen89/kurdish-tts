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

----------------------------------------------------------------------------
WHAT WAS WRONG (and is now fixed)
----------------------------------------------------------------------------
The previous version swapped voice weights with:

    f5tts.ema_model.load_state_dict(state_dict, strict=False)

`strict=False` SILENTLY tolerates key-name mismatches. If a checkpoint's
keys don't line up exactly with the live model's keys (common with EMA
checkpoints that carry an "ema_model." prefix, or bookkeeping keys like
"initted"/"step"/"ema."), the call loads only the keys that happen to
match — often ZERO — and leaves the rest of the model holding the PREVIOUS
voice's weights. Result: audio that is generated from your text but sounds
garbled / mismatched, because it's a Frankenstein of two checkpoints.

The fix:
  1. Normalize checkpoint keys (strip common EMA wrapper prefixes, drop
     non-parameter bookkeeping keys) so they line up with the live model.
  2. Load and then VERIFY: if too many keys are still missing, RAISE instead
     of returning silently-wrong audio.
----------------------------------------------------------------------------
"""

import base64
import os
import tempfile
import threading

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

# If fewer than this fraction of the model's parameters get loaded from a
# checkpoint, we treat the swap as failed (wrong/garbled voice) and raise.
MIN_LOAD_FRACTION = 0.98

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

# Cache of normalized state_dicts per voice so repeated calls don't re-read from disk
_state_dict_cache = {}

# Track which voice's weights are CURRENTLY live in f5tts.ema_model.
_active_voice = DEFAULT_VOICE

# Serialize the swap+infer critical section. Even with max_workers=1 a single
# worker can be asked to handle requests back-to-back; this guarantees no
# request ever runs inference while another is mid-swap.
_infer_lock = threading.RLock()


def _normalize_state_dict(state_dict: dict, model_keys: set) -> dict:
    """Make a checkpoint's keys line up with the live model's keys.

    Handles the common cases:
      - keys wrapped with an 'ema_model.' / 'ema.' / 'module.' prefix
      - bookkeeping entries that aren't real parameters (initted, step, etc.)

    Returns a new dict containing only keys the model actually expects.
    """
    # Drop obvious non-parameter bookkeeping keys.
    junk = {"initted", "step", "ema.initted", "ema.step"}
    cleaned = {k: v for k, v in state_dict.items() if k not in junk}

    # If keys already match, nothing to do.
    if any(k in model_keys for k in cleaned):
        already = sum(1 for k in cleaned if k in model_keys)
        # If a healthy majority already match, keep as-is.
        if already >= len(model_keys) * MIN_LOAD_FRACTION:
            return cleaned

    # Try stripping known wrapper prefixes until keys line up.
    for prefix in ("ema_model.", "ema.", "module.", "model."):
        stripped = {
            (k[len(prefix):] if k.startswith(prefix) else k): v
            for k, v in cleaned.items()
        }
        match_count = sum(1 for k in stripped if k in model_keys)
        if match_count >= len(model_keys) * MIN_LOAD_FRACTION:
            return stripped

    # Nothing lined up cleanly — return cleaned and let the verifier raise.
    return cleaned


def _read_state_dict(voice_name: str) -> dict:
    """Read, normalize, and cache the EMA/model state_dict for a voice."""
    if voice_name not in _state_dict_cache:
        ckpt_path = VOICES[voice_name]["ckpt"]
        print(f"Loading checkpoint for voice '{voice_name}' from {ckpt_path} ...")
        # Load to CPU first; we move tensors to device at load_state_dict time.
        raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if isinstance(raw, dict) and "ema_model_state_dict" in raw:
            state_dict = raw["ema_model_state_dict"]
        elif isinstance(raw, dict) and "model_state_dict" in raw:
            state_dict = raw["model_state_dict"]
        else:
            state_dict = raw

        model_keys = set(f5tts.ema_model.state_dict().keys())
        state_dict = _normalize_state_dict(state_dict, model_keys)
        _state_dict_cache[voice_name] = state_dict
    return _state_dict_cache[voice_name]


def _load_voice_weights(voice_name: str) -> dict:
    """Ensure the EMA model's weights match the requested voice.

    Loads STRICTLY-VERIFIED: if the checkpoint fails to populate essentially
    all of the model's parameters, we RAISE rather than emit garbled audio.
    Returns a dict with (benign) mismatch info; empty if perfectly clean.
    """
    global _active_voice

    if voice_name == _active_voice:
        return {}

    state_dict = _read_state_dict(voice_name)
    model_keys = set(f5tts.ema_model.state_dict().keys())

    incompatible = f5tts.ema_model.load_state_dict(state_dict, strict=False)
    missing = list(getattr(incompatible, "missing_keys", []))
    unexpected = list(getattr(incompatible, "unexpected_keys", []))

    loaded = len(model_keys) - len(missing)
    load_fraction = loaded / max(1, len(model_keys))

    if load_fraction < MIN_LOAD_FRACTION:
        # The swap did NOT really happen. Refuse to produce wrong-voice audio.
        raise RuntimeError(
            f"Voice swap to '{voice_name}' FAILED: only {loaded}/{len(model_keys)} "
            f"params loaded ({load_fraction:.1%}). This means the checkpoint keys "
            f"do not match the model. Sample missing: {missing[:5]} | "
            f"Sample unexpected (in ckpt, not model): {unexpected[:5]}. "
            f"Fix _normalize_state_dict prefix handling for this checkpoint."
        )

    f5tts.ema_model.eval()
    _active_voice = voice_name

    if missing or unexpected:
        # A handful of non-critical keys differing is tolerable past the threshold.
        print(
            f"NOTE: voice '{voice_name}' loaded with minor key diff — "
            f"{len(missing)} missing, {len(unexpected)} unexpected "
            f"(loaded {load_fraction:.1%}). Proceeding."
        )
        return {"missing_count": len(missing), "unexpected_count": len(unexpected),
                "load_fraction": round(load_fraction, 4)}
    return {}


def _inspect_checkpoint(voice_name: str) -> dict:
    """Debug helper: report the raw structure of a voice's .pt file without loading it."""
    ckpt_path = VOICES[voice_name]["ckpt"]
    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    info = {"voice": voice_name, "ckpt_path": ckpt_path}
    if isinstance(raw, dict):
        info["top_level_keys"] = list(raw.keys())
        for key in ("ema_model_state_dict", "model_state_dict"):
            if key in raw and isinstance(raw[key], dict):
                sub_keys = list(raw[key].keys())
                info[f"{key}_sample"] = sub_keys[:10]
                info[f"{key}_count"] = len(sub_keys)
        if "ema_model_state_dict" not in raw and "model_state_dict" not in raw:
            sample = list(raw.keys())[:10]
            info["raw_dict_sample_keys"] = sample
    else:
        info["type"] = str(type(raw))
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

    if job_input.get("debug") == "show_ref_text":
        return {
            "debug_info": {
                v: {
                    "ref_text": cfg["ref_text"],
                    "ref_text_file": cfg["ref_text_file"],
                    "ref_audio": cfg["ref_audio"],
                    "ref_audio_exists": os.path.exists(cfg["ref_audio"]),
                }
                for v, cfg in VOICES.items()
            }
        }

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

    # Hold the lock across the WHOLE swap+infer so a concurrent request can't
    # change the live weights out from under us mid-generation.
    with _infer_lock:
        try:
            mismatch_info = _load_voice_weights(voice)

            if ref_audio_b64:
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
                "active_voice": _active_voice,  # sanity check: should equal "voice"
            }
            if mismatch_info:
                result["warning"] = (
                    f"checkpoint key diff for voice '{voice}': "
                    f"{mismatch_info['missing_count']} missing, "
                    f"{mismatch_info['unexpected_count']} unexpected "
                    f"(loaded {mismatch_info.get('load_fraction')})"
                )
            return result

        except Exception as e:
            return {"error": str(e)}

        finally:
            if tmp_file_to_clean and os.path.exists(tmp_file_to_clean):
                os.remove(tmp_file_to_clean)


runpod.serverless.start({"handler": handler})
