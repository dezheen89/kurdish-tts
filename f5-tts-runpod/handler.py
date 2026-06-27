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
import re
import tempfile
import threading

import runpod
import torch

from f5_tts.api import F5TTS

try:
    import asosoft
except ImportError as e:
    raise ImportError(
        "The 'asosoft' package is required to preprocess Kurdish text before "
        "F5-TTS inference (the model was trained on phonemized text, not raw "
        "Kurdish script). Add `pip install asosoft` to the Dockerfile."
    ) from e

CKPT_DIR = os.environ.get("F5TTS_CKPT_DIR", "/app/ckpts")
VOCAB_FILE = os.path.join(CKPT_DIR, "vocab.txt")

# ----------------------------------------------------------------------------
# TEXT PREPROCESSING — mirrors the model author's own infer.py EXACTLY.
#
# The model was NOT trained on raw Kurdish script. It was trained on a
# phonemized form produced by the `asosoft` library's Kurdish G2P (grapheme-
# to-phoneme) converter. Skipping this step is why text didn't match audio:
# raw Kurdish letters (پ چ ێ ۆ ...) aren't even tokens the model has any
# learned meaning for — only the phonemized output is.
#
# This MUST be applied to both `gen_text` (what the user wants spoken) and
# `ref_text` (the transcript of the reference/prompt audio) before either
# is handed to f5tts.infer().
# ----------------------------------------------------------------------------

_G2P_REPLACEMENTS = {
    "ل؛چ": "د‡",
    "ئ¹": "آ؟",
    "ل¸§": "ل¸¥",
}


def normalize_and_g2p(text: str) -> str:
    """Convert raw Kurdish text into the phonemic form the model expects.
    Mirrors aranemini/central-kurdish-tts's infer.py normalize_and_g2p()."""
    # NOTE: must use a callback here, not a replacement string — re.sub's
    # template parser does its own backslash handling and does NOT support
    # \uXXXX escapes (that crashed with "bad escape \u"). A callback's
    # return value is inserted verbatim, so Python's own (correct) Unicode
    # escape handling applies instead.
    text = re.sub(
        r"(\d{1,8})\s*[-\u2013]\s*(\d{1,8})",
        lambda m: f"{m.group(1)} \u0637\u0647\u0637\u0627 {m.group(2)}",
        text,
    )

    text = re.sub(
        r"\b\d{9,}\b",
        lambda m: f"<NUM:{m.group(0)}>",
        text,
    )

    norm = asosoft.Normalize(
        text,
        changeInitialR=True,
        deepUnicodeCorrectios=True,
        additionalUnicodeCorrections=True,
    )

    norm = asosoft.NormalizePunctuations(
        norm,
        seprateAllPunctuations=True,
    )

    try:
        norm = asosoft.Number2Word(norm)
    except Exception as e:
        print(f"[Number2Word Error] Skipping conversion: {e}")

    g2p = asosoft.KurdishG2P(norm).replace("\u062b\u02c6", "")

    for old, new in _G2P_REPLACEMENTS.items():
        g2p = g2p.replace(old, new)

    return g2p

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

# Read each voice's reference transcript once at startup, and immediately
# convert it to the phonemic form the model expects (same as gen_text).
for _voice, _cfg in VOICES.items():
    if os.path.exists(_cfg["ref_text_file"]):
        with open(_cfg["ref_text_file"], "r", encoding="utf-8") as f:
            _raw_ref_text = f.read().strip()
        _cfg["ref_text_raw"] = _raw_ref_text
        _cfg["ref_text"] = normalize_and_g2p(_raw_ref_text) if _raw_ref_text else None
    else:
        _cfg["ref_text_raw"] = None
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


def _diagnose_vocab() -> dict:
    """Compare vocab.txt against the live model's text-embedding size, and
    report Kurdish character coverage. Pinpoints why text != audio."""
    kurdish_letters = [
        ("\u067e", "p"), ("\u0686", "ch"), ("\u0698", "zh"),
        ("\u06af", "g"), ("\u0695", "rr"), ("\u06a4", "v"),
        ("\u06c6", "o"), ("\u06ce", "e-circ"), ("\u06be", "h2"),
        ("\u06d5", "e"), ("\u0626", "hamza-yeh"), ("\u06cc", "y/i"),
        ("\u06a9", "k-farsi"), ("\u06b5", "ll"),
    ]

    # Read vocab.txt (one token per line; order defines index).
    with open(VOCAB_FILE, "r", encoding="utf-8") as f:
        lines = f.read().split("\n")
    if lines and lines[-1] == "":
        lines = lines[:-1]
    vocab_count = len(lines)
    vocab_set = set(lines)

    # Find the live model's text-embedding matrix and read its row count.
    emb_rows = None
    emb_key = None
    for k, v in f5tts.ema_model.state_dict().items():
        if hasattr(v, "shape") and len(v.shape) == 2 and "text_embed" in k.lower():
            emb_key = k
            emb_rows = int(v.shape[0])
            break
    if emb_rows is None:
        for k, v in f5tts.ema_model.state_dict().items():
            if hasattr(v, "shape") and len(v.shape) == 2 and "embed" in k.lower():
                emb_key = k
                emb_rows = int(v.shape[0])
                break

    coverage = {ch: (ch in vocab_set) for ch, _ in kurdish_letters}
    missing = [ch for ch, ok in coverage.items() if not ok]

    size_matches = emb_rows is not None and emb_rows in (vocab_count, vocab_count + 1)

    if not size_matches:
        verdict = ("EMBEDDING SIZE MISMATCH: vocab.txt is the WRONG FILE for this "
                   "checkpoint. Replace vocab.txt with the one trained alongside "
                   "these weights, then rebuild.")
    elif missing:
        verdict = ("Embedding size matches but Kurdish letters are missing from "
                   "vocab. The model expects NORMALIZED text — apply the author's "
                   "standardization to gen_text before infer().")
    else:
        verdict = ("vocab.txt and checkpoint agree and all Kurdish letters are "
                   "present. Look elsewhere (ref-text exactness / swap).")

    return {
        "vocab_file": VOCAB_FILE,
        "vocab_token_count": vocab_count,
        "embedding_key": emb_key,
        "embedding_rows": emb_rows,
        "embedding_matches_vocab": size_matches,
        "kurdish_coverage": coverage,
        "kurdish_missing": missing,
        "verdict": verdict,
    }


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
                    "ref_text_raw": cfg.get("ref_text_raw"),
                    "ref_text_phonemized": cfg["ref_text"],
                    "ref_text_file": cfg["ref_text_file"],
                    "ref_audio": cfg["ref_audio"],
                    "ref_audio_exists": os.path.exists(cfg["ref_audio"]),
                }
                for v, cfg in VOICES.items()
            }
        }

    # New debug mode: run normalize_and_g2p on arbitrary text without doing
    # inference. Use this to sanity-check the phonemizer output directly:
    #   {"input": {"debug": "test_g2p", "text": "سپاس بۆ هەوڵ"}}
    if job_input.get("debug") == "test_g2p":
        sample = job_input.get("text", "سپاس بۆ هەوڵ یا هاتین")
        try:
            return {"debug_info": {"raw": sample, "phonemized": normalize_and_g2p(sample)}}
        except Exception as e:
            return {"error": f"test_g2p failed: {e}"}

    # Diagnose the vocab/checkpoint relationship. Call:
    #   {"input": {"debug": "diagnose_vocab"}}
    # Tells you whether vocab.txt matches the checkpoint's text-embedding size,
    # and whether the Kurdish letters your users type are even in the vocab.
    if job_input.get("debug") == "diagnose_vocab":
        try:
            return {"debug_info": _diagnose_vocab()}
        except Exception as e:
            return {"error": f"diagnose_vocab failed: {e}"}

    gen_text_raw = job_input.get("text")
    if not gen_text_raw:
        return {"error": "Missing required field: 'text'"}

    voice = job_input.get("voice", DEFAULT_VOICE)
    if voice not in VOICES:
        return {"error": f"Unknown voice '{voice}'. Valid options: {list(VOICES.keys())}"}

    speed = float(job_input.get("speed", 1.0))
    remove_silence = bool(job_input.get("remove_silence", False))

    ref_audio_b64 = job_input.get("ref_audio")
    ref_text_raw_input = job_input.get("ref_text")

    voice_cfg = VOICES[voice]
    tmp_ref_path = voice_cfg["ref_audio"]
    tmp_file_to_clean = None

    # Convert the user's raw Kurdish text into the phonemic form the model
    # was actually trained on. THIS is the step that was missing before —
    # without it the model receives characters it has no learned meaning
    # for, which is why generated audio didn't match the input text.
    try:
        gen_text = normalize_and_g2p(gen_text_raw)
    except Exception as e:
        return {"error": f"G2P preprocessing failed for 'text': {e}"}

    # Hold the lock across the WHOLE swap+infer so a concurrent request can't
    # change the live weights out from under us mid-generation.
    with _infer_lock:
        try:
            mismatch_info = _load_voice_weights(voice)

            if ref_audio_b64:
                if not ref_text_raw_input:
                    return {"error": "ref_text is required when ref_audio is provided"}
                # Caller-supplied ref_text is raw text too — phonemize it the
                # same way, so it matches what the model expects.
                try:
                    ref_text = normalize_and_g2p(ref_text_raw_input)
                except Exception as e:
                    return {"error": f"G2P preprocessing failed for 'ref_text': {e}"}
                audio_bytes = base64.b64decode(ref_audio_b64)
                tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
                tmp.write(audio_bytes)
                tmp.close()
                tmp_ref_path = tmp.name
                tmp_file_to_clean = tmp.name
            else:
                # voice_cfg["ref_text"] was ALREADY phonemized once at
                # startup (see the loading loop above) — do not re-process it.
                ref_text = voice_cfg["ref_text"]
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
                "gen_text_phonemized": gen_text,  # sanity check: G2P actually ran
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
