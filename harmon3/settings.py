"""Persistent application settings.

Human-editable JSON stored beside the app (or under $HARMON3_HOME). Window geometry lives
separately in a QSettings ini because those values are opaque QByteArray blobs.

Writes are atomic: a crash mid-save can never leave a truncated settings file behind.
"""

from __future__ import annotations

import json
import logging
import os
import tempfile
from pathlib import Path

from . import config, prompt as prompt_mod
from .refs import RefSet

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1

DEFAULTS: dict = {
    "schema_version": SCHEMA_VERSION,
    "server_url": config.DEFAULT_SERVER_URL,
    #: Optional: ComfyUI's output directory as this machine sees it. When set and the
    #: produced file is found there, the /view download is skipped.
    "server_output_dir": "",
    #: Where scenes are saved. Empty means the default folder beside the app.
    "scenes_dir": "",
    "aspect_ratio": config.DEFAULT_ASPECT_RATIO,
    "megapixels": config.DEFAULT_MEGAPIXELS,
    "duration_seconds": config.DEFAULT_DURATION,
    "seed": config.DEFAULT_SEED,
    "randomize_seed": True,
    "steps": config.DEFAULT_STEPS,
    "sampler_name": config.DEFAULT_SAMPLER,
    "scheduler": config.DEFAULT_SCHEDULER,
    "schedule": config.DEFAULT_SCHEDULE,
    "upscale_method": config.DEFAULT_UPSCALE_METHOD,
    "shift_video": config.DEFAULT_SHIFT_VIDEO,
    "ref_image_size": config.DEFAULT_REF_IMAGE_SIZE,
    "sage_attention": config.DEFAULT_SAGE_ATTENTION,
    #: The prompt's named sections. None -> fall back to the workflow's own prompt.
    "prompt_sections": None,
    "refs": None,                 # None -> seed from the workflow's baked LoadImage nodes
    #: On by default: the workflow carries the Model Preview Override node, so the live
    #: preview is a real decoded picture rather than a latent smear.
    "show_previews": True,
    #: kind -> the folder its file dialog last used. Kept apart because image, video and
    #: audio references usually live in different places.
    "last_browse_dirs": {},
    #: Name of the scene the editor was working on, reloaded on the next launch.
    "current_scene": "",
    #: sha256 -> the name the file already carries in ComfyUI's input dir
    "upload_cache": {},
    #: Which ONNX weights the Pose toggle renders with, and where they run. See
    #: config.POSE_MODELS / POSE_RUNTIMES.
    "pose_model": config.DEFAULT_POSE_MODEL,
    "pose_runtime": config.DEFAULT_POSE_RUNTIME,
    #: Keypoints below this score are not drawn.
    "pose_kpt_thr": config.DEFAULT_POSE_KPT_THR,
    #: How those keypoints are joined up. Independent of the model: see config.POSE_STYLES.
    "pose_style": config.DEFAULT_POSE_STYLE,
}


def pose_settings(data: dict):
    """The pose settings as the estimator wants them, with anything unusable defaulted."""
    from .pose import PoseSettings

    model = data.get("pose_model") or config.DEFAULT_POSE_MODEL
    runtime = data.get("pose_runtime") or config.DEFAULT_POSE_RUNTIME
    style = data.get("pose_style") or config.DEFAULT_POSE_STYLE
    try:
        threshold = float(data.get("pose_kpt_thr", config.DEFAULT_POSE_KPT_THR))
    except (TypeError, ValueError):
        threshold = config.DEFAULT_POSE_KPT_THR

    return PoseSettings(
        model=model if model in config.POSE_MODELS else config.DEFAULT_POSE_MODEL,
        runtime=runtime if runtime in config.POSE_RUNTIMES else config.DEFAULT_POSE_RUNTIME,
        kpt_thr=min(0.95, max(0.05, threshold)),
        style=style if style in config.POSE_STYLES else config.DEFAULT_POSE_STYLE,
    )


def load_settings(path: Path | None = None) -> dict:
    path = path or config.SETTINGS_PATH
    data = dict(DEFAULTS)

    if path.is_file():
        try:
            with path.open("r", encoding="utf-8") as fh:
                stored = json.load(fh)
            if isinstance(stored, dict):
                data.update({k: v for k, v in stored.items() if k in DEFAULTS})
                _migrate(data, stored)
        except (OSError, ValueError) as exc:
            log.warning("Could not read %s (%s); using defaults", path, exc)

    if not isinstance(data.get("upload_cache"), dict):
        data["upload_cache"] = {}
    if not isinstance(data.get("last_browse_dirs"), dict):
        data["last_browse_dirs"] = {}
    return data


def _migrate(data: dict, stored: dict) -> None:
    """Carry values forward from settings files written by an older version."""
    # last_browse_dir (one folder for everything) -> last_browse_dirs (one per kind).
    legacy_dir = stored.get("last_browse_dir")
    if legacy_dir and not data.get("last_browse_dirs"):
        data["last_browse_dirs"] = {
            kind: legacy_dir for kind in ("image", "video", "audio")
        }

    # prompt_text (one box) -> prompt_sections (six).
    legacy_prompt = stored.get("prompt_text")
    if legacy_prompt and data.get("prompt_sections") is None:
        data["prompt_sections"] = prompt_mod.from_legacy(legacy_prompt)


def save_settings(data: dict, path: Path | None = None) -> None:
    path = path or config.SETTINGS_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    # Serialise from a snapshot. json.dump with indent uses the pure-Python encoder,
    # which iterates dicts as it writes -- so a live upload_cache being written by the
    # job thread mid-save would raise "dictionary changed size during iteration".
    payload = {k: data.get(k, v) for k, v in DEFAULTS.items()}
    payload["upload_cache"] = dict(payload.get("upload_cache") or {})
    payload["schema_version"] = SCHEMA_VERSION

    handle, temp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".settings-", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        os.replace(temp_name, path)
    except Exception as exc:
        # Never let a failed save escape: it is called from the close handler, where an
        # exception would skip the thread shutdown that follows it.
        log.error("Could not write %s: %s", path, exc)
        Path(temp_name).unlink(missing_ok=True)


def apply_to_state(state, data: dict) -> None:
    """Overlay saved settings onto a state already seeded from the workflow.

    Fields left as None in the settings keep the workflow's own value, which is what makes
    a first launch reproduce the shipped run exactly.
    """
    state.aspect_ratio = data.get("aspect_ratio", state.aspect_ratio)
    state.megapixels = float(data.get("megapixels", state.megapixels))
    state.duration_seconds = float(data.get("duration_seconds", state.duration_seconds))
    state.seed = int(data.get("seed", state.seed))
    state.steps = int(data.get("steps", state.steps))
    state.sampler_name = str(data.get("sampler_name", state.sampler_name))
    state.scheduler = str(data.get("scheduler", state.scheduler))
    state.schedule = str(data.get("schedule", state.schedule))
    state.upscale_method = str(data.get("upscale_method", state.upscale_method))
    state.shift_video = float(data.get("shift_video", state.shift_video))
    state.ref_image_size = str(data.get("ref_image_size", state.ref_image_size))
    state.sage_attention = bool(data.get("sage_attention", state.sage_attention))

    if data.get("prompt_sections") is not None:
        state.prompt_sections = prompt_mod.normalise(data["prompt_sections"])
    if data.get("refs") is not None:
        state.refs = RefSet.from_list(data["refs"])


def capture_from_state(data: dict, state) -> dict:
    """Fold the current editor state back into a settings dict, ready to save."""
    data = dict(data)
    data.update({
        "aspect_ratio": state.aspect_ratio,
        "megapixels": state.megapixels,
        "duration_seconds": state.duration_seconds,
        "seed": state.seed,
        "steps": state.steps,
        "sampler_name": state.sampler_name,
        "scheduler": state.scheduler,
        "schedule": state.schedule,
        "upscale_method": state.upscale_method,
        "shift_video": state.shift_video,
        "ref_image_size": state.ref_image_size,
        "sage_attention": state.sage_attention,
        "prompt_sections": dict(state.prompt_sections),
        "refs": state.refs.to_list(),
    })
    return data
