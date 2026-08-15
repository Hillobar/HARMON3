"""Local pose estimation: a reference clip rendered down to its skeleton.

A reference with *Pose* ticked is not sent to the model. What is sent is a clip of the
same length showing that reference's skeleton on black -- the movement without the person.

The estimator runs **here**, in the app, over the frames that will actually be sent. This
was once a second ComfyUI graph and is not any more: a local ONNX pass produces an ordinary
mp4, which then goes through the same upload path as any other reference, so nothing in
``graph_builder`` knows this feature exists.

Three things in here are easy to get wrong and are the reason for most of the code:

* **The rendered clip is a section, not a file.** It starts at the row's mark and runs for
  the generated length, because posing the other 6,000 frames of a long reference is work
  whose output is discarded. The consequence is that the clip *already* starts at the mark,
  so the row must then submit ``skip_first_frames = 0`` -- see ``swap_in``, which is the
  only place both halves of that are stated together.
* **H.264 refuses odd dimensions.** The canvas is rounded down to even before encoding.
  A previous attempt at this feature hit exactly that, from the other direction.
* **A pose clip has no sound of its own**, and a reference whose soundtrack silently
  vanished would renumber every ``<Audio j>`` tag after it. The source's audio for the same
  span is muxed in.

No Qt. ``rtmlib``, ``numpy`` and ``cv2`` are imported lazily, so the rest of the app -- and
the test suite -- still runs on a machine that has none of them.
"""

from __future__ import annotations

import hashlib
import logging
import math
import re
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Callable

import av

from . import config, posefigure

log = logging.getLogger(__name__)

#: COCO-17 index -> OpenPose-18 index. OpenPose slot 1 is the neck, which COCO does not
#: have and which is synthesised from the shoulders; see ``coco17_to_openpose18``.
_COCO_TO_OPENPOSE = (0, 15, 14, 17, 16, 5, 2, 6, 3, 7, 4, 11, 8, 12, 9, 13, 10)
_OPENPOSE_NECK = 1
_COCO_LEFT_SHOULDER, _COCO_RIGHT_SHOULDER = 5, 6


class PoseError(RuntimeError):
    """Anything that stops a pose clip being produced, phrased for the user."""


@dataclass(frozen=True)
class PoseSettings:
    """Everything that changes the rendered output, and therefore the cache key.

    ``runtime`` is deliberately *not* part of the key: CPU and CUDA produce the same
    skeleton, so a clip rendered on one should not be re-rendered on the other.
    """

    model: str = config.DEFAULT_POSE_MODEL
    kpt_thr: float = config.DEFAULT_POSE_KPT_THR
    runtime: str = config.DEFAULT_POSE_RUNTIME
    style: str = config.DEFAULT_POSE_STYLE

    def key_parts(self) -> tuple[str, ...]:
        return (self.model, f"kpt={self.kpt_thr:.3f}", self.style)


@dataclass
class PoseResult:
    """What a render produced, and how well it went."""

    path: Path
    frames: int
    #: Frames where nothing was detected and the previous skeleton was held. A blank frame
    #: reads as a cut to the model, so holding is the lesser evil -- but a lot of holding
    #: means the estimator lost the subject, which is worth saying out loud.
    held: int
    provider: str
    width: int
    height: int
    fps: float
    has_audio: bool

    @property
    def held_badly(self) -> bool:
        return self.frames > 0 and self.held > self.frames // 10


# ------------------------------------------------------------------------------ caching

def cache_key(source_digest: str, settings: PoseSettings, start: int, length: int,
              canvas: tuple[int, int] | None = None, skeleton: bool = True) -> str:
    """Identify a render by its source, its section and everything that shapes it.

    Moving the mark or changing the duration makes an existing clip stale rather than
    re-usable, which is why both are in here: the frames themselves are different. So do
    the canvas and whether a skeleton was drawn -- without the latter, a scaled clip and
    a skeleton of the same section would be the same cache entry and one would be sent in
    place of the other.
    """
    parts = (source_digest, *settings.key_parts(),
             f"start={int(start)}", f"len={int(length)}",
             f"canvas={canvas[0]}x{canvas[1]}" if canvas else "canvas=source",
             "skeleton" if skeleton else "frames")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def cached_path(key: str, source_name: str, prefix: str = "pose") -> Path:
    """Where a render lives. Named after its source so labels and server errors read.

    ``main_window`` matches a node error back to a row by looking for the row's display
    name in the node label, and the label is built from the filename -- so an opaque
    ``pose_a1b2c3.mp4`` would quietly break error attribution.
    """
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(source_name).stem)[:48].strip("._-")
    return config.POSE_CACHE_DIR / f"{prefix}_{stem or 'ref'}_{key[:12]}.mp4"


def cached_clips() -> list[Path]:
    """Every rendered pose clip on disk, and any half-written one left by a cancel.

    Both prefixes: a skeleton and a clip re-encoded smaller are the same kind of thing --
    a section of a reference, rendered here and cached by content -- and live in the same
    folder. Matched by prefix rather than by suffix, so a ``.part.mp4`` abandoned
    mid-render is swept up too; it is the same wasted gigabyte and nothing will ever come
    looking for it.
    """
    try:
        return sorted(p for p in config.POSE_CACHE_DIR.iterdir()
                      if p.is_file() and p.name.startswith(("pose_", "scaled_")))
    except OSError:                       # the folder has never existed
        return []


def cache_usage() -> tuple[int, int]:
    """How many clips are cached, and how many bytes they take."""
    total = 0
    clips = cached_clips()
    for clip in clips:
        try:
            total += clip.stat().st_size
        except OSError:                   # deleted between the glob and the stat
            pass
    return len(clips), total


def clear_cache() -> tuple[int, int, list[str]]:
    """Delete every cached clip. Returns (removed, bytes freed, what would not go).

    A clip that is open elsewhere -- being uploaded, or playing in the preview -- cannot
    be deleted on Windows. That is reported rather than raised: the other forty clips
    should still go, and the one that stayed is not a broken state, just a clip.
    """
    removed, freed, failures = 0, 0, []
    for clip in cached_clips():
        try:
            size = clip.stat().st_size
            clip.unlink()
        except OSError as exc:
            failures.append(f"{clip.name}: {exc.strerror or exc}")
            continue
        removed, freed = removed + 1, freed + size
    return removed, freed, failures


def forget_all(state) -> None:
    """Drop every row's pointer at a rendered clip. Call after clearing the cache.

    Separate from deleting the files because the two go wrong differently: a row still
    pointing at a deleted clip would offer a thumbnail for a file that is not there.
    """
    for row in state.refs.videos:
        row.pose_path = None
        row.pose_section = None


def swap_in(state, resolved: dict[int, str]) -> None:
    """Point posed rows at their rendered clips. Call this on a *copy* of the state.

    ``JobRequest.snapshot`` deep-copies precisely so the submit path can rewrite reference
    filenames without touching what the user is looking at; this rides on that. The live
    row keeps pointing at the user's own file, so settings and scenes still record the
    source rather than a derivative of it.

    ``trim_start`` is zeroed in the same breath as the path is swapped, because the two
    are one fact: the rendered clip *is* the section, so applying the mark again on the
    server would take a section of a section.
    """
    for row in state.refs.videos:
        path = resolved.get(row.uid)
        if not path:
            continue
        row.local_path = path
        # A different file needs its own upload; the old name is the source's.
        row.comfy_name = None
        row.trim_start = 0.0


# ------------------------------------------------------------------------- keypoint work

def coco17_to_openpose18(keypoints, scores):
    """Remap COCO-17 keypoints to OpenPose-18, synthesising the neck.

    Needed for two reasons. ``rtmlib`` does carry this conversion, but inside its
    ``RTMPose``/``RTMO`` classes rather than ``ViTPose``; and its ``draw_skeleton`` reads a
    bare 17 with ``openpose_skeleton=True`` as *animal17* and draws the wrong thing
    entirely. Eighteen keypoints is what selects the human OpenPose skeleton.

    The neck is the midpoint of the shoulders, scored as the weaker of the two -- if one
    shoulder is a guess then so is the neck.
    """
    keypoints, scores = _as_arrays(keypoints, scores, expected=17)
    return _openpose_body(keypoints, scores, width=18)


def coco133_to_openpose134(keypoints, scores):
    """Remap COCO-WholeBody-133 to OpenPose-134: the same body work, one size up.

    The wholebody model is the only one here that carries face and hands, and its raw 133
    is not a layout ``draw_skeleton`` accepts at all under ``openpose_skeleton=True`` --
    it takes 17, 18, 26 or 134 and raises ``NotImplementedError`` on anything else. So
    without this the wholebody option downloads its weights and then fails on the first
    frame it draws.

    Past the 17 body points the two layouts are the same list -- feet, then 68 face, then
    21 keypoints per hand -- displaced by one, because OpenPose inserts the neck at index
    1. Hence the single shift rather than a second table.
    """
    keypoints, scores = _as_arrays(keypoints, scores, expected=133)
    out_kpts, out_scores = _openpose_body(keypoints, scores, width=134)
    out_kpts[:, 18:] = keypoints[:, 17:]
    out_scores[:, 18:] = scores[:, 17:]
    return out_kpts, out_scores


def _as_arrays(keypoints, scores, *, expected: int):
    import numpy as np

    keypoints = np.asarray(keypoints, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    if keypoints.shape[-2] != expected:
        raise PoseError(
            f"expected {expected} keypoints to convert, got {keypoints.shape[-2]}")
    return keypoints, scores


def _openpose_body(keypoints, scores, *, width: int):
    """The 18 body slots every OpenPose layout starts with, neck included."""
    import numpy as np

    people = keypoints.shape[0]
    out_kpts = np.zeros((people, width, 2), dtype=np.float32)
    out_scores = np.zeros((people, width), dtype=np.float32)

    neck = (keypoints[:, _COCO_LEFT_SHOULDER] + keypoints[:, _COCO_RIGHT_SHOULDER]) / 2.0
    neck_score = np.minimum(scores[:, _COCO_LEFT_SHOULDER], scores[:, _COCO_RIGHT_SHOULDER])
    out_kpts[:, _OPENPOSE_NECK] = neck
    out_scores[:, _OPENPOSE_NECK] = neck_score

    for coco_index, openpose_index in enumerate(_COCO_TO_OPENPOSE):
        out_kpts[:, openpose_index] = keypoints[:, coco_index]
        out_scores[:, openpose_index] = scores[:, coco_index]

    return out_kpts, out_scores


#: config.POSE_MODELS' keypoint layout -> what turns it into something drawable. A model
#: added to config with a layout that is not in here fails when it is chosen rather than
#: drawing the wrong skeleton.
_CONVERSIONS = {
    "coco17": coco17_to_openpose18,
    "coco133": coco133_to_openpose134,
}


# ------------------------------------------------------------------------------ styles

#: What ``openpose-torso`` changes, by link rather than by index: OpenPose hangs both hips
#: off the neck, which draws two long struts through the middle of the body and no torso
#: at all. Hanging them off the shoulders instead gives the trunk its actual width.
#:
#: A substitution, deliberately, rather than removing two links and appending two others.
#: ``draw_openpose`` treats the first seventeen links as body parts and draws them as fat
#: alpha-blended ellipses, and everything after that as a thin line -- so a link appended
#: at the end would come out hairline while the limb beside it stayed solid. Swapping in
#: place keeps every index, every colour, and the whole look apart from the one change.
_TORSO_LINKS = {
    ("neck", "right_hip"): ("right_shoulder", "right_hip"),
    ("neck", "left_hip"): ("left_shoulder", "left_hip"),
}

#: Links a style adds outright, as (start, end, colour). The pelvis bar closes the trunk:
#: with the hips hanging off the shoulders and nothing between them, the two legs read as
#: two separate things rather than as one body.
#:
#: Drawn by ``_draw_limb`` rather than added to the table, because of the same index rule.
#: There is no free index below seventeen -- OpenPose-18 uses all of them, and in
#: OpenPose-134 everything above is the hands and face, which are *meant* to be hairlines.
#: An added link therefore has to be drawn separately to come out as solid as the thighs
#: it sits between. Its colour continues the green run the hips already carry.
_ADDED_LINKS = {
    "openpose-torso": ((("right_hip", "left_hip"), [0, 255, 140]),),
}

#: Which styles are a substitution into one of rtmlib's tables at all. Keyed rather than
#: assumed from "not the default", because ``posefigure`` paints its own figure and has no
#: table to substitute into -- and a lookup that fell through to the torso arrangement for
#: it would hand back a plausible, wrong answer rather than nothing.
_SUBSTITUTIONS = {"openpose-torso": _TORSO_LINKS}

#: OpenPose-18 and OpenPose-134 share links 0-16 exactly, so one substitution table covers
#: every model. Built lazily and kept, because it is otherwise rebuilt per frame.
_skeleton_cache: dict[tuple[int, str], tuple] = {}


def skeleton_for(keypoints: int, style: str):
    """The (keypoint_info, skeleton_info) to draw with, or ``None`` for rtmlib's own.

    ``None`` rather than a copy of the stock table on the default path: the stock path
    stays exactly the code it was, so a style that changes nothing cannot change anything.
    A style that draws itself rather than substituting -- see ``posefigure`` -- gets the
    same ``None``, and never reaches here anyway because ``draw`` branches on it first.
    """
    substitutions = _SUBSTITUTIONS.get(style)
    if substitutions is None:
        return None

    cached = _skeleton_cache.get((keypoints, style))
    if cached is not None:
        return cached

    from rtmlib.visualization import skeleton as skeleton_mod

    name = {18: "openpose18", 134: "openpose134"}.get(keypoints)
    if name is None:
        raise PoseError(f"no OpenPose skeleton for {keypoints} keypoints")
    table = getattr(skeleton_mod, name)

    links = {}
    for index, info in table["skeleton_info"].items():
        replacement = substitutions.get(tuple(info["link"]))
        links[index] = dict(info, link=replacement) if replacement else dict(info)

    result = (table["keypoint_info"], links)
    _skeleton_cache[(keypoints, style)] = result
    return result


def draw(canvas, keypoints, scores, settings: PoseSettings, *, radius: int,
         line_width: int, facing=None):
    """Draw every subject in the configured style. Returns the canvas to keep.

    The return value matters and is not the argument: the OpenPose style alpha-blends its
    limbs, and ``cv2.addWeighted`` hands back a *new* array -- so everything drawn after
    the first limb lands on a copy, and a canvas passed in by reference comes back
    untouched and completely black.

    ``facing`` is the running front-or-back estimate the ``figure`` style draws from, and
    is ignored by the other two. Passing nothing leaves it frontal, which is what a caller
    with no clip to track across -- a test, a single frame -- should get.
    """
    from rtmlib import draw_skeleton
    from rtmlib.visualization.draw import draw_openpose

    count = keypoints.shape[-2]
    if settings.style == posefigure.STYLE:
        if count < 18:
            raise PoseError(f"no figure to draw from {count} keypoints")
        for person in range(keypoints.shape[0]):
            canvas = posefigure.paint(canvas, keypoints[person], scores[person],
                                      kpt_thr=settings.kpt_thr, radius=radius,
                                      line_width=line_width, facing=facing)
        return canvas

    custom = skeleton_for(count, settings.style)
    if custom is None:
        return draw_skeleton(canvas, keypoints, scores, openpose_skeleton=True,
                             kpt_thr=settings.kpt_thr, radius=radius,
                             line_width=line_width)

    keypoint_info, skeleton_info = custom
    slot = {info["name"]: index for index, info in keypoint_info.items()}
    for person in range(keypoints.shape[0]):
        # The same arguments draw_skeleton passes on for the OpenPose styles, so the two
        # paths differ in the table and in nothing else.
        canvas = draw_openpose(canvas, keypoints[person], scores[person],
                               keypoint_info, skeleton_info, settings.kpt_thr,
                               radius * 2, alpha=0.6, line_width=line_width * 2)
        for (start, end), colour in _ADDED_LINKS.get(settings.style, ()):
            canvas = _draw_limb(canvas, keypoints[person], scores[person],
                                slot[start], slot[end], colour,
                                kpt_thr=settings.kpt_thr, line_width=line_width * 2)
    return canvas


def _draw_limb(canvas, keypoints, scores, start: int, end: int, colour,
               *, kpt_thr: float, line_width: int):
    """One more limb, in the shape ``draw_openpose`` gives the seventeen it knows about.

    The geometry is copied from it deliberately -- an ellipse along the bone rather than a
    line -- so an added link is indistinguishable from a built-in one. Same guards, too:
    a limb whose end is off the canvas or unconfident is skipped rather than drawn to a
    point clamped at the edge.

    The drawing itself now lives in ``posefigure.bone``, which is the same code moved down
    so that the style that paints its own figure and the style that adds one link to
    rtmlib's are not two implementations of one ellipse. ``alpha=0.6`` is what preserves
    the pixels this has always produced; see that function for what rtmlib does with it.
    """
    return posefigure.bone(canvas, keypoints, scores, start, end, colour,
                           kpt_thr=kpt_thr, line_width=line_width, alpha=0.6)


def _pick_subject(bboxes, previous):
    """One subject per reference: the biggest box, or whichever continues the last one.

    A dance reference has one dancer, and switching mid-clip to a passer-by is worse than
    any amount of jitter. Continuity wins over size whenever the two disagree.
    """
    if bboxes is None or len(bboxes) == 0:
        return previous
    boxes = [tuple(float(v) for v in box[:4]) for box in bboxes]

    if previous is not None:
        best, best_iou = None, 0.0
        for box in boxes:
            score = _iou(box, previous)
            if score > best_iou:
                best, best_iou = box, score
        if best is not None and best_iou > 0.2:
            return best

    return max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))


def _iou(a, b) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    overlap = (right - left) * (bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - overlap
    return overlap / union if union > 0 else 0.0


# ------------------------------------------------------------------------------- runtime

def preferred_device(runtime: str = config.DEFAULT_POSE_RUNTIME) -> tuple[str, str]:
    """Return (device, explanation) for what the estimator will actually run on.

    Answered without loading a model, so the UI can say "CPU, because CUDA is not
    available here" before anyone waits two minutes to find out.
    """
    if runtime == "cpu":
        return "cpu", "CPU, because the runtime is set to CPU"

    try:
        import onnxruntime
    except ImportError:
        return "cpu", "CPU, because onnxruntime is not installed"

    # Since 1.21 this is what makes the CUDA and cuDNN wheels in site-packages visible to
    # the loader; without it a venv with no system CUDA toolkit reports no CUDA provider.
    try:
        onnxruntime.preload_dlls()
    except Exception as exc:                       # older runtime, or nothing to preload
        log.debug("preload_dlls() unavailable or failed: %s", exc)

    if "CUDAExecutionProvider" in onnxruntime.get_available_providers():
        return "cuda", "CUDA"
    if runtime == "cuda":
        return "cpu", ("CPU, because CUDA was asked for but onnxruntime reports no CUDA "
                       "provider - check onnxruntime-gpu and the nvidia-*-cu13 wheels")
    return "cpu", "CPU, because no CUDA provider is available"


def model_path(model: str) -> Path:
    try:
        filename, _url, _layout = config.POSE_MODELS[model]
    except KeyError:
        raise PoseError(f"unknown pose model {model!r}") from None
    return config.POSE_MODELS_DIR / filename


def missing_models(settings: PoseSettings) -> list[tuple[str, str, Path]]:
    """The (label, url, destination) triples that still have to be downloaded."""
    path = model_path(settings.model)
    if path.is_file():
        return []
    _filename, url, _layout = config.POSE_MODELS[settings.model]
    return [(settings.model, url, path)]


def download(url: str, destination: Path,
             on_progress: Callable[[int, int], None] | None = None,
             should_stop: Callable[[], bool] | None = None) -> Path:
    """Fetch a weights file, atomically. Same .part-then-replace idiom as the downloader.

    A half-written 1.2 GB ONNX that looks complete is a confusing failure much later, so
    the real name only appears once the bytes are all there.
    """
    import requests

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")

    with requests.get(url, stream=True, timeout=60) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length") or 0)
        done = 0
        with partial.open("wb") as fh:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if should_stop is not None and should_stop():
                    partial.unlink(missing_ok=True)
                    raise PoseError("the download was cancelled")
                fh.write(chunk)
                done += len(chunk)
                if on_progress is not None:
                    on_progress(done, total)

    partial.replace(destination)
    return destination


class Estimator:
    """A loaded detector and pose model, held only for as long as a render takes.

    ComfyUI is on the same GPU. Holding a session open between runs would keep a couple of
    gigabytes of VRAM away from the thing this app exists to drive, so ``render`` opens
    one, uses it, and closes it in a ``finally``.
    """

    def __init__(self, settings: PoseSettings):
        self.settings = settings
        weights = model_path(settings.model)
        if not weights.is_file():
            raise PoseError(
                f"the {settings.model} weights are not downloaded yet ({weights})")

        try:
            from rtmlib import YOLOX, ViTPose
        except ImportError as exc:
            raise PoseError(
                "pose estimation needs rtmlib and onnxruntime - "
                "run setup.bat, or pip install -r requirements.txt") from exc

        self.device, self.explanation = preferred_device(settings.runtime)
        _filename, _url, self.layout = config.POSE_MODELS[settings.model]

        # The detector is handed its URL rather than a local path: rtmlib knows how to
        # unpack OpenMMLab's zipped SDK bundles and how to fall back to the HuggingFace
        # mirror, and reimplementing that here would buy nothing.
        self.detector = YOLOX(onnx_model=config.POSE_DETECTOR[1],
                              model_input_size=config.POSE_DETECTOR_INPUT_SIZE,
                              backend="onnxruntime", device=self.device)
        self.pose = ViTPose(onnx_model=str(weights),
                            model_input_size=config.POSE_INPUT_SIZE,
                            backend="onnxruntime", device=self.device)

    def keypoints_for(self, image, box):
        """Estimate one subject, returning OpenPose-18 keypoints and scores, or None."""
        import numpy as np

        if box is None:
            return None
        keypoints, scores = self.pose(image, bboxes=np.asarray([box], dtype=np.float32))
        if keypoints is None or len(keypoints) == 0:
            return None
        # Every layout is converted, not just COCO-17: draw_skeleton's OpenPose branch
        # accepts 18, 26 and 134 and nothing else, so an unconverted layout reaches it as
        # a NotImplementedError several minutes into a render.
        try:
            convert = _CONVERSIONS[self.layout]
        except KeyError:
            raise PoseError(
                f"no OpenPose conversion for the {self.layout} keypoint layout") from None
        return convert(keypoints, scores)

    def close(self) -> None:
        """Drop both sessions so the GPU memory goes back before ComfyUI is asked for it."""
        self.detector = self.pose = None


# ------------------------------------------------------------------------------ the pass

def render(source: str | Path, start: int, length: int, settings: PoseSettings,
           out_path: str | Path,
           on_frame: Callable[[int, int], None] | None = None,
           should_stop: Callable[[], bool] | None = None,
           canvas: tuple[int, int] | None = None,
           skeleton: bool = True) -> PoseResult:
    """Render ``length`` frames of ``source`` from frame ``start`` into a new clip.

    Two jobs, because they are the same job. With ``skeleton`` the frames are replaced by
    the pose drawn from them; without it they are passed through. Either way this is
    "a section of a reference, re-encoded at a chosen size", and both need the seeking,
    the section arithmetic, the even-dimension rule and -- above all -- the soundtrack
    carried across, because a reference that silently loses its audio renumbers every
    ``<Audio j>`` after it. Splitting them would mean saying all of that twice.

    ``canvas`` is the frame size to write at, defaulting to the source's own.

    ``on_frame(done, total)`` is called as it goes, and ``should_stop()`` is checked every
    frame -- this takes tens of seconds and the user has to be able to give up on it.

    Written to a ``.part`` and renamed once it is whole, the same way downloads and the
    settings file are. Without that, cancelling half way would leave a truncated clip at
    the name the cache looks for, and the next run would send six frames and say nothing
    about it.
    """
    source = Path(source)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    partial = out_path.with_suffix(".part" + out_path.suffix)

    # Only when there is a skeleton to draw. A scale-only pass has no business loading a
    # gigabyte of ONNX, and would fail on a machine with no weights downloaded.
    estimator = Estimator(settings) if skeleton else None
    try:
        with av.open(str(source)) as container:
            if not container.streams.video:
                raise PoseError(f"{source.name} has no video track to read")
            stream = container.streams.video[0]
            fps = float(stream.average_rate or stream.guessed_rate or config.FPS)

            # Rounded down to even: H.264 in yuv420p refuses odd dimensions, and a
            # reference that fails to encode after two minutes of estimation is a poor
            # way to find that out.
            source_size = (even(stream.codec_context.width),
                           even(stream.codec_context.height))
            width, height = (even(canvas[0]), even(canvas[1])) if canvas else source_size
            if min(width, height, *source_size) < 2:
                raise PoseError(f"{source.name} reports an unusable size")

            frames, held = _render_frames(
                container, stream, estimator, settings, partial,
                start=start, length=length, fps=fps, width=width, height=height,
                on_frame=on_frame, should_stop=should_stop)

        has_audio = _mux_audio(source, partial, start / fps, frames / fps)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    finally:
        # Including on the way out of a cancel: ComfyUI is on this GPU and is about to be
        # asked for most of it.
        if estimator is not None:
            estimator.close()

    partial.replace(out_path)
    return PoseResult(path=out_path, frames=frames, held=held,
                      provider=estimator.explanation if estimator else "",
                      width=width, height=height, fps=fps, has_audio=has_audio)


def _render_frames(container, stream, estimator, settings, destination, *, start, length,
                   fps, width, height, on_frame, should_stop) -> tuple[int, int]:
    import numpy as np

    # Scaled to the frame: the library's 2px default is invisible on anything above SD,
    # and the skeleton is the entire signal this clip carries.
    line_width = max(2, round(height / 180))
    radius = max(2, round(height / 260))

    black = np.zeros((height, width, 3), dtype=np.uint8)
    written = held = 0
    box = None
    last_pose = None
    facing = posefigure.Facing()
    since_detect = config.POSE_DETECT_EVERY

    with av.open(str(destination), "w") as output:
        out_stream = output.add_stream("libx264", rate=Fraction(stream.average_rate
                                                                or stream.guessed_rate
                                                                or config.FPS))
        out_stream.width, out_stream.height = width, height
        out_stream.pix_fmt = "yuv420p"
        out_stream.options = {"crf": str(config.POSE_VIDEO_CRF), "preset": "medium"}

        for frame in _frames_from(container, stream, start, length, fps):
            if should_stop is not None and should_stop():
                raise PoseError("the pass was cancelled")

            image = _fit(frame.to_ndarray(format="bgr24"), width, height)

            if estimator is None:
                # Scale only: the frames themselves are what is being sent.
                canvas = image
            else:
                if since_detect >= config.POSE_DETECT_EVERY or box is None:
                    box = _pick_subject(estimator.detector(image), box)
                    since_detect = 0
                since_detect += 1

                pose = estimator.keypoints_for(image, box)
                if pose is None:
                    # Hold rather than blink: a blank frame reads as a cut to the model.
                    pose, held = last_pose, held + 1
                else:
                    last_pose = pose
                    # Only a fresh estimate votes. A held frame is the previous keypoints
                    # drawn again, and folding those in would count one frame's evidence
                    # as many and talk the estimate into whatever it already believed.
                    facing.update(pose[0][0], pose[1][0], settings.kpt_thr)
                    box = _box_from(pose, settings.kpt_thr, width, height) or box

                canvas = black.copy()
                if pose is not None:
                    keypoints, scores = pose
                    canvas = draw(canvas, keypoints, scores, settings, radius=radius,
                                  line_width=line_width, facing=facing)

            out_frame = av.VideoFrame.from_ndarray(
                np.ascontiguousarray(canvas), format="bgr24")
            for packet in out_stream.encode(out_frame):
                output.mux(packet)

            written += 1
            if on_frame is not None:
                on_frame(written, length)

        for packet in out_stream.encode():
            output.mux(packet)

    if written == 0:
        raise PoseError("the section is empty - nothing was decoded")
    return written, held


def _fit(image, width: int, height: int):
    """Bring a decoded frame to the canvas being written.

    A slice when it is already the right size, which is the case for a pose render at the
    source's own dimensions and trims nothing but the odd row an even-rounded canvas
    leaves behind. A real resample otherwise -- the previous slice-only version would have
    *cropped* a reference asked to be smaller, quietly sending a corner of it.
    """
    if image.shape[1] == width and image.shape[0] == height:
        return image
    if image.shape[1] >= width and image.shape[0] >= height:
        import cv2

        # INTER_AREA is the one that averages rather than samples, which is what keeps a
        # downscaled reference from shimmering between frames.
        return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

    import cv2

    return cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)


def _frames_from(container, stream, start: int, length: int, fps: float):
    """Decode ``length`` frames from index ``start``, seeking rather than scanning.

    A mark 4,300 frames into a reference is two and a half minutes of decoding to skip
    past. Seeking lands on the keyframe before it and the rest is discarded by index.
    """
    if start > 0:
        try:
            target = int((start / fps) / float(stream.time_base))
            container.seek(target, stream=stream, backward=True, any_frame=False)
        except Exception as exc:      # a stream that cannot seek is still decodable
            log.info("Seek to frame %s failed (%s); decoding from the start", start, exc)

    emitted = 0
    for frame in container.decode(stream):
        index = start if frame.pts is None else round(
            float(frame.pts * stream.time_base) * fps)
        if index < start:
            continue
        yield frame
        emitted += 1
        if emitted >= length:
            return


def _box_from(pose, kpt_thr: float, width: int, height: int):
    """A bounding box around the keypoints just found, to track the subject forward.

    Cheaper and steadier than re-detecting: the person is where they were a frame ago.
    """
    import numpy as np

    keypoints, scores = pose
    visible = keypoints[0][scores[0] >= kpt_thr]
    if len(visible) < 4:
        return None
    left, top = np.min(visible, axis=0)
    right, bottom = np.max(visible, axis=0)
    pad_x = max(8.0, (right - left) * 0.25)
    pad_y = max(8.0, (bottom - top) * 0.25)
    return (max(0.0, float(left - pad_x)), max(0.0, float(top - pad_y)),
            min(float(width), float(right + pad_x)), min(float(height), float(bottom + pad_y)))


def _mux_audio(source: Path, video_path: Path, start_s: float, duration_s: float) -> bool:
    """Copy the source's audio for the same span into the finished pose clip.

    Without this a posed reference would arrive silent, and a video whose soundtrack goes
    missing does not just lose its sound -- it stops emitting its ``<Audio j>`` tag, which
    renumbers every standalone audio after it and quietly repoints the prompt.
    """
    with av.open(str(source)) as probe:
        if not probe.streams.audio:
            return False

    merged = video_path.with_suffix(".withaudio.mp4")
    try:
        with av.open(str(video_path)) as video_in, \
                av.open(str(source)) as audio_in, \
                av.open(str(merged), "w") as output:
            in_video = video_in.streams.video[0]
            in_audio = audio_in.streams.audio[0]

            out_video = output.add_stream_from_template(in_video)
            out_audio = output.add_stream("aac", rate=in_audio.codec_context.sample_rate)

            for packet in video_in.demux(in_video):
                if packet.dts is None:
                    continue
                packet.stream = out_video
                output.mux(packet)

            resampler = av.AudioResampler(format=out_audio.codec_context.format,
                                          layout=out_audio.codec_context.layout,
                                          rate=out_audio.codec_context.sample_rate)
            end_s = start_s + duration_s
            try:
                audio_in.seek(int(start_s / float(in_audio.time_base)), stream=in_audio,
                              backward=True)
            except Exception:         # decoding from the start still lands on the window
                pass

            for frame in audio_in.decode(in_audio):
                at = float(frame.pts * in_audio.time_base) if frame.pts is not None else 0.0
                if at < start_s:
                    continue
                if at >= end_s:
                    break
                for resampled in resampler.resample(frame):
                    resampled.pts = None
                    for packet in out_audio.encode(resampled):
                        output.mux(packet)

            for packet in out_audio.encode():
                output.mux(packet)
    except Exception as exc:
        # A silent pose clip is a real loss but not a fatal one, and it is better than
        # failing the whole run over a soundtrack.
        log.warning("Could not carry %s's audio into the pose clip: %s", source.name, exc)
        merged.unlink(missing_ok=True)
        return False

    merged.replace(video_path)
    return True


def frames_for(duration_seconds: float) -> int:
    """The generated length a pose clip has to cover, in frames."""
    from . import mathmirror

    return mathmirror.frames_from_seconds(mathmirror.clamp_duration(duration_seconds))


def section_for(row, duration_seconds: float) -> tuple[int, int]:
    """The (start, length) of the section a row's pose clip must cover.

    The same arithmetic the graph does, so the rendered clip and the loader agree on which
    frames those are: from the mark, for the generated length.
    """
    return max(0, int(row.trim_start)), frames_for(duration_seconds)


def forget_stale(row, duration_seconds: float) -> bool:
    """Drop a row's pose pointer if it no longer describes what would be sent.

    Deliberately a tuple comparison and a stat, not a re-hash: this runs every time
    anything in the editor changes, and hashing a 400 MB reference on every keystroke to
    discover that nothing moved would be its own bug. A file edited in place is caught
    later, by ``resolve``, which does hash -- and lands on a different name.
    """
    if not getattr(row, "pose_path", None):
        return False
    if row.pose_section == section_for(row, duration_seconds) and Path(row.pose_path).is_file():
        return False
    row.pose_path = None
    row.pose_section = None
    return True


def canvas_for(row) -> tuple[int, int] | None:
    """The frame size this row is re-encoded at, or None for the source's own.

    Read off the row rather than passed in, because it is the row's own property: the
    slider on a video sets a share of the canvas the node would otherwise use.
    """
    from . import scaling

    if not (getattr(row, "scales", False) and row.width and row.height):
        return None
    return scaling.video_target_size(row.width, row.height, row.scale_percent)


def needs_render(row) -> bool:
    """Whether this row has a local pass to run at all -- a skeleton, a rescale, or both."""
    return bool(row.local_path) and (row.poses or canvas_for(row) is not None)


def resolve(row, duration_seconds: float, settings: PoseSettings) -> tuple[Path, bool]:
    """Where this row's rendered clip belongs, and whether it is already there."""
    from .comfy_http import sha256_of

    start, length = section_for(row, duration_seconds)
    canvas = canvas_for(row)
    key = cache_key(sha256_of(Path(row.local_path)), settings, start, length,
                    canvas, row.poses)
    path = cached_path(key, Path(row.local_path).name,
                       prefix="pose" if row.poses else "scaled")
    return path, path.is_file()


def even(value: int) -> int:
    """Round down to an even number. H.264 in yuv420p will not take anything else."""
    return (int(math.floor(value)) // 2) * 2
