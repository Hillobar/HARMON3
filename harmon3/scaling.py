"""Per-image downscaling, applied before upload.

``MiniMaxH3ReferenceToVideo`` has one ``ref_image_size`` widget for every reference it
receives, so there is no per-image setting to expose. What it does have -- checked against
``comfy_extras/nodes_minimax_h3.py`` -- is a scale factor of ``min(1.0, ...)`` in *both*
of its modes: a reference is never enlarged, only capped. So an image made smaller on the
way in stays smaller, and shrinking one here is a per-image size control expressed as a
ceiling the node will not raise.

It is worth doing because reference tokens ride through every sampling step. Under
``max``, a 4000x3000 reference encodes at 2720x2048 -- 21,760 tokens -- while the same
image at 40% encodes at 1600x1216 and costs 7,600. The identity reference can keep the
full 2048 while the others stop paying for detail nobody asked for.

Sizes are snapped to multiples of 32 because that is the grid the node rounds to; matching
it means the size shown here is the size actually encoded, rather than one the node then
nudges by up to sixteen pixels.

Videos are covered too, and mean something different by it. The same file shows they never
consult ``ref_image_size`` at all: every reference video is fitted to a fixed ~1344x768
canvas, and only a clip already smaller than that keeps its own size. So a clip's ceiling
is a share of *that canvas* rather than of the file -- see ``video_target_size`` -- and it
is applied by re-encoding the section, which ``pose`` already knows how to do.
"""

from __future__ import annotations

import hashlib
import logging
import re
from pathlib import Path

from . import config

log = logging.getLogger(__name__)

#: The node's canvas multiple. Sizes off this grid get rounded onto it there instead.
GRID = 32

#: The slider's range. Ten per cent of a 4000px image is still 400px, and below that a
#: reference stops carrying anything the model can use.
MIN_PERCENT = 10
MAX_PERCENT = 100

#: The short edge is never taken below this, whatever the slider says.
#:
#: Not a quality preference -- an aspect-ratio one. Each axis is rounded onto the 32-pixel
#: grid *independently*, which is what the node does, and the error that introduces is
#: relative: at 384x288 it is a rounding, while at 80x60 both axes land on 64 and a 4:3
#: reference arrives square. Eight grid cells keeps the distortion under a few per cent.
#: A source already smaller than this is left alone rather than enlarged to meet it.
MIN_SHORT_EDGE = 256


class ScaleError(RuntimeError):
    """An image could not be rescaled, phrased for the user."""


def clamp_percent(value) -> int:
    try:
        percent = int(round(float(value)))
    except (TypeError, ValueError):
        return MAX_PERCENT
    return max(MIN_PERCENT, min(MAX_PERCENT, percent))


def target_size(width: int, height: int, percent: int) -> tuple[int, int]:
    """The size an image is written at, on the node's own 32-pixel grid.

    Never larger than the source: the point of this is a ceiling, and rounding a 1010px
    axis *up* to 1024 to satisfy the grid would be an upscale nobody asked for.
    """
    width, height = int(width or 0), int(height or 0)
    if width <= 0 or height <= 0:
        return 0, 0

    # One scale for both axes, so the floor cannot squash the picture on its way in.
    # Capped at 1.0 because this is a ceiling: nothing here ever enlarges anything.
    scale = min(1.0, max(clamp_percent(percent) / 100, MIN_SHORT_EDGE / min(width, height)))
    return (min(max(GRID, round(width * scale / GRID) * GRID), width),
            min(max(GRID, round(height * scale / GRID) * GRID), height))


# -------------------------------------------------------------------------------- video

#: What the node fits a reference *video* to, copied from ``adapt_canvas`` in
#: nodes_minimax_h3.py: a 768-pixel short edge, capped at 768*1344 of area, each axis
#: rounded to 32. It does this to every clip regardless of ``ref_image_size``, which
#: videos never consult at all.
VIDEO_SHORT_EDGE = 768
VIDEO_MAX_PIXELS = 768 * 1344


def video_canvas(width: int, height: int) -> tuple[int, int]:
    """The size the node would encode this clip at if nothing intervened.

    Including its "source smaller than the canvas" branch, which is the node's own
    never-upscale guard and the reason shrinking a clip here has any effect at all.
    """
    width, height = int(width or 0), int(height or 0)
    if width <= 0 or height <= 0:
        return 0, 0

    ratio = width / height
    nominal_w, nominal_h = ((VIDEO_SHORT_EDGE * ratio, VIDEO_SHORT_EDGE) if ratio >= 1.0
                            else (VIDEO_SHORT_EDGE, VIDEO_SHORT_EDGE / ratio))
    if nominal_w * nominal_h > VIDEO_MAX_PIXELS:
        shrink = (VIDEO_MAX_PIXELS / (nominal_w * nominal_h)) ** 0.5
        nominal_w, nominal_h = nominal_w * shrink, nominal_h * shrink

    canvas_w = max(GRID, round(nominal_w / GRID) * GRID)
    canvas_h = max(GRID, round(nominal_h / GRID) * GRID)
    if width * height < canvas_w * canvas_h:
        return max(GRID, round(width / GRID) * GRID), max(GRID, round(height / GRID) * GRID)
    return canvas_w, canvas_h


def video_target_size(width: int, height: int, percent: int) -> tuple[int, int]:
    """The frame size a clip is re-encoded at, as a share of what the node would use.

    A share of the *canvas*, not of the source file, and that difference is the whole
    reason this exists as a separate function. Every clip above about a megapixel is
    flattened onto the same 1344x768 canvas, so a slider reading "50% of a 4K source"
    would move from 3840 to 1920 and change precisely nothing about what the model
    receives. Measured against the canvas instead, every position on the slider does
    something.
    """
    canvas_w, canvas_h = video_canvas(width, height)
    if canvas_w <= 0 or canvas_h <= 0:
        return 0, 0

    scale = min(1.0, max(clamp_percent(percent) / 100,
                         MIN_SHORT_EDGE / min(canvas_w, canvas_h)))
    return (min(max(GRID, round(canvas_w * scale / GRID) * GRID), canvas_w),
            min(max(GRID, round(canvas_h * scale / GRID) * GRID), canvas_h))


def describe_video(width, height, percent: int) -> str:
    """What a clip's slider currently amounts to, for the readout beside it."""
    if not width or not height:
        return "size not known yet"
    canvas = video_canvas(width, height)
    target = video_target_size(width, height, percent)
    if target == canvas:
        return f"{target[0]}x{target[1]} per frame  -  the model's own size for this clip"
    share = (target[0] * target[1]) / (canvas[0] * canvas[1]) * 100
    return (f"{target[0]}x{target[1]} per frame  -  {share:.0f}% of the tokens "
            f"(normally {canvas[0]}x{canvas[1]})")


#: The node's own cap on a reference image's short edge under ``ref_image_size = max``.
REF_IMAGE_SHORT_EDGE = 2048


def node_image_size(width: int, height: int, mode: str,
                    output: tuple[int, int]) -> tuple[int, int]:
    """What the node will encode a reference *image* at, given the file it receives.

    A mirror of the sizing in ``MiniMaxH3ReferenceToVideo.execute``, kept here so the
    bundle can state what the model actually sees rather than what was uploaded. Both of
    its branches are ``min(1.0, ...)``: a reference is capped, never enlarged.
    """
    width, height = int(width or 0), int(height or 0)
    if width <= 0 or height <= 0:
        return 0, 0

    if mode == "match":
        out_w, out_h = int(output[0] or 0), int(output[1] or 0)
        if out_w <= 0 or out_h <= 0:
            return 0, 0
        scale = min(1.0, ((out_w * out_h) / (width * height)) ** 0.5)
    else:
        scale = min(1.0, REF_IMAGE_SHORT_EDGE / min(width, height))

    return (max(GRID, round(width * scale / GRID) * GRID),
            max(GRID, round(height * scale / GRID) * GRID))


def latent_tokens(width: int, height: int) -> int:
    """Reference tokens for one encoded frame -- the 16x VAE stride, both axes."""
    return (int(width) // 16) * (int(height) // 16)


def megapixels(width, height) -> float:
    if not width or not height:
        return 0.0
    return (int(width) * int(height)) / 1_000_000


def format_megapixels(width, height) -> str:
    """"0.6 MP" -- a sense of scale rather than a measurement.

    The precision follows the value because the range is wide: 12 MP wants no decimals
    worth of noise, and a thumbnail at the bottom of the slider would otherwise read
    "0.00 MP", which looks like nothing at all rather than like something small.
    """
    value = megapixels(width, height)
    if value <= 0:
        return ""
    if value < 0.1:
        return f"{value:.3f} MP"
    return f"{value:.2f} MP" if value < 1 else f"{value:.1f} MP"


def describe(width, height, percent: int) -> str:
    """What a row's slider currently amounts to, for the readout beside it."""
    if not width or not height:
        return "size not known yet"
    percent = clamp_percent(percent)
    if percent >= MAX_PERCENT:
        return f"{int(width)}x{int(height)}  -  {format_megapixels(width, height)}"
    new_w, new_h = target_size(width, height, percent)
    return (f"{new_w}x{new_h}  -  {format_megapixels(new_w, new_h)}"
            f"  (from {int(width)}x{int(height)})")


# ------------------------------------------------------------------------------ caching

def cache_key(source_digest: str, width: int, height: int, percent: int) -> str:
    """Identify a rescale by its source and the size it was written at.

    The *size* rather than the percentage: two percentages that snap to the same grid
    position produce the same file, and re-encoding it twice would be work for nothing.
    """
    new_w, new_h = target_size(width, height, percent)
    parts = (source_digest, f"{new_w}x{new_h}")
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def cached_path(key: str, source_name: str) -> Path:
    """Where a rescale lives. Named after its source, so a server error still reads.

    The same reasoning as the pose cache: ``main_window`` matches a node error back to a
    row by looking for the row's name in the node label, and an opaque ``scaled_a1b2.png``
    would quietly break that.
    """
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", Path(source_name).stem)[:48].strip("._-")
    return config.SCALE_CACHE_DIR / f"scaled_{stem or 'ref'}_{key[:12]}.png"


def wants_scaling(row) -> bool:
    """Whether this row has a rescale to do at all.

    A target that comes back the same size as the source is not one: the short-edge floor
    can absorb the whole reduction on an already-small image, and writing a copy of it
    would be a cache entry and an upload for nothing.
    """
    if not (getattr(row, "scales", False) and row.local_path and row.width and row.height):
        return False
    return target_size(row.width, row.height, row.scale_percent) != (int(row.width),
                                                                     int(row.height))


def resolve(row) -> tuple[Path, bool]:
    """Where this row's rescaled copy belongs, and whether it is already there."""
    from .comfy_http import sha256_of

    key = cache_key(sha256_of(Path(row.local_path)), row.width, row.height,
                    row.scale_percent)
    path = cached_path(key, Path(row.local_path).name)
    return path, path.is_file()


def swap_in(state, resolved: dict[int, str]) -> None:
    """Point scaled rows at their rescaled copies. Call this on a *copy* of the state.

    The same arrangement the pose pass uses, and for the same reason: the live row goes on
    naming the user's own file, so settings and scenes record the original rather than a
    derivative of it, and the slider stays meaningful next time.
    """
    for row in state.refs.images:
        path = resolved.get(row.uid)
        if not path:
            continue
        row.local_path = path
        # A different file needs its own upload; the old name is the source's.
        row.comfy_name = None


# ------------------------------------------------------------------------------ the pass

def write_scaled(source: str | Path, destination: str | Path,
                 size: tuple[int, int]) -> Path:
    """Write ``source`` at ``size``, atomically.

    The one place this package touches Qt outside ``ui``. Qt is the only image codec the
    app has -- there is no Pillow here, and OpenCV, which arrived with the pose estimator,
    cannot open a path with a non-ASCII character in it on Windows, which is a poor way to
    lose someone's reference. It is already loaded by the time this runs.

    PNG, whatever went in: a rescale is a derivative that then gets encoded by a VAE, and
    putting a second generation of JPEG artefacts in front of that would be a strange
    thing to do to save a few hundred kilobytes.
    """
    from PySide6.QtCore import Qt

    from . import imaging

    source, destination = Path(source), Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)

    # Through `imaging`: PNG carries no EXIF, so writing the stored pixels of a sideways
    # photograph would send it sideways while the same reference unscaled arrives upright.
    image = imaging.load(source)
    if image.isNull():
        raise ScaleError(f"{source.name} could not be read for rescaling")

    width, height = size
    if width <= 0 or height <= 0:
        raise ScaleError(f"{source.name} would rescale to nothing")

    width, height = _agreeing_size(image, (width, height), source.name)
    scaled = image.scaled(width, height, Qt.IgnoreAspectRatio, Qt.SmoothTransformation)

    # .part then rename, as everywhere else here: a half-written PNG at the name the cache
    # looks for would be found and uploaded by the next run.
    partial = destination.with_suffix(".part" + destination.suffix)
    if not scaled.save(str(partial), "PNG"):
        partial.unlink(missing_ok=True)
        raise ScaleError(f"{destination.name} could not be written")
    partial.replace(destination)
    return destination


#: How far the requested shape may differ from the picture's before it is treated as a
#: mistake rather than as rounding. Snapping each axis to the 32-pixel grid moves the
#: aspect by at most about a sixteenth at the smallest size this writes; a tenth is
#: comfortably outside that and nowhere near a transposition, which is a factor of two or
#: more.
_ASPECT_TOLERANCE = 0.10


def _agreeing_size(image, size: tuple[int, int], name: str) -> tuple[int, int]:
    """``size``, unless it describes a differently-shaped picture than the one loaded.

    The sizes here are worked out from what the *row* says a reference measures, and the
    picture is loaded separately. Those two came apart once already -- a probe that could
    not see an EXIF rotation reported a portrait photograph as landscape, and the copy
    written from it was the picture squashed sideways into a landscape box, silently.

    Stretching is the one outcome worth ruling out entirely, so a real disagreement is
    resolved in favour of the pixels: the requested pixel budget is kept and spent on the
    shape the image actually is.
    """
    width, height = size
    if image.width() <= 0 or image.height() <= 0:
        return width, height

    requested = width / height
    actual = image.width() / image.height()
    if abs(requested / actual - 1) <= _ASPECT_TOLERANCE:
        return width, height

    log.warning("%s was to be written %sx%s, which is not its shape (%sx%s) - "
                "keeping the shape", name, width, height, image.width(), image.height())
    scale = ((width * height) / (image.width() * image.height())) ** 0.5
    return (min(max(GRID, round(image.width() * scale / GRID) * GRID), image.width()),
            min(max(GRID, round(image.height() * scale / GRID) * GRID), image.height()))


def render(row) -> Path:
    """Produce this row's rescaled copy if it is not already cached. Returns its path."""
    destination, cached = resolve(row)
    if cached:
        return destination
    size = target_size(row.width, row.height, row.scale_percent)
    log.info("Rescaling %s to %sx%s", Path(row.local_path).name, *size)
    return write_scaled(row.local_path, destination, size)


def cached_copies() -> list[Path]:
    try:
        return sorted(p for p in config.SCALE_CACHE_DIR.glob("scaled_*") if p.is_file())
    except OSError:
        return []
