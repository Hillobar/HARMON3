"""The generation geometry: aspect ratio and duration in, width/height/length out.

    (aspect_ratio, megapixels) -> (width, height), snapped to a multiple of 32
    seconds                    -> frame count on the 17k+5 grid the model requires

These began as client-side mirrors of two ComfyUI nodes -- ``ResolutionSelector`` and a
``ComfyMathExpression`` -- so the GUI could show the numbers live. The workflow no longer
carries either, so this module is now *authoritative*: the builder writes its results into
the reference node as literals, and there is nothing left on the server to disagree with.

Both use Python's builtin ``round`` (banker's rounding), which is what ResolutionSelector
called and what the expression evaluator resolved ``round`` to. ``int(x + 0.5)`` would
disagree on exact .5 values and change existing users' resolutions. Do not "simplify".

No Qt, no network.
"""

from __future__ import annotations

import math

from . import config

#: Exact combo strings accepted by ResolutionSelector.aspect_ratio, mapped to (w, h).
#: Order matters: it is the order the combo box presents.
ASPECT_RATIOS: dict[str, tuple[int, int]] = {
    "1:1 (Square)": (1, 1),
    "2:3 (Portrait Photo)": (2, 3),
    "3:2 (Photo)": (3, 2),
    "3:4 (Portrait Standard)": (3, 4),
    "4:3 (Standard)": (4, 3),
    "9:16 (Portrait Widescreen)": (9, 16),
    "16:9 (Widescreen)": (16, 9),
    "21:9 (Ultrawide)": (21, 9),
}


def resolution(aspect_ratio: str, megapixels: float,
               multiple: int = config.MULTIPLE) -> tuple[int, int]:
    """Return the (width, height) the reference node is given.

    ``multiple`` is fixed at 32 throughout the app and is only a parameter so this can be
    tested across the whole 8..128 range rather than the single value the app sends.
    """
    try:
        w_ratio, h_ratio = ASPECT_RATIOS[aspect_ratio]
    except KeyError:
        raise ValueError(f"Unknown aspect ratio {aspect_ratio!r}") from None

    scale = math.sqrt(megapixels * 1024 * 1024 / (w_ratio * h_ratio))
    width = round(w_ratio * scale / multiple) * multiple
    height = round(h_ratio * scale / multiple) * multiple
    return width, height


def frames_from_seconds(seconds: float) -> int:
    """Return the frame count a duration in seconds becomes.

    Rounds to 24 fps then up to the next value satisfying ``n % 17 == 5``, which is the
    grid MiniMaxH3ReferenceToVideo.length requires. Sending a length off that grid is
    rejected by the node, so this quantisation is not cosmetic.
    """
    n = max(config.FRAME_REM, round(seconds * config.FPS))
    return n + (config.FRAME_REM - n % config.FRAME_MOD) % config.FRAME_MOD


def nearest_aspect_ratio(width: int, height: int) -> str:
    """The offered ratio closest to ``width:height``, for seeding state from a workflow.

    Approximate by nature: the workflow holds literal dimensions, which need not sit on
    any offered ratio at all. Compared in log space so 16:9 and 9:16 are equally far from
    a square, rather than the wide one always winning.
    """
    if not width or not height:
        return config.DEFAULT_ASPECT_RATIO
    target = math.log(width / height)
    return min(ASPECT_RATIOS,
               key=lambda name: abs(math.log(_ratio(ASPECT_RATIOS[name])) - target))


def megapixels_for(width: int, height: int) -> float:
    """The megapixel figure that reproduces roughly ``width x height``, on the UI's step."""
    if not width or not height:
        return config.DEFAULT_MEGAPIXELS
    raw = width * height / (1024 * 1024)
    stepped = round(raw / config.STEP_MEGAPIXELS) * config.STEP_MEGAPIXELS
    return round(min(max(stepped, config.MIN_MEGAPIXELS), config.MAX_MEGAPIXELS), 1)


def _ratio(pair: tuple[int, int]) -> float:
    return pair[0] / pair[1]


def true_seconds(frames: int) -> float:
    """Actual clip length once the frame count has been snapped to the 17k+5 grid."""
    return frames / config.FPS


def clamp_duration(seconds: float) -> float:
    """Clamp a duration so the resulting frame count stays within the node's 5..3600.

    The widget also enforces this, but the builder re-asserts it: nothing else catches an
    over-long duration (PrimitiveFloat's range is effectively unbounded), so it would
    surface as a 400 from /prompt.
    """
    if seconds != seconds:  # NaN
        return config.DEFAULT_DURATION
    return min(max(seconds, config.MIN_DURATION), config.MAX_DURATION)


def duration_error(seconds: float) -> str | None:
    """Human-readable reason a duration is unusable, or None if it is fine."""
    frames = frames_from_seconds(seconds)
    if frames < config.MIN_FRAMES:
        return f"Duration too short: {frames} frames (minimum {config.MIN_FRAMES})"
    if frames > config.MAX_FRAMES:
        return (
            f"Duration too long: {frames} frames exceeds the model's {config.MAX_FRAMES} "
            f"(max {config.MAX_DURATION:.1f} s)"
        )
    return None


def resolution_error(width: int, height: int) -> str | None:
    """Human-readable reason a computed resolution is unusable, or None if it is fine."""
    if width < config.MIN_DIMENSION or height < config.MIN_DIMENSION:
        return (
            f"Resolution {width}x{height} is below the model's {config.MIN_DIMENSION} px "
            "minimum - raise megapixels"
        )
    if width > config.MAX_DIMENSION or height > config.MAX_DIMENSION:
        return f"Resolution {width}x{height} exceeds the model's {config.MAX_DIMENSION} px maximum"
    return None
