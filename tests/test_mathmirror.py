"""Parity tests for the client-side mirrors of ComfyUI's resolution and frame math."""

from __future__ import annotations

import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmon3 import config                                    # noqa: E402
from harmon3.mathmirror import (                              # noqa: E402
    ASPECT_RATIOS,
    clamp_duration,
    duration_error,
    frames_from_seconds,
    resolution,
    resolution_error,
    true_seconds,
)


def test_shipped_workflow_resolution():
    """The workflow's own settings must reproduce what ComfyUI computes for them."""
    assert resolution("16:9 (Widescreen)", 0.4, 32) == (864, 480)


def test_shipped_workflow_frames():
    assert frames_from_seconds(5.0) == 124
    assert math.isclose(true_seconds(124), 124 / 24)


def test_known_frame_counts():
    assert frames_from_seconds(0.2) == 5
    assert frames_from_seconds(10.0) == 243
    # Anything at or below the 5-frame floor clamps up, never down.
    assert frames_from_seconds(0.0) == 5
    assert frames_from_seconds(-3.0) == 5


def test_frame_grid_invariants():
    """Every reachable frame count sits on the 17k+5 grid and inside the node's range."""
    seconds = config.MIN_DURATION
    while seconds <= config.MAX_DURATION:
        frames = frames_from_seconds(seconds)
        assert frames % config.FRAME_MOD == config.FRAME_REM, seconds
        assert config.MIN_FRAMES <= frames <= config.MAX_FRAMES, seconds
        seconds = round(seconds + 0.05, 4)


def test_max_duration_is_the_true_ceiling():
    """MAX_DURATION must be the largest duration that still fits length's max of 3600."""
    assert frames_from_seconds(config.MAX_DURATION) == config.MAX_ALIGNED_FRAMES == 3592
    assert frames_from_seconds(config.MAX_DURATION + 0.05) > config.MAX_FRAMES


def test_clamp_duration():
    assert clamp_duration(1e9) == config.MAX_DURATION
    assert clamp_duration(0.0) == config.MIN_DURATION
    assert clamp_duration(5.0) == 5.0
    assert clamp_duration(float("nan")) == config.DEFAULT_DURATION


def test_duration_error():
    assert duration_error(5.0) is None
    assert duration_error(config.MAX_DURATION) is None
    assert duration_error(200.0) is not None


def test_resolution_sweep_invariants():
    """Sweep the full parameter space; flag any combo the model would reject."""
    megapixels = config.MIN_MEGAPIXELS
    while megapixels <= config.MAX_MEGAPIXELS + 1e-9:
        mp = round(megapixels, 1)
        for multiple in range(config.MIN_MULTIPLE, config.MAX_MULTIPLE + 1, config.STEP_MULTIPLE):
            for aspect in ASPECT_RATIOS:
                width, height = resolution(aspect, mp, multiple)
                assert width % multiple == 0, (aspect, mp, multiple)
                assert height % multiple == 0, (aspect, mp, multiple)
                # A too-small result is legal arithmetic but illegal input; the UI must
                # surface it rather than let it 400 at submit.
                err = resolution_error(width, height)
                if width < config.MIN_DIMENSION or height < config.MIN_DIMENSION:
                    assert err is not None
                else:
                    assert err is None
        megapixels += config.STEP_MEGAPIXELS


def test_resolution_error_thresholds():
    """Guard rails for node 136's width/height range, independent of the sweep."""
    assert resolution_error(864, 480) is None
    assert resolution_error(config.MIN_DIMENSION, config.MIN_DIMENSION) is None
    assert "below the model" in resolution_error(0, 480)
    assert "below the model" in resolution_error(864, 16)
    assert "exceeds" in resolution_error(config.MAX_DIMENSION + 32, 480)


def test_resolution_rejects_unknown_aspect():
    try:
        resolution("5:4 (Nope)", 1.0, 32)
    except ValueError as exc:
        assert "5:4 (Nope)" in str(exc)
    else:
        raise AssertionError("expected ValueError")


def test_aspect_ratio_strings_are_exact():
    """These strings are sent verbatim to a COMBO input; typos become 400s."""
    assert list(ASPECT_RATIOS) == [
        "1:1 (Square)",
        "2:3 (Portrait Photo)",
        "3:2 (Photo)",
        "3:4 (Portrait Standard)",
        "4:3 (Standard)",
        "9:16 (Portrait Widescreen)",
        "16:9 (Widescreen)",
        "21:9 (Ultrawide)",
    ]
