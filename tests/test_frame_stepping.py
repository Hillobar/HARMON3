"""Stepping frames with the wheel, over the picture or over the track.

The arithmetic is the part worth pinning: a step has to land on the frame you asked for
from where the *player* is, in the frame rate of whatever is on screen, without running
off either end of the clip. A trackpad, which sends a stream of deltas far smaller than a
notch, has to add up to the same thing rather than to nothing at all.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtCore import QPoint, QPointF, Qt            # noqa: E402
from PySide6.QtGui import QWheelEvent                     # noqa: E402
from PySide6.QtWidgets import QApplication                # noqa: E402

from harmon3 import config                                # noqa: E402
from harmon3.refs import VIDEO, RefRow                    # noqa: E402
from harmon3.ui.timeline import (                         # noqa: E402
    COARSE_STEP,
    WHEEL_NOTCH,
    TimelineBar,
    WheelStepper,
)


@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def bar(qapp):
    widget = TimelineBar()
    yield widget
    widget.deleteLater()


class _FakePlayer:
    """Enough of QMediaPlayer for the arithmetic: a position that can be moved."""

    def __init__(self, duration_ms=10_000):
        from PySide6.QtMultimedia import QMediaPlayer

        self._position = 0
        self._duration = duration_ms
        self.state = QMediaPlayer.PausedState
        self.paused = False

    def position(self):
        return self._position

    def setPosition(self, value):
        self._position = int(value)

    def duration(self):
        return self._duration

    def playbackState(self):
        return self.state

    def pause(self):
        from PySide6.QtMultimedia import QMediaPlayer

        self.paused = True
        self.state = QMediaPlayer.PausedState


def _wheel(degrees_eighths: int, shift: bool = False) -> QWheelEvent:
    modifiers = Qt.ShiftModifier if shift else Qt.NoModifier
    return QWheelEvent(QPointF(10, 10), QPointF(10, 10), QPoint(0, 0),
                       QPoint(0, degrees_eighths), Qt.NoButton, modifiers,
                       Qt.NoScrollPhase, False)


def _attach(bar, player=None, duration_ms=10_000):
    """Point a bar at a fake player, the way bind() would with a real one."""
    player = player or _FakePlayer(duration_ms)
    bar._player = player
    bar._duration_ms = duration_ms
    return player


# ------------------------------------------------------------------- the wheel itself

def test_one_notch_is_one_step():
    stepper = WheelStepper()
    assert stepper.frames(_wheel(WHEEL_NOTCH)) == 1


def test_shift_makes_it_ten():
    stepper = WheelStepper()
    assert stepper.frames(_wheel(WHEEL_NOTCH, shift=True)) == COARSE_STEP


def test_scrolling_back_steps_back():
    stepper = WheelStepper()
    assert stepper.frames(_wheel(-WHEEL_NOTCH)) == -1
    assert stepper.frames(_wheel(-WHEEL_NOTCH, shift=True)) == -COARSE_STEP


def test_a_trackpads_small_deltas_add_up_to_a_frame():
    """Rounding each event on its own would make a trackpad do nothing at all."""
    stepper = WheelStepper()
    moved = [stepper.frames(_wheel(WHEEL_NOTCH // 8)) for _ in range(8)]
    assert sum(moved) == 1
    assert moved.count(0) == 7          # seven do nothing, the eighth spends the carry


def test_the_carry_does_not_leak_across_a_reversal():
    """Half a frame forward then half a frame back is where you started, not a step."""
    stepper = WheelStepper()
    stepper.frames(_wheel(WHEEL_NOTCH // 2))
    assert stepper.frames(_wheel(-WHEEL_NOTCH // 2)) == 0


def test_a_wheel_with_no_vertical_delta_is_ignored():
    stepper = WheelStepper()
    event = QWheelEvent(QPointF(10, 10), QPointF(10, 10), QPoint(0, 0), QPoint(0, 0),
                        Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    assert stepper.frames(event) == 0


# ------------------------------------------------------------------------ the stepping

def test_a_step_moves_exactly_one_frame_of_the_result(bar):
    """No reference bound, so the clip is one this app generated at config.FPS."""
    player = _attach(bar)
    bar.step_frames(1)
    assert player.position() == pytest.approx(round(1000 / config.FPS), abs=1)


def test_ten_frames_is_ten_times_one(bar):
    player = _attach(bar)
    bar.step_frames(COARSE_STEP)
    assert player.position() == pytest.approx(round(10 * 1000 / config.FPS), abs=1)


def test_a_bound_reference_steps_in_its_own_frame_rate(bar):
    """A 60 fps reference has shorter frames than the clip being generated."""
    row = RefRow(kind=VIDEO, local_path="D:/refs/clip.mp4")
    row.fps, row.frame_count = 60.0, 600
    bar.row = row
    player = _attach(bar)

    bar.step_frames(1)
    assert player.position() == pytest.approx(round(1000 / 60), abs=1)


def test_stepping_back_from_the_start_stays_at_the_start(bar):
    player = _attach(bar)
    bar.step_frames(-5)
    assert player.position() == 0


def test_stepping_past_the_end_stops_a_hair_short_of_it(bar):
    """Landing exactly on the duration is EndOfMedia, and the frame parks back at zero."""
    player = _attach(bar, duration_ms=1000)
    bar.step_frames(10_000)
    assert player.position() == 999


def test_stepping_stops_playback(bar):
    """Otherwise the next position update overwrites the frame you just asked for."""
    from PySide6.QtMultimedia import QMediaPlayer

    player = _attach(bar)
    player.state = QMediaPlayer.PlayingState
    bar.step_frames(1)
    assert player.paused is True


def test_a_step_of_nothing_does_nothing(bar):
    player = _attach(bar)
    player.setPosition(500)
    bar.step_frames(0)
    assert player.position() == 500


def test_stepping_with_nothing_loaded_is_harmless(bar):
    bar.step_frames(1)          # no player, no duration
    assert bar.duration_ms() == 0


def test_the_track_follows_the_step(bar):
    player = _attach(bar)
    bar.step_frames(3)
    assert bar.track.position() == player.position()


def test_successive_steps_accumulate_rather_than_repeating(bar):
    """Read from the player, not the track: track updates arrive late and would drift."""
    player = _attach(bar)
    for _ in range(4):
        bar.step_frames(1)
    assert player.position() == pytest.approx(round(4 * 1000 / config.FPS), abs=2)


# --------------------------------------------------------------------------- the track

def test_the_wheel_over_the_track_asks_for_a_step(bar):
    _attach(bar)
    bar.track.set_duration(10_000)

    seen = []
    bar.track.stepped.connect(seen.append)
    bar.track.wheelEvent(_wheel(WHEEL_NOTCH))
    assert seen == [1]


def test_the_wheel_over_an_empty_track_does_nothing(bar):
    bar.track.set_duration(0)
    seen = []
    bar.track.stepped.connect(seen.append)
    bar.track.wheelEvent(_wheel(WHEEL_NOTCH))
    assert seen == []


def test_the_arrow_keys_and_the_wheel_move_by_the_same_amount(bar):
    _attach(bar)
    bar.refresh()
    assert bar.track.step_ms == pytest.approx(round(1000 / config.FPS), abs=1)


# -------------------------------------------------------------------------- the picture

@pytest.fixture
def frame(qapp):
    from harmon3.ui.player import VideoPlayer

    widget = VideoPlayer()
    yield widget
    widget.deleteLater()


#: Everything the result frame can have on screen, by the name it is known by.
SURFACES = ("the result video", "the placeholder", "the live preview",
            "a reference clip", "a reference still")


def _surfaces(frame):
    return dict(zip(SURFACES, (
        frame.video_widget,
        frame.placeholder,
        frame.live_preview,
        frame.media_view.panes[0].video_widget,
        frame.media_view.panes[0].image_label,
    )))


@pytest.mark.parametrize("name", SURFACES)
def test_the_wheel_over_any_surface_steps_the_timeline(qapp, frame, name):
    """Delivered by a filter, not by propagation: the wheel does not reliably reach this
    widget from QVideoWidget's native surface, which would leave the gesture working
    everywhere except over the picture it is aimed at."""
    player = _attach(frame.timeline)
    qapp.sendEvent(_surfaces(frame)[name], _wheel(WHEEL_NOTCH))
    assert player.position() == pytest.approx(round(1000 / config.FPS), abs=1)


def test_shift_over_the_picture_steps_ten(qapp, frame):
    player = _attach(frame.timeline)
    player.setPosition(1000)
    qapp.sendEvent(frame.video_widget, _wheel(-WHEEL_NOTCH, shift=True))
    assert player.position() == pytest.approx(1000 - round(10 * 1000 / config.FPS), abs=2)


def test_the_wheel_over_the_volume_slider_is_left_alone(qapp, frame):
    """A slider under the pointer should take the wheel; it is outside what is filtered."""
    player = _attach(frame.timeline)
    qapp.sendEvent(frame.volume_slider, _wheel(WHEEL_NOTCH))
    assert player.position() == 0


def test_the_picture_and_the_track_carry_their_deltas_separately(qapp, frame):
    """Half a frame rolled on one must not finish itself off on the other."""
    player = _attach(frame.timeline)
    frame.timeline.track.set_duration(10_000)
    qapp.sendEvent(frame.video_widget, _wheel(WHEEL_NOTCH // 2))
    frame.timeline.track.wheelEvent(_wheel(WHEEL_NOTCH // 2))
    assert player.position() == 0
