"""The result frame: the transport follows what is on screen, and trimming happens there.

Nothing here decodes anything. The files are empty placeholders with the right suffixes,
because what is under test is which player the controls are pointed at and how a marked
window is converted -- not whether this machine can play an mp4.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6.QtWidgets")

from harmon3.refs import AUDIO, IMAGE, VIDEO, RefRow     # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    import os

    from PySide6.QtWidgets import QApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def files(tmp_path):
    """One of each kind, existing but empty: MediaView only asks whether the file is there."""
    made = {}
    for kind, name in ((IMAGE, "still.png"), (VIDEO, "clip.mp4"), (AUDIO, "roar.wav")):
        path = tmp_path / name
        path.write_bytes(b"")
        made[kind] = str(path)
    return made


@pytest.fixture
def player(qapp):
    from harmon3.ui.player import VideoPlayer

    widget = VideoPlayer()
    yield widget
    widget.clear()
    widget.deleteLater()


def _shown(widget) -> bool:
    """Whether the widget would be on screen. The window itself is never shown here."""
    return not widget.isHidden()


def _item(path, row=None):
    from harmon3.ui.media_view import MediaItem

    return MediaItem(path=path, caption=Path(path).name, row=row)


def _video_row(path, *, fps=24.0, frames=240):
    row = RefRow(kind=VIDEO, local_path=path)
    row.fps, row.frame_count, row.duration_s = fps, frames, frames / fps
    return row


def _audio_row(path, *, duration=10.0):
    row = RefRow(kind=AUDIO, local_path=path)
    row.duration_s = duration
    return row


# ---------------------------------------------------------------------------------
# The stop button
# ---------------------------------------------------------------------------------

def test_nothing_to_stop_before_anything_is_shown(player):
    assert _shown(player.stop_button) is False
    assert player.play_button.isEnabled() is False


def test_stopping_is_offered_once_a_result_is_loaded(player, files):
    player.load(files[VIDEO], autoplay=False)
    assert _shown(player.stop_button) is True
    assert player.play_button.isEnabled() is True


def test_stopping_is_offered_for_a_reference_clip(player, files):
    player.show_media([_item(files[VIDEO])])
    assert _shown(player.stop_button) is True


def test_stopping_is_offered_for_a_reference_that_is_only_sound(player, files):
    """Audio is the case that most needs it: there is nothing to look at, only to silence."""
    player.show_media([_item(files[AUDIO])])
    assert _shown(player.stop_button) is True


def test_a_still_has_nothing_to_stop(player, files):
    player.show_media([_item(files[IMAGE])])
    assert _shown(player.stop_button) is False
    assert player.play_button.isEnabled() is False


def test_stopping_goes_away_again_when_the_frame_is_cleared(player, files):
    player.load(files[VIDEO], autoplay=False)
    player.clear()
    assert _shown(player.stop_button) is False


def test_stop_reaches_every_player_the_frame_owns(player, files):
    """What it does to each of them -- park, not stop -- is in test_player_frames.py."""
    from PySide6.QtMultimedia import QMediaPlayer

    player.show_media([_item(files[AUDIO]), _item(files[VIDEO])])
    player.stop()

    for pane in player.media_view.panes:
        assert pane.player.playbackState() != QMediaPlayer.PlayingState


# ---------------------------------------------------------------------------------
# What the transport is pointed at
# ---------------------------------------------------------------------------------

def test_the_transport_drives_the_result_while_the_result_is_showing(player, files):
    player.load(files[VIDEO], autoplay=False)
    assert player._bound is player.player


def test_the_transport_moves_to_the_reference_being_shown(player, files):
    """Otherwise Play would restart the finished video behind a reference clip."""
    player.load(files[VIDEO], autoplay=False)
    player.show_media([_item(files[AUDIO])])
    assert player._bound is player.media_view.panes[0].player


def test_the_transport_lets_go_of_a_still(player, files):
    player.show_media([_item(files[IMAGE])])
    assert player._bound is None


def test_the_live_preview_has_no_transport(player, files):
    from harmon3.preview import PreviewClip

    player.load(files[VIDEO], autoplay=False)
    player.show_live_preview(PreviewClip())
    assert player._bound is None
    assert _shown(player.stop_button) is False


def test_the_volume_reaches_the_reference_panes(player, files):
    player.show_media([_item(files[AUDIO])])
    player.volume_slider.setValue(20)
    assert player.media_view.panes[0].audio.volume() == pytest.approx(0.2, abs=0.01)


# ---------------------------------------------------------------------------------
# One timeline, which marks only when there is something to mark
# ---------------------------------------------------------------------------------

def test_there_is_exactly_one_timeline(player):
    """It scrubs whatever is on screen and carries the in point when that is a reference.
    A second track showing the same clip was the same information, one line apart."""
    from PySide6.QtWidgets import QSlider

    assert not hasattr(player, "position_slider")
    # The volume slider is the only one left.
    assert [s.toolTip() for s in player.findChildren(QSlider)] == ["Volume"]
    assert _shown(player.timeline) is True


def test_the_timeline_marks_a_clip_that_can_be_trimmed(player, files):
    row = _video_row(files[VIDEO])
    player.show_media([_item(files[VIDEO], row)])
    assert player.timeline.row is row
    assert _shown(player.timeline.mark_in_button) is True


def test_the_marking_controls_stay_hidden_for_a_still(player, files):
    row = RefRow(kind=IMAGE, local_path=files[IMAGE])
    player.show_media([_item(files[IMAGE], row)])
    assert player.timeline.row is None
    assert _shown(player.timeline.mark_in_button) is False


def test_two_clips_side_by_side_have_no_single_thing_to_trim(player, files):
    row = _video_row(files[VIDEO])
    player.show_media([_item(files[VIDEO], row), _item(files[AUDIO], _audio_row(files[AUDIO]))])
    assert player.timeline.row is None
    assert _shown(player.timeline.mark_in_button) is False


def test_the_finished_result_is_scrubbed_but_not_marked(player, files):
    player.show_media([_item(files[VIDEO], _video_row(files[VIDEO]))])
    player.load(files[VIDEO], autoplay=False)
    assert player.timeline.row is None
    assert _shown(player.timeline) is True
    assert _shown(player.timeline.mark_in_button) is False


def test_a_reference_with_no_local_copy_can_still_be_marked(player):
    """Restored from history: no file to mark against, but the number still applies."""
    row = RefRow(kind=AUDIO, comfy_name="harmon3/roar.wav")
    player.edit_trim(row)

    assert player.timeline.row is row
    assert player.timeline.mark_in_button.isEnabled() is False
    assert player.timeline.start_spin.isEnabled() is True


def test_the_timeline_lets_go_when_its_reference_is_removed(player, files):
    row = _video_row(files[VIDEO])
    player.show_media([_item(files[VIDEO], row)])
    player.forget_rows_not_in([])
    assert player.timeline.row is None
    assert _shown(player.timeline.mark_in_button) is False


def test_the_editor_closes_when_its_reference_is_pointed_somewhere_else(player, files, tmp_path):
    row = _video_row(files[VIDEO])
    player.show_media([_item(files[VIDEO], row)])

    other = tmp_path / "other.mp4"
    other.write_bytes(b"")
    row.local_path = str(other)
    player.forget_rows_not_in([row])
    assert player.timeline.row is None


def test_the_editor_closes_when_its_reference_gains_a_local_file(player, files):
    """It was opened with no clip to mark against; now there is one, so reopen properly."""
    row = RefRow(kind=AUDIO, comfy_name="harmon3/roar.wav")
    player.edit_trim(row)

    row.local_path = files[AUDIO]
    player.forget_rows_not_in([row])
    assert player.timeline.row is None


def test_the_editor_survives_an_unrelated_reference_being_removed(player, files):
    row = _video_row(files[VIDEO])
    player.show_media([_item(files[VIDEO], row)])
    player.forget_rows_not_in([row, _audio_row(files[AUDIO])])
    assert player.timeline.row is row


# ---------------------------------------------------------------------------------
# Marking a window
# ---------------------------------------------------------------------------------

@pytest.fixture
def bar(qapp):
    from harmon3.ui.timeline import TimelineBar

    widget = TimelineBar()
    yield widget
    widget.unbind()
    widget.deleteLater()


def test_a_video_mark_is_stored_in_frames(bar, files):
    row = _video_row(files[VIDEO], fps=24.0, frames=240)
    bar.bind(row, None)
    bar._set_start(1000)
    assert row.trim_start == 24.0


def test_an_audio_mark_is_stored_in_seconds(bar, files):
    row = _audio_row(files[AUDIO])
    bar.bind(row, None)
    bar._set_start(1500)
    assert row.trim_start == 1.5


def test_there_is_no_out_point_to_mark(bar):
    """The generated length is the out point, so an editor control for it would be a
    second, disagreeing source of truth."""
    assert not hasattr(bar, "mark_out")
    assert not hasattr(bar, "mark_out_button")
    assert not hasattr(bar, "end_spin")


def test_there_is_nothing_to_switch_trimming_on_with(bar, files):
    """Every reference with a timeline is cut to the generated length, so a toggle could
    only ever be on -- and one that could be off would mean sending something else."""
    assert not hasattr(bar, "enable_box")

    row = _video_row(files[VIDEO])
    bar.bind(row, None)
    assert bar.window_ms()[1] > 0


def test_the_section_runs_from_the_mark_for_the_generated_length(bar, files):
    row = _audio_row(files[AUDIO], duration=30.0)
    bar.bind(row, None)
    bar.set_target_frames(120)                     # five seconds at 24 fps
    bar._set_start(4000)
    assert bar.window_ms() == (4000, 9000)


def test_changing_the_duration_moves_the_far_edge(bar, files):
    row = _audio_row(files[AUDIO], duration=30.0)
    bar.bind(row, None)
    bar._set_start(4000)
    bar.set_target_frames(120)
    assert bar.window_ms() == (4000, 9000)
    bar.set_target_frames(240)
    assert bar.window_ms() == (4000, 14_000)


def test_the_section_stops_where_the_reference_does(bar, files):
    row = _audio_row(files[AUDIO], duration=8.0)
    bar.bind(row, None)
    bar.set_target_frames(240)                     # ten seconds; the file has eight
    bar._set_start(6000)
    assert bar.window_ms() == (6000, 8000)
    assert "short" in bar.describe()


def test_an_unmarked_reference_starts_at_its_own_beginning(bar, files):
    row = _audio_row(files[AUDIO], duration=8.0)
    bar.bind(row, None)
    bar.set_target_frames(240)                     # ten seconds; the file has eight
    assert bar.window_ms() == (0, 8000)


def test_resetting_takes_it_back_to_the_beginning(bar, files):
    row = _video_row(files[VIDEO])
    bar.bind(row, None)
    bar._set_start(1000)
    bar.reset()
    assert row.marked is False
    assert row.trim_start == 0.0


def test_the_frame_rate_can_be_measured_when_it_was_never_probed(bar, files):
    """A file that decodes but was not probed still has a length and a frame count."""
    row = RefRow(kind=VIDEO, local_path=files[VIDEO])
    row.frame_count = 120
    bar.bind(row, None)
    bar._duration_ms = 5000
    assert bar._fps() == pytest.approx(24.0)
    assert bar._usable() is True


def test_a_video_of_unknown_rate_is_not_guessed_at(bar, files):
    row = RefRow(kind=VIDEO, local_path=files[VIDEO])
    bar.bind(row, None)
    assert bar._usable() is False
    assert "frame rate" in bar.describe()


def test_the_summary_says_what_will_be_sent(bar, files):
    row = _video_row(files[VIDEO], fps=24.0, frames=240)
    bar.bind(row, None)
    bar.set_target_frames(48)
    bar._set_start(1000)
    assert "48 frames" in bar.describe()
    assert "2.00s" in bar.describe()


# ---------------------------------------------------------------------------------
# The way in, from the reference row
# ---------------------------------------------------------------------------------

def test_a_clip_row_opens_in_the_result_frame_when_clicked(qapp, files):
    """The way in is the row itself. There is no Trim button any more: with nothing to
    switch on, it was a second button that did what clicking the row already does."""
    from harmon3.ui.ref_panel import RefRowWidget

    widget = RefRowWidget(_video_row(files[VIDEO]))
    assert not hasattr(widget, "trim_button")

    asked = []
    widget.preview_requested.connect(lambda row, which, additive: asked.append(which))
    widget.thumb.clicked.emit("source", False)

    assert asked == ["source"]
    widget.deleteLater()


def test_the_row_no_longer_carries_the_controls_itself(qapp, files):
    """They moved to the result frame; a stray copy here would be a second source of truth."""
    from PySide6.QtWidgets import QDoubleSpinBox
    from harmon3.ui.ref_panel import RefRowWidget

    widget = RefRowWidget(_video_row(files[VIDEO]))
    assert widget.findChildren(QDoubleSpinBox) == []
    widget.deleteLater()


def test_the_row_still_shows_the_section_that_is_set(qapp, files):
    from harmon3.ui.ref_panel import RefRowWidget

    row = _video_row(files[VIDEO], fps=24.0, frames=600)
    row.trim_start = 24
    widget = RefRowWidget(row)
    widget.refresh_details(target_frames=240)

    assert "from 24-264f" in widget.detail_label.text()
    widget.deleteLater()


def test_an_unmarked_row_says_nothing_about_where_it_starts(qapp, files):
    from harmon3.ui.ref_panel import RefRowWidget

    widget = RefRowWidget(_video_row(files[VIDEO], fps=24.0, frames=600))
    widget.refresh_details(target_frames=240)

    assert "from" not in widget.detail_label.text()
    widget.deleteLater()


# ---------------------------------------------------------------------------------
# The timeline itself
# ---------------------------------------------------------------------------------

@pytest.fixture
def timeline(qapp):
    from harmon3.ui.timeline import TimelineTrack

    widget = TimelineTrack()
    widget.resize(400, 44)
    widget.set_duration(10_000)
    widget.set_window(2000, 8000)
    widget.set_marking(True)
    yield widget
    widget.deleteLater()


def test_a_position_maps_to_a_pixel_and_back(timeline):
    for milliseconds in (0, 2500, 5000, 10_000):
        assert timeline._ms_at(timeline._x_for(milliseconds)) == pytest.approx(
            milliseconds, abs=30)


def test_the_ends_of_the_track_are_the_ends_of_the_clip(timeline):
    assert timeline._x_for(0) < timeline._x_for(10_000)
    assert timeline._ms_at(-500) == 0
    assert timeline._ms_at(10_000) == 10_000


def test_grabbing_near_the_handle_takes_the_handle(timeline):
    assert timeline._hit(timeline._x_for(2000)) == "in"


def test_the_far_edge_is_not_something_to_grab(timeline):
    """It is where the section runs out, which the duration parameter decides."""
    assert timeline._hit(timeline._x_for(8000)) == "playhead"


def test_grabbing_anywhere_else_scrubs(timeline):
    assert timeline._hit(timeline._x_for(5000)) == "playhead"


def test_the_handle_can_be_dragged_anywhere_on_the_track(timeline):
    """Nothing to collide with any more: a mark past the old out point simply moves the
    section, rather than being blocked by an edge that no longer means anything."""
    timeline._drag = "in"
    timeline._apply_drag(timeline._x_for(9500))
    assert timeline._in == pytest.approx(9500, abs=40)


def test_an_empty_clip_offers_nothing_to_grab(timeline):
    timeline.set_duration(0)
    assert timeline._hit(50) == ""


def test_a_track_that_is_not_marking_has_no_handle_to_grab(timeline):
    """The same track scrubs a finished video, which has nothing to mark: a handle there
    would offer an edit with nowhere to go."""
    timeline.set_marking(False)
    assert timeline._hit(timeline._x_for(2000)) == "playhead"


def test_dragging_a_handle_marks_the_reference_and_says_so(qapp, files):
    """The whole chain: a mouse on the track, a window on the row, and the panel told."""
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtTest import QTest

    from harmon3.ui.timeline import TimelineBar

    row = _video_row(files[VIDEO], fps=24.0, frames=240)      # ten seconds
    bar = TimelineBar()
    bar.resize(500, 100)
    bar.bind(row, None)
    announced = []
    bar.changed.connect(lambda: announced.append(row.trim_start))

    track = bar.track
    track.resize(500, 44)
    start = QPoint(track._x_for(0), track.height() - 12)
    QTest.mousePress(track, Qt.LeftButton, Qt.NoModifier, start)
    track.mouseMoveEvent(_move_event(track, track._x_for(2000), start.y()))
    QTest.mouseRelease(track, Qt.LeftButton, Qt.NoModifier,
                       QPoint(track._x_for(2000), start.y()))

    assert row.trim_start == pytest.approx(48, abs=2)         # two seconds at 24 fps
    assert announced, "the rest of the window was never told the edit had finished"
    bar.unbind()
    bar.deleteLater()


def _move_event(widget, x, y):
    """QTest.mouseMove does not deliver a move with a button held; this one does."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QMouseEvent

    return QMouseEvent(
        QMouseEvent.MouseMove, QPointF(x, y),
        widget.mapToGlobal(QPoint(x, y)),
        Qt.NoButton, Qt.LeftButton, Qt.NoModifier)
