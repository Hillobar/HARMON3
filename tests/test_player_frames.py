"""The result frame keeps a picture up when it is not playing.

Two halves. The first drives a real QMediaPlayer against a real clip, because the
behaviour being worked around is the media backend's own: stopping a player tears its
decode pipeline down and the video surface goes black, and a seek while stopped moves the
position without presenting anything. Those tests are what pin ``park()`` in place --
"simplifying" it back to ``QMediaPlayer.stop()`` breaks them.

The second half checks the widget routes through it, using undecodable placeholder files:
QVideoWidget cannot decode real media in a headless test run, so anything needing both a
real clip and the widget is not testable here and is not pretended otherwise.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6.QtWidgets")

from harmon3.refs import AUDIO, VIDEO, RefRow           # noqa: E402
from harmon3.ui import player as player_mod             # noqa: E402
from harmon3.ui.player import park                      # noqa: E402

CLIP_FRAMES = 24
CLIP_FPS = 24


@pytest.fixture(scope="module")
def qapp():
    import os

    from PySide6.QtWidgets import QApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app


def _spin(milliseconds: int) -> None:
    from PySide6.QtCore import QEventLoop, QTimer

    loop = QEventLoop()
    QTimer.singleShot(milliseconds, loop.quit)
    loop.exec()


def _wait(predicate, timeout: float = 6.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        _spin(30)
    return predicate()


@pytest.fixture(scope="module")
def clip(tmp_path_factory):
    """A genuinely decodable one-second clip.

    An empty placeholder cannot show any of this: a file the platform never decodes has no
    frame to lose in the first place.
    """
    av = pytest.importorskip("av", reason="PyAV is needed to author a real test clip")
    path = tmp_path_factory.mktemp("media") / "clip.mp4"

    container = av.open(str(path), "w")
    stream = container.add_stream("libx264", rate=CLIP_FPS)
    stream.width, stream.height, stream.pix_fmt = 160, 96, "yuv420p"
    for index in range(CLIP_FRAMES):
        frame = av.VideoFrame(160, 96, "rgb24")
        plane = frame.planes[0]
        # A different flat colour per frame, so a frame is unmistakably *a* frame.
        plane.update(bytes([(index * 9) % 256]) * plane.buffer_size)
        for packet in stream.encode(frame):
            container.mux(packet)
    for packet in stream.encode():
        container.mux(packet)
    container.close()
    return str(path)


@pytest.fixture
def playing(qapp, clip):
    """A real player that has decoded at least one frame, plus its sink."""
    from PySide6.QtCore import QUrl
    from PySide6.QtMultimedia import QMediaPlayer, QVideoSink

    sink = QVideoSink()
    player = QMediaPlayer()
    player.setVideoSink(sink)
    player.setSource(QUrl.fromLocalFile(clip))
    _wait(lambda: player.duration() > 0)
    player.play()

    if not _wait(lambda: sink.videoFrame().isValid()):
        pytest.skip("this platform cannot decode the test clip")

    yield player, sink
    player.setSource(QUrl())


# ---------------------------------------------------------------------------------
# What the media backend actually does, and why park() exists
# ---------------------------------------------------------------------------------

def test_stopping_a_player_loses_its_picture(playing):
    """The bug, stated as a fact about Qt: this is why nothing here calls stop()."""
    player, sink = playing
    player.stop()
    _spin(300)
    assert sink.videoFrame().isValid() is False


def test_seeking_a_stopped_player_shows_nothing(playing):
    """And this is why a scrub after a stop used to leave the frame black."""
    player, sink = playing
    player.stop()
    _spin(200)
    player.setPosition(600)
    _spin(300)

    assert player.position() == 600      # the position moves...
    assert sink.videoFrame().isValid() is False   # ...but no frame is presented


def test_parking_keeps_the_picture(playing):
    player, sink = playing
    park(player, 600)

    assert _wait(lambda: sink.videoFrame().isValid()) is True
    assert player.position() == 600


def test_parking_rewinds_without_going_black(playing):
    player, sink = playing
    park(player)

    assert _wait(lambda: sink.videoFrame().isValid()) is True
    assert player.position() == 0


def test_a_parked_player_is_paused_rather_than_stopped(playing):
    """The distinction is the whole point: a paused player still has a pipeline."""
    from PySide6.QtMultimedia import QMediaPlayer

    player, _sink = playing
    park(player)
    assert player.playbackState() == QMediaPlayer.PausedState


def test_a_parked_player_can_still_be_scrubbed(playing):
    player, sink = playing
    park(player)
    _wait(lambda: sink.videoFrame().isValid())

    player.setPosition(800)
    assert _wait(lambda: player.position() == 800) is True
    assert sink.videoFrame().isValid() is True


def test_parking_a_player_with_nothing_loaded_is_harmless(qapp):
    from PySide6.QtMultimedia import QMediaPlayer

    player = QMediaPlayer()
    park(player, 500)
    assert player.playbackState() == QMediaPlayer.StoppedState


# ---------------------------------------------------------------------------------
# The widget routes through it
# ---------------------------------------------------------------------------------

@pytest.fixture
def files(tmp_path):
    made = {}
    for kind, name in ((VIDEO, "clip.mp4"), (AUDIO, "roar.wav")):
        path = tmp_path / name
        path.write_bytes(b"")
        made[kind] = str(path)
    return made


@pytest.fixture
def widget(qapp):
    from harmon3.ui.player import VideoPlayer

    made = VideoPlayer()
    yield made
    made.clear()
    made.deleteLater()


@pytest.fixture
def parked(monkeypatch):
    """Records what the widget parks, so the routing can be checked without decoding."""
    calls = []
    monkeypatch.setattr(player_mod, "park",
                        lambda player, position=0: calls.append((player, position)))
    return calls


def _item(path, row=None):
    from harmon3.ui.media_view import MediaItem

    return MediaItem(path=path, caption=Path(path).name, row=row)


def test_stop_parks_the_result_rather_than_stopping_it(widget, files, parked):
    widget.load(files[VIDEO], autoplay=False)
    parked.clear()
    widget.stop()

    assert parked == [(widget.player, 0)]


def test_stop_parks_the_reference_panes(widget, files, parked):
    widget.show_media([_item(files[VIDEO])])
    parked.clear()
    widget.stop()

    assert [player for player, _ in parked] == [pane.player for pane in widget.media_view.panes]
    assert {position for _, position in parked} == {0}


def test_stop_leaves_the_hidden_result_alone(widget, files, parked):
    """A reference is on screen; the finished video behind it is not the thing being stopped."""
    widget.load(files[VIDEO], autoplay=False)
    widget.show_media([_item(files[AUDIO])])
    parked.clear()
    widget.stop()

    assert widget.player not in [player for player, _ in parked]


def test_stop_rewinds_to_the_in_point_of_a_marked_reference(widget, files, parked):
    """Playback is confined to the section, so its start is where Stop goes."""
    row = RefRow(kind=VIDEO, local_path=files[VIDEO])
    row.fps, row.frame_count, row.duration_s = 24.0, 240, 10.0
    row.trim_start = 48                                                # two seconds in
    widget.show_media([_item(files[VIDEO], row)])
    parked.clear()
    widget.stop()

    bound = widget.timeline.player
    assert dict((player, position) for player, position in parked)[bound] == 2000


def test_an_untrimmed_reference_rewinds_to_the_start(widget, files, parked):
    row = RefRow(kind=VIDEO, local_path=files[VIDEO])
    row.fps, row.frame_count, row.duration_s = 24.0, 240, 10.0
    widget.show_media([_item(files[VIDEO], row)])
    parked.clear()
    widget.stop()

    assert {position for _, position in parked} == {0}


def test_stop_does_nothing_when_there_is_nothing_on_screen(widget, parked):
    widget.stop()
    assert parked == []


def test_reaching_the_end_of_a_clip_parks_it(widget, files, parked):
    """Deferred out of the status signal, so the check is on what the deferred call does."""
    widget.load(files[VIDEO], autoplay=False)
    parked.clear()
    widget._park_result()

    assert parked == [(widget.player, 0)]


def test_parking_is_never_done_from_inside_the_status_signal(widget, files, parked):
    """Seeking a player from inside its own mediaStatusChanged took the process down."""
    from PySide6.QtMultimedia import QMediaPlayer

    widget.load(files[VIDEO], autoplay=False)
    parked.clear()
    widget._on_media_status(QMediaPlayer.EndOfMedia)
    widget._on_media_status(QMediaPlayer.LoadedMedia)

    assert parked == []          # nothing happens until the event loop comes back round


def test_dragging_the_timeline_seeks_while_the_mouse_is_down(widget, files, monkeypatch):
    """Otherwise the picture only catches up once the mouse is released."""
    widget.show_media([_item(files[VIDEO])])
    seeks = []
    monkeypatch.setattr(widget._bound, "setPosition", seeks.append)

    widget.timeline.track.scrubbed.emit(1500)
    widget.timeline.track.scrubbed.emit(2500)

    assert seeks == [1500, 2500]


def test_the_players_own_position_updates_do_not_feed_back_into_a_seek(widget, files,
                                                                      monkeypatch):
    """The track follows the player through set_position, which announces nothing -- only
    a mouse on the track emits ``scrubbed``, so a seek cannot chase its own updates."""
    widget.show_media([_item(files[VIDEO])])
    seeks = []
    monkeypatch.setattr(widget._bound, "setPosition", seeks.append)

    widget.timeline._on_player_position(3000)     # exactly what the player's signal does
    assert seeks == []
