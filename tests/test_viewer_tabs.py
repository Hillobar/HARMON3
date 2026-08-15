"""The viewer's three tabs: what raises them, and what they keep when they are not in front.

The frame used to swap one surface between a reference, the sampler's preview and the
finished video, so opening a reference threw the result away. On tabs all three stay. What
is worth pinning is the part that is not obvious from the widget tree: which arrivals raise
a tab and which do not, that the transport follows the tab in front, and that leaving a tab
parks what is on it rather than stopping it.

Nothing here decodes anything -- the files are empty placeholders with the right suffixes.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6.QtWidgets")

from harmon3.refs import AUDIO, IMAGE, VIDEO, RefRow                 # noqa: E402
from harmon3.ui.player import TAB_LATENT, TAB_REFERENCE, TAB_RESULTS  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    import os

    from PySide6.QtWidgets import QApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def files(tmp_path):
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


def _item(path, row=None):
    from harmon3.ui.media_view import MediaItem

    return MediaItem(path=path, caption=Path(path).name, row=row)


def _video_row(path, *, fps=24.0, frames=240):
    row = RefRow(kind=VIDEO, local_path=path)
    row.fps, row.frame_count, row.duration_s = fps, frames, frames / fps
    return row


def _clip(step: int = 1):
    from PySide6.QtGui import QImage

    from harmon3.preview import PreviewClip

    image = QImage(8, 8, QImage.Format_RGB32)
    image.fill(0)
    return PreviewClip(frames=[image], fps=8.0, step=step, total=20, sigma=0.5)


# ---------------------------------------------------------------------------------
# Three tabs, and the lines that say what an empty one is for
# ---------------------------------------------------------------------------------

def test_the_frame_has_three_tabs(player):
    labels = [player.tabs.tabText(i) for i in range(player.tabs.count())]
    assert labels == ["Reference", "Latent", "Results"]


def test_it_opens_on_results(player):
    """Nothing has happened yet, and "queue a run" is the thing to say first."""
    assert player.tabs.currentIndex() == TAB_RESULTS
    assert player.results_page.currentWidget() is player.placeholder


def test_each_empty_tab_explains_itself(player):
    """One line about references and one about results; each on the tab it is about."""
    assert "reference" in player.ref_hint.text().lower()
    assert "queue" in player.placeholder.text().lower()


# ---------------------------------------------------------------------------------
# What raises a tab
# ---------------------------------------------------------------------------------

def test_opening_a_reference_raises_its_tab(player, files):
    player.load(files[VIDEO], autoplay=False)
    player.show_media([_item(files[IMAGE])])

    assert player.tabs.currentIndex() == TAB_REFERENCE
    assert len(player.shown_media()) == 1


def test_a_finished_video_raises_the_results_tab(player, files):
    player.show_media([_item(files[IMAGE])])
    player.load(files[VIDEO], autoplay=False)

    assert player.tabs.currentIndex() == TAB_RESULTS
    assert player.results_page.currentWidget() is player.video_widget


def test_a_result_does_not_throw_the_references_away(player, files):
    """The whole point of the tabs: the two no longer share one surface."""
    player.show_media([_item(files[IMAGE])])
    player.load(files[VIDEO], autoplay=False)

    assert player.media_view.items()


def test_shown_media_is_empty_while_another_tab_is_in_front(player, files):
    """Shift-click builds on what is on screen, and off the Reference tab that is nothing."""
    player.show_media([_item(files[IMAGE])])
    player.tabs.setCurrentIndex(TAB_RESULTS)

    assert player.shown_media() == []


def test_the_first_preview_of_a_run_raises_the_latent_tab(player):
    player.show_live_preview(_clip())
    assert player.tabs.currentIndex() == TAB_LATENT


def test_later_previews_leave_the_choice_alone(player):
    """They arrive many times a second; raising on each would make the frame unusable."""
    player.show_live_preview(_clip(1))
    player.tabs.setCurrentIndex(TAB_RESULTS)

    for step in range(2, 8):
        player.show_live_preview(_clip(step))

    assert player.tabs.currentIndex() == TAB_RESULTS


def test_the_next_run_claims_the_frame_again(player):
    """Once per run, not once ever: the end of a run puts the claim back."""
    player.show_live_preview(_clip(1))
    player.end_live_preview()
    player.tabs.setCurrentIndex(TAB_RESULTS)

    player.show_live_preview(_clip(1))
    assert player.tabs.currentIndex() == TAB_LATENT


def test_clearing_goes_back_to_the_empty_result(player, files):
    player.show_media([_item(files[IMAGE])])
    player.load(files[VIDEO], autoplay=False)

    player.clear()

    assert player.tabs.currentIndex() == TAB_RESULTS
    assert player.results_page.currentWidget() is player.placeholder
    assert player.shown_media() == []


# ---------------------------------------------------------------------------------
# The transport follows the tab in front
# ---------------------------------------------------------------------------------

def test_the_transport_follows_a_tab_switch(player, files):
    player.load(files[VIDEO], autoplay=False)
    player.show_media([_item(files[VIDEO], _video_row(files[VIDEO]))])
    assert player._bound is player.media_view.panes[0].player

    player.tabs.setCurrentIndex(TAB_RESULTS)
    assert player._bound is player.player


def test_the_latent_tab_drives_nothing(player):
    """It is pixmaps on a timer; there is no player behind it to Play or Stop."""
    player.show_live_preview(_clip())

    assert player._bound is None
    assert player.play_button.isEnabled() is False


def test_leaving_a_tab_parks_it_rather_than_stopping_it(player, files, monkeypatch):
    """A stop tears the decode pipeline down and the surface goes black with it."""
    from harmon3.ui import player as player_mod

    player.load(files[VIDEO], autoplay=False)
    player.show_media([_item(files[VIDEO], _video_row(files[VIDEO]))])

    stopped = []
    monkeypatch.setattr(player_mod.QMediaPlayer, "stop", lambda self: stopped.append(self))

    player.tabs.setCurrentIndex(TAB_RESULTS)

    assert stopped == []
    assert player.media_view.items()


# ---------------------------------------------------------------------------------
# The wheel
# ---------------------------------------------------------------------------------

def test_the_wheel_over_the_tab_bar_does_nothing(player, files):
    """QTabBar rolls through its tabs; the frame steps frames. Neither, over the bar."""
    from PySide6.QtCore import QPoint, QPointF, Qt
    from PySide6.QtGui import QWheelEvent
    from PySide6.QtWidgets import QApplication

    player.show_media([_item(files[VIDEO], _video_row(files[VIDEO]))])
    stepped = []
    player.timeline.step_frames = lambda frames: stepped.append(frames)

    bar = player.tabs.tabBar()
    event = QWheelEvent(
        QPointF(10, 5), QPointF(10, 5), QPoint(0, 0), QPoint(0, 120),
        Qt.NoButton, Qt.NoModifier, Qt.NoScrollPhase, False)
    QApplication.sendEvent(bar, event)

    assert stepped == []
    assert player.tabs.currentIndex() == TAB_REFERENCE
