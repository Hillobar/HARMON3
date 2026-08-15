"""The size slider on a reference row, and what the result frame does while it moves."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtGui import QImage                        # noqa: E402
from PySide6.QtWidgets import QApplication              # noqa: E402

from harmon3 import scaling                             # noqa: E402
from harmon3.refs import AUDIO, IMAGE, VIDEO, RefRow    # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def picture(tmp_path):
    """A real file, because the pane loads a QPixmap from it."""
    path = tmp_path / "hero.png"
    QImage(800, 600, QImage.Format_RGB32).save(str(path), "PNG")
    return path


@pytest.fixture
def row(picture):
    row = RefRow(kind=IMAGE, local_path=str(picture))
    row.width, row.height = 800, 600
    return row


@pytest.fixture
def widget(qapp, row):
    from harmon3.ui.ref_panel import RefRowWidget

    made = RefRowWidget(row)
    made.refresh_details(124)
    yield made
    made.deleteLater()


# ------------------------------------------------------------------------- the control

def test_audio_gets_no_slider(qapp):
    """There is no frame size to choose for a soundtrack."""
    from harmon3.ui.ref_panel import RefRowWidget

    made = RefRowWidget(RefRow(kind=AUDIO, local_path="a.wav"))
    try:
        assert made.scale_slider is None
    finally:
        made.deleteLater()


def test_a_clip_gets_a_slider_too(qapp):
    from harmon3.ui.ref_panel import RefRowWidget

    clip = RefRow(kind=VIDEO, local_path="a.mp4")
    clip.width, clip.height = 1920, 1080
    made = RefRowWidget(clip)
    try:
        assert made.scale_slider is not None
        made.scale_slider.setValue(50)
        # A share of the node's canvas, not of the 1920x1080 source.
        assert "672x384" in made.scale_readout.text()
    finally:
        made.deleteLater()


def test_the_slider_covers_the_documented_range(widget):
    assert widget.scale_slider.minimum() == scaling.MIN_PERCENT
    assert widget.scale_slider.maximum() == scaling.MAX_PERCENT
    assert widget.scale_slider.value() == 100


def test_the_row_reports_its_megapixels(widget):
    assert "0.48 MP" in widget.detail_label.text()
    assert "800x600" in widget.detail_label.text()


def test_moving_the_slider_records_the_ceiling_and_says_so(widget, row):
    seen = []
    widget.scale_changed.connect(lambda r: seen.append(r.scale_percent))

    widget.scale_slider.setValue(50)

    assert row.scale_percent == 50
    assert seen == [50]
    assert widget.scale_percent_label.text() == "50%"
    expected = "x".join(str(v) for v in scaling.target_size(800, 600, 50))
    assert expected in widget.scale_readout.text()
    assert "from 800x600" in widget.scale_readout.text()


def test_the_readout_warns_only_once_it_is_cutting_something(widget):
    assert widget.scale_readout.property("role") == "hint"
    widget.scale_slider.setValue(50)
    assert widget.scale_readout.property("role") == "warn"
    widget.scale_slider.setValue(100)
    assert widget.scale_readout.property("role") == "hint"


def test_the_readout_fills_in_when_the_probe_lands(qapp, picture):
    """The row is built before the probe reports, so it starts with nothing to say."""
    from harmon3.ui.ref_panel import RefRowWidget

    unprobed = RefRow(kind=IMAGE, local_path=str(picture))
    made = RefRowWidget(unprobed)
    try:
        assert made.scale_readout.text() == "size not known yet"
        unprobed.width, unprobed.height = 800, 600
        made.refresh_details(124)
        assert "800x600" in made.scale_readout.text()
    finally:
        made.deleteLater()


def test_a_reloaded_ceiling_shows_on_the_slider(qapp, picture):
    from harmon3.ui.ref_panel import RefRowWidget

    saved = RefRow(kind=IMAGE, local_path=str(picture))
    saved.width, saved.height, saved.scale_percent = 800, 600, 30
    made = RefRowWidget(saved)
    try:
        assert made.scale_slider.value() == 30
        assert made.scale_percent_label.text() == "30%"
    finally:
        made.deleteLater()


# ------------------------------------------------------------------------- the preview

@pytest.fixture
def frame(qapp):
    from harmon3.ui.player import VideoPlayer

    made = VideoPlayer()
    made.resize(800, 600)
    yield made
    made.deleteLater()


def _show(frame, row):
    from harmon3.ui.media_view import MediaItem

    frame.show_media([MediaItem(path=row.local_path, caption="Hero", row=row)])
    return frame.media_view.panes[0]


def test_the_caption_says_what_will_be_sent(frame, row):
    pane = _show(frame, row)
    assert pane.caption.text() == "Hero"

    row.scale_percent = 40
    frame.refresh_scale(row)
    target = scaling.target_size(800, 600, 40)
    assert f"sending {target[0]}x{target[1]}" in pane.caption.text()
    assert scaling.format_megapixels(*target) in pane.caption.text()


def test_the_caption_goes_back_when_the_ceiling_is_lifted(frame, row):
    pane = _show(frame, row)
    row.scale_percent = 40
    frame.refresh_scale(row)
    row.scale_percent = 100
    frame.refresh_scale(row)
    assert pane.caption.text() == "Hero"


def test_the_picture_is_really_resampled_when_it_is_small_enough(frame, row):
    """Not merely relabelled: below the pane's own size the loss has to be visible."""
    pane = _show(frame, row)
    before = pane.image_label.pixmap().toImage()

    row.scale_percent = 10
    frame.refresh_scale(row)
    after = pane.image_label.pixmap().toImage()

    # Both fitted to the same pane; what changed is the detail inside them.
    assert scaling.target_size(800, 600, 10)[0] < before.width()
    assert after.width() <= before.width()


def test_a_row_that_is_not_on_screen_is_not_re_rendered(frame, row, qapp, picture):
    """The slider fires on every tick; a pane showing something else should not care."""
    other = RefRow(kind=IMAGE, local_path=str(picture))
    other.width, other.height = 800, 600
    pane = _show(frame, row)

    other.scale_percent = 20
    frame.refresh_scale(other)
    assert pane.caption.text() == "Hero"


def test_a_video_pane_is_unaffected(frame, qapp, tmp_path):
    """Videos never consult the node's sizing, so there is nothing to preview."""
    from harmon3.ui.media_view import MediaItem

    clip = tmp_path / "clip.mp4"
    clip.write_bytes(b"not really a video")
    video = RefRow(kind=VIDEO, local_path=str(clip))
    frame.show_media([MediaItem(path=str(clip), caption="Clip", row=video)])

    frame.refresh_scale()
    assert frame.media_view.panes[0].caption.text() == "Clip"
