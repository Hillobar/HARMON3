"""A photograph stored sideways with an EXIF rotation, and every place that reads one.

ComfyUI's LoadImage obeys the EXIF Orientation tag (``ImageOps.exif_transpose``) and Qt
does not, so left alone the app and the server disagree about which way up a reference is.
That disagreement is invisible in the worst way: the thumbnail merely looks wrong, but a
rescaled copy is written as PNG -- which carries no EXIF -- and would arrive at the model
sideways while the same reference unscaled arrives upright.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6.QtWidgets")

from PySide6.QtGui import QColor, QImage                # noqa: E402
from PySide6.QtWidgets import QApplication              # noqa: E402

from harmon3 import config, imaging, scaling            # noqa: E402
from harmon3.refs import IMAGE, RefRow                  # noqa: E402

#: EXIF orientation 6: "the camera was turned a quarter clockwise", by far the commonest.
ROTATE_90 = 6
#: Orientation 2 mirrors horizontally without turning anything.
MIRROR = 2


@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app


def _exif_app1(orientation: int) -> bytes:
    """A minimal APP1 segment carrying nothing but an Orientation tag."""
    tiff = b"II" + struct.pack("<HI", 42, 8)
    tiff += (struct.pack("<H", 1)
             + struct.pack("<HHI", 0x0112, 3, 1)      # tag 0x0112, SHORT, count 1
             + struct.pack("<HH", orientation, 0)
             + struct.pack("<I", 0))                  # no next IFD
    payload = b"Exif\x00\x00" + tiff
    return b"\xFF\xE1" + struct.pack(">H", len(payload) + 2) + payload


def _photo(tmp_path, name, orientation=None, width=400, height=200):
    """A landscape JPEG, optionally tagged as needing turning."""
    plain = tmp_path / f"{name}-plain.jpg"
    picture = QImage(width, height, QImage.Format_RGB32)
    picture.fill(QColor("red"))
    assert picture.save(str(plain), "JPEG")
    if orientation is None:
        return plain

    data = plain.read_bytes()
    tagged = tmp_path / f"{name}.jpg"
    tagged.write_bytes(data[:2] + _exif_app1(orientation) + data[2:])
    return tagged


# ------------------------------------------------------------------------ the fixture

def test_the_fixture_really_is_tagged(qapp, tmp_path):
    """If Qt stopped reporting the tag, every test below would pass for the wrong reason."""
    from PySide6.QtGui import QImageIOHandler, QImageReader

    reader = QImageReader(str(_photo(tmp_path, "rot", ROTATE_90)))
    rotate90 = QImageIOHandler.Transformation.TransformationRotate90
    assert reader.transformation() & rotate90


# ------------------------------------------------------------------------- the reading

def test_a_sideways_photograph_reports_the_shape_the_model_will_see(qapp, tmp_path):
    assert imaging.oriented_size(_photo(tmp_path, "rot", ROTATE_90)) == (200, 400)


def test_an_untagged_photograph_is_left_exactly_as_it_is(qapp, tmp_path):
    assert imaging.oriented_size(_photo(tmp_path, "plain")) == (400, 200)


def test_a_mirrored_photograph_is_still_as_wide_as_it_was(qapp, tmp_path):
    """Only a quarter turn swaps the axes; mirroring must not be mistaken for one."""
    assert imaging.oriented_size(_photo(tmp_path, "mir", MIRROR)) == (400, 200)


def test_something_that_is_not_an_image_reports_nothing(qapp, tmp_path):
    junk = tmp_path / "nope.jpg"
    junk.write_bytes(b"not a picture")
    assert imaging.oriented_size(junk) is None


def test_loading_turns_the_picture_the_right_way_up(qapp, tmp_path):
    image = imaging.load(_photo(tmp_path, "rot", ROTATE_90))
    assert (image.width(), image.height()) == (200, 400)


def test_a_pixmap_is_turned_the_same_way(qapp, tmp_path):
    pixmap = imaging.load_pixmap(_photo(tmp_path, "rot", ROTATE_90))
    assert (pixmap.width(), pixmap.height()) == (200, 400)


def test_an_unreadable_file_gives_a_null_pixmap_rather_than_raising(qapp, tmp_path):
    junk = tmp_path / "nope.png"
    junk.write_bytes(b"not a picture")
    assert imaging.load_pixmap(junk).isNull()


# ---------------------------------------------------------------- what actually goes out

def test_a_rescaled_copy_is_written_the_right_way_up(qapp, tmp_path, monkeypatch):
    """PNG carries no EXIF, so a rescale is the last chance to get this right."""
    monkeypatch.setattr(config, "SCALE_CACHE_DIR", tmp_path / "scaled")
    source = _photo(tmp_path, "rot", ROTATE_90, width=800, height=600)

    row = RefRow(kind=IMAGE, local_path=str(source))
    # The probe reports the oriented shape, so the row holds 600x800.
    row.width, row.height = imaging.oriented_size(source)
    row.scale_percent = 50

    written = QImage(str(scaling.render(row)))
    assert written.height() > written.width()                 # still portrait
    assert (written.width(), written.height()) == scaling.target_size(600, 800, 50)


def test_the_thumbnail_is_turned_too(qapp, tmp_path):
    from harmon3.ui import ref_panel

    source = _photo(tmp_path, "rot", ROTATE_90)
    ref_panel._thumb_cache.pop(str(source), None)
    thumb = ref_panel.thumbnail_for(str(source))
    assert thumb is not None
    assert thumb.height() > thumb.width()


def test_the_result_frame_shows_it_turned(qapp, tmp_path):
    from harmon3.ui.media_view import MediaItem, MediaPane

    source = _photo(tmp_path, "rot", ROTATE_90)
    pane = MediaPane()
    try:
        pane.show_item(MediaItem(path=str(source), caption="Photo"))
        assert pane._pixmap.height() > pane._pixmap.width()
    finally:
        pane.deleteLater()


# ------------------------------------------------------------------- who probes an image

def test_a_still_is_probed_by_qt_rather_than_by_pyav(qapp, tmp_path):
    """PyAV opens a still happily -- as a one-frame video -- and reports the dimensions
    stored in the file, with no notion of the Orientation tag. It used to be asked first,
    so a portrait photograph came back 1000x600 and every size derived from it was
    computed for a landscape picture."""
    from harmon3.ui.probe import MediaProbe

    source = _photo(tmp_path, "rot", ROTATE_90, width=1000, height=600)

    seen = {}
    probe = MediaProbe()
    probe.probed.connect(lambda _uid, info: seen.update(info))
    probe.probe(1, str(source), "image")

    assert (seen.get("width"), seen.get("height")) == (600, 1000)


def test_an_unreadable_still_is_still_reported_as_unreadable(qapp, tmp_path):
    from harmon3.ui.probe import MediaProbe

    junk = tmp_path / "nope.png"
    junk.write_bytes(b"not a picture")

    problems = []
    probe = MediaProbe()
    probe.unreadable.connect(lambda _uid, why: problems.append(why))
    probe.probe(1, str(junk), "image")

    assert problems


def test_a_rescaled_photograph_keeps_its_shape_end_to_end(qapp, tmp_path, monkeypatch):
    """The bug this guards: probe says landscape, the loader hands back portrait, and the
    copy written from the two is the picture squashed sideways."""
    from harmon3.ui.probe import MediaProbe

    monkeypatch.setattr(config, "SCALE_CACHE_DIR", tmp_path / "scaled")
    source = _photo(tmp_path, "rot", ROTATE_90, width=1000, height=600)

    seen = {}
    probe = MediaProbe()
    probe.probed.connect(lambda _uid, info: seen.update(info))
    probe.probe(1, str(source), "image")

    row = RefRow(kind=IMAGE, local_path=str(source))
    row.width, row.height = seen["width"], seen["height"]
    row.scale_percent = 50

    written = QImage(str(scaling.render(row)))
    assert written.height() > written.width()
    assert written.width() / written.height() == pytest.approx(600 / 1000, rel=0.08)
