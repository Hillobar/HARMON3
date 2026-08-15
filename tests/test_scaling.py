"""Per-image size ceilings: the arithmetic, the cache, and the substitution.

The node has one ``ref_image_size`` for every reference it receives, so a per-image size
can only be a ceiling applied before upload. That works because the node's scale factor is
``min(1.0, ...)`` in both of its modes -- it never enlarges a reference -- which is the
fact the whole feature rests on and the reason these tests check the sizing against the
node's own 32-pixel grid rather than against a percentage.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmon3 import config, scaling                     # noqa: E402
from harmon3.graph_builder import BuildState            # noqa: E402
from harmon3.refs import AUDIO, IMAGE, VIDEO, RefRow, RefSet   # noqa: E402
from harmon3.scenes import signature_of                 # noqa: E402


def _image(path="D:/refs/hero.png", width=4000, height=3000, percent=100):
    row = RefRow(kind=IMAGE, local_path=path)
    row.width, row.height = width, height
    row.scale_percent = percent
    return row


# --------------------------------------------------------------------------- the range

@pytest.mark.parametrize("given, expected", [
    (100, 100), (10, 10), (55, 55),
    (0, scaling.MIN_PERCENT), (-20, scaling.MIN_PERCENT),
    (400, scaling.MAX_PERCENT),
    ("nonsense", scaling.MAX_PERCENT), (None, scaling.MAX_PERCENT),
])
def test_the_percentage_is_brought_into_range(given, expected):
    """The slider cannot produce a bad one; a hand-edited scene file can."""
    assert scaling.clamp_percent(given) == expected


# ---------------------------------------------------------------------------- the size

def test_a_target_lands_on_the_nodes_own_grid():
    """32 is what the node rounds to; matching it means the size shown is the size sent."""
    width, height = scaling.target_size(4000, 3000, 40)
    assert width % scaling.GRID == 0
    assert height % scaling.GRID == 0
    assert (width, height) == (1600, 1216)


def test_a_target_is_never_larger_than_the_source():
    """Rounding a 1010px axis up to 1024 to satisfy the grid would be an upscale."""
    assert scaling.target_size(1010, 1010, 100) == (1010, 1010)
    assert scaling.target_size(20, 20, 100) == (20, 20)
    assert scaling.target_size(200, 200, 10) == (192, 192)   # floor cannot enlarge


def test_full_size_is_the_source_size():
    assert scaling.target_size(1920, 1080, 100) == (1920, 1080)


def test_smaller_percentages_give_smaller_targets():
    sizes = [scaling.target_size(4000, 3000, p) for p in (100, 75, 50, 25, 10)]
    widths = [w for w, _h in sizes]
    assert widths == sorted(widths, reverse=True)
    assert len(set(widths)) == len(widths)


def test_the_short_edge_has_a_floor_so_the_aspect_survives():
    """Each axis snaps to the grid on its own, so at 80x60 both land on 64 and a 4:3
    reference arrives square. The floor keeps the rounding relative rather than total."""
    width, height = scaling.target_size(800, 600, 10)
    assert min(width, height) == scaling.MIN_SHORT_EDGE
    assert width / height == pytest.approx(800 / 600, rel=0.06)


def test_the_floor_does_not_apply_to_an_image_with_room_to_spare():
    assert scaling.target_size(4000, 3000, 10) == (384, 288)


@pytest.mark.parametrize("width, height, expected", [
    (1000, 1000, "1.0 MP"),
    (4000, 3000, "12.0 MP"),
    (800, 600, "0.48 MP"),
    (64, 64, "0.004 MP"),        # not "0.00 MP", which reads as nothing at all
    (0, 0, ""),
])
def test_megapixels_are_reported_at_a_useful_precision(width, height, expected):
    assert scaling.format_megapixels(width, height) == expected


def test_the_readout_says_what_will_be_sent():
    assert scaling.describe(4000, 3000, 100) == "4000x3000  -  12.0 MP"
    assert "1600x1216" in scaling.describe(4000, 3000, 40)
    assert "from 4000x3000" in scaling.describe(4000, 3000, 40)


def test_the_readout_waits_for_the_probe():
    assert scaling.describe(None, None, 50) == "size not known yet"


# ----------------------------------------------------------------------------- the row

def test_pictures_and_clips_carry_a_ceiling_and_audio_does_not():
    assert _image().supports_scale is True
    assert RefRow(kind=VIDEO, local_path="a.mp4").supports_scale is True
    assert RefRow(kind=AUDIO, local_path="a.wav").supports_scale is False


def test_a_row_at_full_size_is_not_scaling():
    row = _image(percent=100)
    assert row.scales is False
    assert scaling.wants_scaling(row) is False
    assert row.scaled_size() == (4000, 3000)


def test_a_row_below_full_size_is_scaling():
    row = _image(percent=40)
    assert row.scales is True
    assert scaling.wants_scaling(row) is True
    assert row.scaled_size() == (1600, 1216)


def test_an_unprobed_row_has_nothing_to_scale_yet():
    """Width and height arrive asynchronously; until they do there is no target."""
    row = _image(percent=40)
    row.width = row.height = None
    assert row.scaled_size() is None
    assert scaling.wants_scaling(row) is False


def test_a_server_file_has_nothing_local_to_resize():
    row = RefRow(kind=IMAGE, comfy_name="already-there.png")
    row.width, row.height, row.scale_percent = 4000, 3000, 40
    assert scaling.wants_scaling(row) is False


# ------------------------------------------------------------------------ persistence

def test_the_ceiling_survives_a_round_trip():
    restored = RefRow.from_dict(_image(percent=35).to_dict())
    assert restored.scale_percent == 35


def test_a_reference_saved_before_ceilings_reads_as_full_size():
    restored = RefRow.from_dict({"kind": IMAGE, "local_path": "a.png"})
    assert restored.scale_percent == 100


def test_a_hand_edited_ceiling_is_brought_into_range():
    data = _image().to_dict()
    data["scale_percent"] = 5000
    assert RefRow.from_dict(data).scale_percent == scaling.MAX_PERCENT


def test_a_clips_ceiling_survives_a_round_trip():
    data = {"kind": VIDEO, "local_path": "a.mp4", "scale_percent": 40}
    assert RefRow.from_dict(data).scale_percent == 40


def test_audio_never_gains_a_ceiling_from_a_file():
    data = {"kind": AUDIO, "local_path": "a.wav", "scale_percent": 40}
    assert RefRow.from_dict(data).scale_percent == 100


def test_changing_the_ceiling_marks_a_scene_modified():
    """It changes what is sent, so it is an edit -- unlike a probe result."""
    def signature(percent):
        row = _image(percent=percent)
        return signature_of("p", 5.0, 1, False, RefSet(images=[row]).to_list())

    assert signature(100) != signature(40)
    assert signature(40) == signature(40)


# ---------------------------------------------------------------------------- the cache

def test_two_percentages_that_snap_together_share_one_file():
    """The key is the target size, not the slider position -- the file is identical."""
    a, b = scaling.target_size(4000, 3000, 40), scaling.target_size(4000, 3000, 41)
    if a != b:
        pytest.skip("these two do not collide on this grid")
    assert scaling.cache_key("d", 4000, 3000, 40) == scaling.cache_key("d", 4000, 3000, 41)


def test_a_different_target_is_a_different_file():
    base = scaling.cache_key("d", 4000, 3000, 40)
    assert scaling.cache_key("d", 4000, 3000, 70) != base
    assert scaling.cache_key("other", 4000, 3000, 40) != base


def test_the_cached_file_is_named_after_its_source():
    """A node error is matched back to a row by looking for its name in the label."""
    path = scaling.cached_path("a1b2c3d4e5f6", "Robyn portrait.png")
    assert "Robyn_portrait" in path.name
    assert path.suffix == ".png"


# ----------------------------------------------------------------------- substitution

def test_only_the_snapshot_is_repointed():
    """The live row keeps naming the user's own file, so the slider stays meaningful."""
    row = _image(percent=40)
    state = BuildState(refs=RefSet(images=[row]))

    scaling.swap_in(state, {row.uid: "D:/runs/scaled/hero_abc.png"})
    assert state.refs.images[0].local_path == "D:/runs/scaled/hero_abc.png"
    assert state.refs.images[0].comfy_name is None


def test_a_row_with_no_rescale_is_left_alone():
    row = _image(percent=100)
    state = BuildState(refs=RefSet(images=[row]))
    scaling.swap_in(state, {})
    assert state.refs.images[0].local_path == "D:/refs/hero.png"


# --------------------------------------------------------------------------- the encode

@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "SCALE_CACHE_DIR", tmp_path / "scaled")
    return tmp_path / "scaled"


needs_qt = pytest.mark.skipif(
    __import__("importlib.util", fromlist=["util"]).find_spec("PySide6") is None,
    reason="Qt is the image codec here")


@needs_qt
def test_a_rescale_is_written_at_the_size_asked_for(qtapp, tmp_path, cache_dir):
    from PySide6.QtGui import QImage

    source = tmp_path / "big.png"
    QImage(800, 600, QImage.Format_RGB32).save(str(source), "PNG")

    row = _image(str(source), 800, 600, percent=50)
    written = scaling.render(row)

    result = QImage(str(written))
    assert (result.width(), result.height()) == scaling.target_size(800, 600, 50)
    assert written.parent == cache_dir


@needs_qt
def test_a_second_render_reuses_the_first(qtapp, tmp_path, cache_dir):
    from PySide6.QtGui import QImage

    source = tmp_path / "big.png"
    QImage(800, 600, QImage.Format_RGB32).save(str(source), "PNG")
    row = _image(str(source), 800, 600, percent=50)

    first = scaling.render(row)
    stamp = first.stat().st_mtime_ns
    second = scaling.render(row)
    assert second == first
    assert second.stat().st_mtime_ns == stamp        # not rewritten


@needs_qt
def test_an_unreadable_source_is_refused_rather_than_written(qtapp, tmp_path, cache_dir):
    source = tmp_path / "not-an-image.png"
    source.write_bytes(b"nope")
    row = _image(str(source), 800, 600, percent=50)

    with pytest.raises(scaling.ScaleError):
        scaling.render(row)
    assert scaling.cached_copies() == []


@needs_qt
def test_nothing_is_left_behind_when_a_write_fails(qtapp, tmp_path, cache_dir):
    """A half-written PNG at the name the cache looks for would be found and uploaded."""
    source = tmp_path / "not-an-image.png"
    source.write_bytes(b"nope")
    row = _image(str(source), 800, 600, percent=50)
    with pytest.raises(scaling.ScaleError):
        scaling.render(row)
    assert list(cache_dir.glob("*.part*")) == []


# -------------------------------------------------------------------------- the default

def test_the_node_setting_defaults_to_max():
    """Under "match" the node caps every reference at the generation's area anyway, which
    would override most of what a per-image slider was set to."""
    assert config.DEFAULT_REF_IMAGE_SIZE == "max"


@pytest.fixture(scope="module")
def qtapp():
    import os
    pytest.importorskip("PySide6.QtWidgets")
    from PySide6.QtWidgets import QApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app


# ---------------------------------------------------------------------------------
# Video: a share of the model's canvas rather than of the file
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("width, height, expected", [
    (3840, 2160, (1344, 768)),      # every HD-and-up clip lands on the same canvas
    (1920, 1080, (1344, 768)),
    (1280, 720, (1280, 704)),
    (444, 250, (448, 256)),         # already smaller: the node keeps the source size
])
def test_the_canvas_matches_the_nodes_own_rule(width, height, expected):
    assert scaling.video_canvas(width, height) == expected


def test_a_clips_slider_measures_the_canvas_not_the_file():
    """A share of the source would move a 4K clip from 3840 to 1920 and change nothing:
    both are flattened onto 1344x768 before anything is encoded."""
    hd = scaling.video_target_size(1920, 1080, 50)
    uhd = scaling.video_target_size(3840, 2160, 50)
    assert hd == uhd == (672, 384)


def test_full_size_is_the_canvas_the_node_would_have_used():
    assert scaling.video_target_size(1920, 1080, 100) == scaling.video_canvas(1920, 1080)


def test_every_position_on_a_clips_slider_does_something():
    sizes = [scaling.video_target_size(1920, 1080, p) for p in (100, 75, 50)]
    assert len(set(sizes)) == 3


def test_a_clip_already_below_the_floor_cannot_shrink():
    """Its slider is inert, and the readout says so rather than promising a change."""
    assert scaling.video_target_size(444, 250, 10) == scaling.video_canvas(444, 250)
    assert "the model's own size" in scaling.describe_video(444, 250, 10)


def test_a_clips_readout_says_what_it_saves():
    text = scaling.describe_video(1920, 1080, 50)
    assert "672x384 per frame" in text
    assert "% of the tokens" in text
    assert "normally 1344x768" in text


def test_a_clip_scales_only_below_its_canvas():
    row = RefRow(kind=VIDEO, local_path="a.mp4")
    row.width, row.height = 1920, 1080

    row.scale_percent = 100
    assert row.scaled_size() == (1344, 768)
    row.scale_percent = 50
    assert row.scaled_size() == (672, 384)


def test_an_unscaled_clip_needs_no_render_of_its_own():
    from harmon3 import pose

    row = RefRow(kind=VIDEO, local_path="a.mp4")
    row.width, row.height = 1920, 1080
    assert pose.canvas_for(row) is None
    assert pose.needs_render(row) is False


def test_a_scaled_clip_needs_a_render_even_without_a_pose():
    from harmon3 import pose

    row = RefRow(kind=VIDEO, local_path="a.mp4")
    row.width, row.height, row.scale_percent = 1920, 1080, 50
    assert pose.canvas_for(row) == (672, 384)
    assert pose.needs_render(row) is True


def test_a_skeleton_and_a_rescale_of_one_section_are_different_files():
    """Same source, same section, same settings -- and one must not be sent for the other."""
    from harmon3 import pose

    settings = pose.PoseSettings()
    posed = pose.cache_key("d", settings, 0, 124, (672, 384), True)
    scaled = pose.cache_key("d", settings, 0, 124, (672, 384), False)
    full = pose.cache_key("d", settings, 0, 124, None, True)
    assert len({posed, scaled, full}) == 3


def test_a_wrongly_shaped_request_is_resolved_in_favour_of_the_picture(qtapp, tmp_path,
                                                                       cache_dir):
    """The sizes come from what the row says a reference measures, and the picture is
    loaded separately. Those two came apart once already; stretching is the one outcome
    worth ruling out entirely."""
    from PySide6.QtGui import QImage

    source = tmp_path / "portrait.png"
    QImage(600, 1000, QImage.Format_RGB32).save(str(source), "PNG")

    # A landscape request for a portrait picture, as a bad probe would produce.
    written = QImage(str(scaling.write_scaled(source, cache_dir / "out.png", (512, 288))))

    assert written.height() > written.width()
    assert written.width() / written.height() == pytest.approx(600 / 1000, rel=0.08)


def test_the_grid_rounding_is_not_mistaken_for_a_disagreement(qtapp, tmp_path, cache_dir):
    """A few per cent is the 32-pixel snap doing its job and must be left alone."""
    from PySide6.QtGui import QImage

    source = tmp_path / "portrait.png"
    QImage(600, 1000, QImage.Format_RGB32).save(str(source), "PNG")

    asked = scaling.target_size(600, 1000, 50)
    written = QImage(str(scaling.write_scaled(source, cache_dir / "out.png", asked)))
    assert (written.width(), written.height()) == asked
