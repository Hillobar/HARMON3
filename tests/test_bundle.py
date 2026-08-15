"""Exporting what the node is about to be given.

The bundle exists to be believed, so what it must never do is describe something the run
would not send. These tests mostly pin that: the files come from the job snapshot rather
than the rows on screen, the sizes are measured off the exported files rather than read
off the rows, and the folder-clearing cannot be pointed at somebody's pictures.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmon3 import bundle, config, prompt as prompt_mod   # noqa: E402
from harmon3.graph_builder import BuildState               # noqa: E402
from harmon3.refs import AUDIO, IMAGE, VIDEO, RefRow, RefSet  # noqa: E402

qt = pytest.importorskip("PySide6.QtGui")


@pytest.fixture(scope="module")
def qapp():
    import os
    from PySide6.QtWidgets import QApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def out(tmp_path):
    return tmp_path / "reference_bundle"


def _picture(tmp_path, name="hero.png", width=800, height=600):
    from PySide6.QtGui import QImage

    path = tmp_path / name
    QImage(width, height, QImage.Format_RGB32).save(str(path), "PNG")
    return path


def _state(**kw):
    state = BuildState(
        prompt_sections=prompt_mod.from_legacy(kw.pop("prompt", "A hero shot.")),
        refs=kw.pop("refs", RefSet()),
    )
    for key, value in kw.items():
        setattr(state, key, value)
    return state


def _export(state, sent=None, **kw):
    return bundle.export(state, sent or state, **kw)


# ------------------------------------------------------------------------- what it writes

def test_the_bundle_carries_the_prompt_the_node_receives(qapp, out):
    state = _state(prompt="A hero shot of <Picture 1>.")
    bundle.export(state, state, directory=out)

    assert (out / bundle.PROMPT_NAME).read_text(encoding="utf-8") == state.prompt_text


def test_a_reference_is_named_for_the_tag_that_addresses_it(qapp, tmp_path, out):
    row = RefRow(kind=IMAGE, local_path=str(_picture(tmp_path)))
    row.width, row.height = 800, 600
    state = _state(refs=RefSet(images=[row]))

    result = bundle.export(state, state, directory=out)

    assert len(result.copied) == 1
    assert "Picture_1" in result.copied[0]
    assert result.copied[0].endswith(".png")
    assert (out / result.copied[0]).is_file()


def test_the_extension_is_not_doubled(qapp, tmp_path, out):
    row = RefRow(kind=IMAGE, local_path=str(_picture(tmp_path)))
    state = _state(refs=RefSet(images=[row]))
    result = bundle.export(state, state, directory=out)
    assert not result.copied[0].endswith(".png.png")


def test_the_manifest_records_the_generation_rather_than_the_graphs_links(qapp, out):
    """width, height and length are wired from other nodes; the graph holds ["115", 0]."""
    state = _state(aspect_ratio="16:9 (Widescreen)", megapixels=0.4, duration_seconds=5.0)
    bundle.export(state, state, directory=out)

    manifest = json.loads((out / bundle.MANIFEST_NAME).read_text(encoding="utf-8"))
    generation = manifest["generation"]
    assert generation["width"] == state.resolution[0]
    assert generation["height"] == state.resolution[1]
    assert isinstance(generation["length_frames"], int)


def test_an_unknown_reference_size_is_normalised_the_way_the_graph_does(qapp, out):
    state = _state(ref_image_size="nonsense")
    bundle.export(state, state, directory=out)
    manifest = json.loads((out / bundle.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["generation"]["ref_image_size"] == config.DEFAULT_REF_IMAGE_SIZE


# --------------------------------------------------------------- the snapshot, not the row

def test_the_file_exported_is_the_one_that_would_be_sent(qapp, tmp_path, out):
    """A posed or rescaled row on screen still names the user's own file. Exporting that
    would show the one thing the model never sees."""
    original = _picture(tmp_path, "original.png")
    prepared = _picture(tmp_path, "prepared.png", 400, 300)

    live = RefRow(kind=IMAGE, local_path=str(original))
    live.width, live.height = 800, 600
    sent = RefRow(kind=IMAGE, local_path=str(prepared))
    sent.width, sent.height = 800, 600           # the row still carries the original's

    result = bundle.export(_state(refs=RefSet(images=[live])),
                           _state(refs=RefSet(images=[sent])), directory=out)

    from PySide6.QtGui import QImage
    exported = QImage(str(out / result.copied[0]))
    assert (exported.width(), exported.height()) == (400, 300)


def test_the_sizes_are_measured_not_believed(qapp, tmp_path, out):
    """The row's numbers describe the file that went in, not the one going out."""
    live = RefRow(kind=IMAGE, local_path=str(_picture(tmp_path, "original.png")))
    live.width, live.height = 800, 600
    sent = RefRow(kind=IMAGE, local_path=str(_picture(tmp_path, "prepared.png", 400, 300)))
    sent.width, sent.height = 800, 600

    bundle.export(_state(refs=RefSet(images=[live])),
                  _state(refs=RefSet(images=[sent])), directory=out)

    entry = json.loads((out / bundle.MANIFEST_NAME).read_text(encoding="utf-8"))["references"][0]
    assert entry["sent_size"] == "400x300"
    assert entry["original_size"] == "800x600"
    assert entry["prepared_copy"] is not None


def test_what_the_node_will_do_is_computed_from_the_sent_file(qapp, tmp_path, out):
    live = RefRow(kind=IMAGE, local_path=str(_picture(tmp_path, "original.png", 4000, 3000)))
    live.width, live.height = 4000, 3000
    sent = RefRow(kind=IMAGE, local_path=str(_picture(tmp_path, "prepared.png", 512, 384)))

    bundle.export(_state(refs=RefSet(images=[live]), ref_image_size="max"),
                  _state(refs=RefSet(images=[sent])), directory=out)

    view = json.loads((out / bundle.MANIFEST_NAME).read_text(
        encoding="utf-8"))["references"][0]["node_will_encode"]
    # 512 is below the 2048 cap, so the node keeps it -- computed from 512, not from 4000.
    assert view["encoded_size"] == "512x384"
    assert view["reference_tokens"] == (512 // 16) * (384 // 16)


def test_a_clip_is_reported_against_the_canvas_that_ignores_ref_image_size(qapp, out,
                                                                          tmp_path):
    clip = RefRow(kind=VIDEO, local_path=str(tmp_path / "missing.mp4"))
    clip.width, clip.height = 1920, 1080
    state = _state(refs=RefSet(videos=[clip]))

    bundle.export(state, state, directory=out)
    view = json.loads((out / bundle.MANIFEST_NAME).read_text(
        encoding="utf-8"))["references"][0]["node_will_encode"]
    assert "never consult ref_image_size" in view["sized_by"]


# ------------------------------------------------------------------------------- the tags

def test_a_soundtrack_gets_its_own_line_pointing_at_the_same_file(qapp, tmp_path, out):
    """It has a tag of its own and no file of its own, and both facts matter."""
    clip = RefRow(kind=VIDEO, local_path=str(tmp_path / "clip.mp4"), use_soundtrack=True)
    (tmp_path / "clip.mp4").write_bytes(b"x")
    state = _state(refs=RefSet(videos=[clip]))

    bundle.export(state, state, directory=out)
    entries = json.loads((out / bundle.MANIFEST_NAME).read_text(encoding="utf-8"))["references"]
    audio = [e for e in entries if e["tag"] == "<Audio 1>"]
    assert len(audio) == 1
    assert audio[0]["file_in_bundle"] == entries[0]["file_in_bundle"]


def test_a_server_file_is_reported_rather_than_copied(qapp, out):
    row = RefRow(kind=IMAGE, comfy_name="already-there.png")
    state = _state(refs=RefSet(images=[row]))

    result = bundle.export(state, state, directory=out)
    assert result.copied == []
    assert any("already-there.png" in problem for problem in result.missing)
    assert result.ok is False


def test_a_reference_whose_file_has_gone_is_reported(qapp, tmp_path, out):
    row = RefRow(kind=IMAGE, local_path=str(tmp_path / "vanished.png"))
    state = _state(refs=RefSet(images=[row]))

    result = bundle.export(state, state, directory=out)
    assert any("vanished.png" in problem for problem in result.missing)


def test_notes_are_carried_into_the_manifest(qapp, out):
    bundle.export(_state(), _state(), directory=out, notes=["a skeleton is not rendered"])
    manifest = json.loads((out / bundle.MANIFEST_NAME).read_text(encoding="utf-8"))
    assert manifest["notes"] == ["a skeleton is not rendered"]


# ------------------------------------------------------------------------ clearing safely

def test_a_second_export_replaces_the_first(qapp, tmp_path, out):
    row = RefRow(kind=IMAGE, local_path=str(_picture(tmp_path)))
    first = bundle.export(_state(refs=RefSet(images=[row])), _state(refs=RefSet(images=[row])),
                          directory=out)
    assert (out / first.copied[0]).is_file()

    bundle.export(_state(), _state(), directory=out)
    assert not (out / first.copied[0]).exists()
    assert (out / bundle.MANIFEST_NAME).is_file()


def test_a_folder_this_did_not_write_is_left_alone(qapp, tmp_path):
    """Clearing someone's pictures because the path was wrong would be unforgivable."""
    theirs = tmp_path / "my pictures"
    theirs.mkdir()
    keep = theirs / "wedding.png"
    keep.write_bytes(b"precious")

    with pytest.raises(bundle.BundleError):
        bundle.export(_state(), _state(), directory=theirs)
    assert keep.read_bytes() == b"precious"


def test_an_empty_folder_is_fine_to_use(qapp, tmp_path):
    empty = tmp_path / "empty"
    empty.mkdir()
    bundle.export(_state(), _state(), directory=empty)
    assert (empty / bundle.MANIFEST_NAME).is_file()


def test_a_file_where_the_folder_should_be_is_refused(qapp, tmp_path):
    blocker = tmp_path / "reference_bundle"
    blocker.write_bytes(b"not a folder")
    with pytest.raises(bundle.BundleError):
        bundle.export(_state(), _state(), directory=blocker)


def test_measuring_something_unreadable_returns_nothing_rather_than_raising(qapp, tmp_path):
    junk = tmp_path / "junk.png"
    junk.write_bytes(b"not a picture")
    assert bundle.measure(junk, IMAGE) is None
    assert bundle.measure(junk, VIDEO) is None
    assert bundle.measure(junk, AUDIO) is None
