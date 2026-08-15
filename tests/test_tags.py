"""Tests for the <Picture i> / <Video k> / <Audio j> ordinal algorithm and prompt remap."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmon3 import config                                    # noqa: E402
from harmon3.refs import (                                    # noqa: E402
    AUDIO,
    IMAGE,
    VIDEO,
    RefRow,
    RefSet,
    compute_tags,
    remap_prompt,
    row_warnings,
    tag_migration,
    tags_in_prompt,
    unknown_tags,
    unused_tags,
)


import pytest                                                  # noqa: E402


def _row(kind, name, **kw):
    return RefRow(kind=kind, local_path=f"C:/refs/{name}", **kw)


@pytest.fixture
def real_video(tmp_path):
    """A reference row pointing at a file that actually exists.

    row_warnings reports a vanished local file, so anything exercising the *other*
    diagnostics needs a real path or that error masks them.
    """
    path = tmp_path / "clip.mp4"
    path.write_bytes(b"not really a video, but it exists")

    def make(**kw):
        return RefRow(kind=VIDEO, local_path=str(path), **kw)

    return make


def _refset(images=0, videos=0, audios=0, soundtracks=None):
    soundtracks = soundtracks if soundtracks is not None else [True] * videos
    return RefSet(
        images=[_row(IMAGE, f"img{i}.png") for i in range(images)],
        videos=[_row(VIDEO, f"vid{k}.mp4", use_soundtrack=soundtracks[k]) for k in range(videos)],
        audios=[_row(AUDIO, f"aud{j}.wav") for j in range(audios)],
    )


def test_canonical_example():
    """2 images + 1 video with soundtrack + 1 standalone audio.

    The soundtrack's label is emitted immediately BEFORE its video's label, which is what
    pushes the standalone audio to <Audio 2>.
    """
    refset = _refset(images=2, videos=1, audios=1)
    tags = compute_tags(refset)
    assert tags.order == ["<Picture 1>", "<Picture 2>", "<Audio 1>", "<Video 1>", "<Audio 2>"]
    assert tags.tag_for(refset.images[0]) == "<Picture 1>"
    assert tags.tag_for(refset.videos[0]) == "<Video 1>"
    assert tags.soundtrack_tag_for(refset.videos[0]) == "<Audio 1>"
    assert tags.tag_for(refset.audios[0]) == "<Audio 2>"


def test_unchecking_soundtrack_collapses_standalone_audio():
    refset = _refset(images=2, videos=1, audios=1)
    before = compute_tags(refset)

    refset.videos[0].use_soundtrack = False
    after = compute_tags(refset)

    assert after.order == ["<Picture 1>", "<Picture 2>", "<Video 1>", "<Audio 1>"]
    assert after.soundtrack_tag_for(refset.videos[0]) is None
    # The standalone audio row kept its identity but moved from <Audio 2> to <Audio 1>.
    assert tag_migration(before, after) == {"<Audio 2>": "<Audio 1>"}


def test_multiple_videos_interleave_audio_ordinals():
    refset = _refset(images=1, videos=2, audios=1, soundtracks=[True, True])
    tags = compute_tags(refset)
    assert tags.order == [
        "<Picture 1>",
        "<Audio 1>", "<Video 1>",
        "<Audio 2>", "<Video 2>",
        "<Audio 3>",
    ]


def test_partial_soundtracks():
    """A video without a soundtrack contributes no audio ordinal."""
    refset = _refset(videos=2, audios=1, soundtracks=[False, True])
    tags = compute_tags(refset)
    assert tags.order == ["<Video 1>", "<Audio 1>", "<Video 2>", "<Audio 2>"]
    assert tags.soundtrack_tag_for(refset.videos[0]) is None
    assert tags.soundtrack_tag_for(refset.videos[1]) == "<Audio 1>"


def test_empty_refset():
    tags = compute_tags(RefSet())
    assert tags.order == []
    assert tags.all_tags() == set()


def test_migration_is_empty_when_nothing_moves():
    refset = _refset(images=2, videos=1, audios=1)
    assert tag_migration(compute_tags(refset), compute_tags(refset)) == {}


def test_simultaneous_remap_does_not_cascade():
    """1->2 and 2->3 applied together must not turn the original 1 into 3.

    Chained str.replace calls would; a single scan of the original text cannot.
    """
    prompt = "Use <Audio 1> then <Audio 2> to drive it."
    migration = {"<Audio 1>": "<Audio 2>", "<Audio 2>": "<Audio 3>"}
    assert remap_prompt(prompt, migration) == "Use <Audio 2> then <Audio 3> to drive it."


def test_remap_leaves_unmigrated_tags_alone():
    prompt = "<Picture 1> and <Picture 2> and <Audio 1>"
    assert remap_prompt(prompt, {"<Audio 1>": "<Audio 2>"}) == (
        "<Picture 1> and <Picture 2> and <Audio 2>"
    )


def test_remap_with_empty_migration_is_identity():
    prompt = "unchanged <Video 1>"
    assert remap_prompt(prompt, {}) is prompt


def test_remap_tolerates_loose_tag_spelling():
    """The model's own prompts vary in spacing/case; the rewriter must still find them."""
    prompt = "<audio  1> and < Audio 1 >"
    assert remap_prompt(prompt, {"<Audio 1>": "<Audio 4>"}) == "<Audio 4> and <Audio 4>"


def test_tags_in_prompt_canonicalises():
    assert tags_in_prompt("<picture 2> <VIDEO 1> <Audio  3>") == [
        "<Picture 2>", "<Video 1>", "<Audio 3>",
    ]


def test_shipped_prompt_tags():
    """Whatever tags the shipped prompt uses are parsed out of it, in order."""
    workflow = config.load_workflow()
    prompt = workflow.graph[workflow.roles.promptinput]["inputs"]["value"]
    assert tags_in_prompt(prompt) == sorted(set(tags_in_prompt(prompt)),
                                            key=tags_in_prompt(prompt).index)


def test_unknown_tags_flags_missing_references():
    refset = _refset(images=2)
    tags = compute_tags(refset)
    prompt = "Use <Picture 1>, <Picture 2> and <Audio 1>."
    assert unknown_tags(prompt, tags) == ["<Audio 1>"]
    assert unknown_tags("Use <Picture 1> and <Picture 2>.", tags) == []


def test_unknown_tags_deduplicates():
    tags = compute_tags(RefSet())
    assert unknown_tags("<Audio 1> <Audio 1> <Audio 2>", tags) == ["<Audio 1>", "<Audio 2>"]


def test_unused_tags_reports_in_presentation_order():
    refset = _refset(images=2, audios=1)
    tags = compute_tags(refset)
    assert unused_tags("Only <Picture 2> here.", tags) == ["<Picture 1>", "<Audio 1>"]


def test_refset_respects_kind_limits():
    refset = RefSet()
    for i in range(config.MAX_REF_IMAGES + 3):
        if refset.can_add(IMAGE):
            refset.images.append(_row(IMAGE, f"i{i}.png"))
    assert len(refset.images) == config.MAX_REF_IMAGES
    assert not refset.can_add(IMAGE)


def test_refset_roundtrip_through_settings():
    refset = _refset(images=2, videos=1, audios=1, soundtracks=[False])
    restored = RefSet.from_list(refset.to_list())
    assert [r.display_name for r in restored.images] == ["img0.png", "img1.png"]
    assert restored.videos[0].use_soundtrack is False
    assert compute_tags(restored).order == compute_tags(refset).order


def test_refset_from_list_drops_overflow_and_junk():
    entries = [{"kind": "image", "local_path": f"C:/x/{i}.png"} for i in range(12)]
    entries.append({"kind": "bogus", "local_path": "C:/x/y.png"})
    entries.append({"kind": "image"})  # neither path nor server name
    restored = RefSet.from_list(entries)
    assert len(restored.images) == config.MAX_REF_IMAGES


def test_server_file_row_needs_no_upload():
    row = RefRow(kind=IMAGE, comfy_name="red_superboy_on_city_roof.png")
    assert row.needs_upload is False
    assert row.display_name == "red_superboy_on_city_roof.png"


def test_row_warnings_blocks_too_short_video(real_video):
    row = real_video()
    row.frame_count, row.fps, row.has_audio = 3, 24.0, True
    errors, _ = row_warnings(row, target_frames=124)
    assert errors and "at least 5" in errors[0]


def test_row_warnings_flags_wrong_fps_and_truncation(real_video):
    row = real_video()
    row.frame_count, row.fps, row.has_audio = 900, 30.0, True
    errors, warnings = row_warnings(row, target_frames=124)
    assert not errors
    assert any("truncated to 124" in w for w in warnings)
    assert any("30 fps" in w for w in warnings)


def test_row_warnings_flags_missing_soundtrack(real_video):
    row = real_video(use_soundtrack=True)
    row.frame_count, row.fps, row.has_audio = 124, 24.0, False
    errors, warnings = row_warnings(row, target_frames=124)
    assert not errors
    assert any("no audio track" in w for w in warnings)


def test_row_warnings_silent_when_unprobed(real_video, tmp_path):
    image = tmp_path / "a.png"
    image.write_bytes(b"x")
    assert row_warnings(real_video(), target_frames=124) == ([], [])
    assert row_warnings(RefRow(kind=IMAGE, local_path=str(image)), target_frames=124) == ([], [])


def test_row_warnings_flags_a_vanished_local_file():
    """A scene reloaded weeks later may reference files that have since been moved."""
    row = _row(IMAGE, "gone.png")
    assert row.local_missing is True
    errors, _ = row_warnings(row, target_frames=124)
    assert errors and "no longer exists" in errors[0]


def test_server_rows_are_never_reported_as_locally_missing():
    row = RefRow(kind=IMAGE, comfy_name="on_the_server.png")
    assert row.local_missing is False
    assert row_warnings(row, target_frames=124) == ([], [])
