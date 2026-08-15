"""Trim marks: where a video or audio reference starts, and how much of it is sent.

A reference has an in point and no out point. MiniMaxH3ReferenceToVideo truncates every
reference to the generated length, so the section runs from the mark for exactly that
long: the duration parameter is the out point, and these tests are mostly about that
length arriving from the right place.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmon3 import config, graph_builder, mathmirror                # noqa: E402
from harmon3.graph_builder import BuildState, build_graph            # noqa: E402
from harmon3.refs import (                                           # noqa: E402
    AUDIO,
    IMAGE,
    VIDEO,
    RefRow,
    RefSet,
    row_warnings,
)

WORKFLOW = config.load_workflow()
BASE, ROLES = WORKFLOW.graph, WORKFLOW.roles

#: The generated length of a five-second clip, which is what most of these send.
FRAMES_5S = mathmirror.frames_from_seconds(5.0)


def _video(name="clip.mp4", *, start=0, fps=30.0, frames=300, **kw):
    row = RefRow(kind=VIDEO, comfy_name=name, **kw)
    row.fps, row.frame_count = fps, frames
    row.trim_start = start
    return row


def _audio(name="roar.wav", *, start=0, duration=10.0):
    row = RefRow(kind=AUDIO, comfy_name=name)
    row.duration_s = duration
    row.trim_start = start
    return row


def _minimax(built):
    return built.graph[ROLES.reference]["inputs"]


# ---------------------------------------------------------------------------------
# What can be trimmed
# ---------------------------------------------------------------------------------

def test_only_things_with_a_timeline_can_be_trimmed():
    assert RefRow(kind=VIDEO, comfy_name="a.mp4").supports_trim is True
    assert RefRow(kind=AUDIO, comfy_name="a.wav").supports_trim is True
    assert RefRow(kind=IMAGE, comfy_name="a.png").supports_trim is False


def test_an_image_never_reports_a_mark_even_if_the_field_is_set():
    row = RefRow(kind=IMAGE, comfy_name="a.png")
    row.trim_start = 1
    assert row.marked is False


def test_marked_means_it_starts_somewhere_other_than_the_beginning():
    """Not a switch for whether it is cut -- everything with a timeline is -- only
    whether there is a mark worth drawing or saying."""
    assert _video(start=10).marked is True
    assert _video(start=0).marked is False


def test_a_reference_stores_neither_an_out_point_nor_a_switch():
    stored = _video(start=5).to_dict()
    assert "trim_end" not in stored
    assert "trim_enabled" not in stored


# ---------------------------------------------------------------------------------
# The length comes from the generated length
# ---------------------------------------------------------------------------------

def test_a_videos_section_is_as_long_as_the_clip_being_generated():
    assert _video(start=24).trim_length(124) == 124
    assert _video(start=24).trim_span(124) == (24, 124)


def test_an_audios_section_is_the_same_length_in_seconds():
    """Audio is cut by time, and the generated length is a frame count at 24 fps."""
    assert _audio(start=1.5).trim_length(120) == 5.0
    assert _audio(start=1.5).trim_span(120) == (1.5, 5.0)


def test_the_section_ends_where_the_reference_runs_out():
    """What is shown has to be what exists: asking a loader for frames past the end of a
    file is harmless, but drawing them as though they were there is not."""
    assert _video(start=24, frames=300).trim_end(124) == 148
    assert _video(start=250, frames=300).trim_end(124) == 300
    assert _audio(start=8.0, duration=10.0).trim_end(120) == 10.0


def test_an_unprobed_reference_is_taken_at_its_word():
    row = RefRow(kind=VIDEO, comfy_name="a.mp4")
    row.trim_start = 10
    assert row.trim_end(124) == 134


# ---------------------------------------------------------------------------------
# Units
# ---------------------------------------------------------------------------------

def test_a_videos_section_converts_to_seconds_through_the_sources_frame_rate():
    """124 source frames at 30 fps is 4.13 s of the file, whatever it becomes on output."""
    start, length = _video(start=30, fps=30.0).trim_seconds(120)
    assert (start, length) == (1.0, 4.0)


def test_a_video_with_no_known_frame_rate_cannot_express_seconds():
    """Frames are windowed by index and sound by time; without fps they cannot be lined up."""
    assert _video(start=30, fps=None).trim_seconds(120) is None


def test_audio_is_already_in_seconds():
    assert _audio(start=1.5).trim_seconds(120) == (1.5, 5.0)


def test_a_row_says_where_it_starts_and_where_that_reaches():
    assert _video(start=24).trim_summary(120) == "from 24-144f (4.00s)"
    assert _audio(start=1.5).trim_summary(120) == "from 1.50-6.50s"


def test_a_row_with_no_frame_rate_says_only_the_frames():
    assert _video(start=24, fps=None).trim_summary(120) == "from 24-144f"


def test_a_row_that_runs_out_says_where_it_actually_ends():
    assert _video(start=250, frames=300).trim_summary(120) == "from 250-300f (1.67s)"


def test_a_row_marked_at_the_beginning_says_nothing():
    """Every reference is cut to the same length, so "0-120f" on every row would be
    noise on exactly the rows that have nothing to report."""
    assert _video().trim_summary(120) == ""
    assert _audio().trim_summary(120) == ""


def test_trim_limit_uses_what_was_probed():
    assert _video(frames=300).trim_limit() == 300
    assert _audio(duration=12.5).trim_limit() == 12.5
    assert RefRow(kind=VIDEO, comfy_name="a.mp4").trim_limit() == 0


# ---------------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------------

def _loader(built, index=0):
    return built.graph[str(ROLES.injected['video'] + ROLES.injected['video_stride'] * index)]["inputs"]


def test_an_untrimmed_video_starts_at_the_beginning_and_runs_the_generated_length():
    built = build_graph(BASE, BuildState(refs=RefSet(videos=[_video()]), duration_seconds=5.0), ROLES)
    loader = _loader(built)

    assert loader["skip_first_frames"] == 0
    assert loader["frame_load_cap"] == built.frames
    assert _minimax(built)["ref_videos.ref_video_0"] == [str(ROLES.injected['video']), 0]


def test_the_mark_moves_the_start_and_nothing_else():
    """The whole reason the chain is one node: the old one decoded the entire file."""
    refs = RefSet(videos=[_video(start=24, use_soundtrack=False)])
    built = build_graph(BASE, BuildState(refs=refs), ROLES)
    loader = _loader(built)

    assert built.graph[str(ROLES.injected['video'])]["class_type"] == "VHS_LoadVideo"
    assert loader["skip_first_frames"] == 24
    assert loader["frame_load_cap"] == built.frames


def test_the_cap_is_always_the_servers_own_length():
    """Linked rather than computed here, marked or not, so the two cannot disagree -- and
    so a duration change needs no rebuild of anything on this side."""
    for row in (_video(), _video(start=0), _video(start=5000)):
        built = build_graph(BASE, BuildState(refs=RefSet(videos=[row]), duration_seconds=9.0), ROLES)
        assert _loader(built)["frame_load_cap"] == built.frames


def test_the_frame_rate_is_never_forced():
    """The mark is a frame index into the source; resampling would move it."""
    built = build_graph(BASE, BuildState(refs=RefSet(videos=[_video(start=24)])), ROLES)
    loader = _loader(built)
    assert loader["force_rate"] == 0
    assert loader["select_every_nth"] == 1


def test_the_soundtrack_comes_off_the_same_node_already_windowed():
    """VHS derives the audio start and duration from the same two inputs, so the frames
    and the sound cannot drift apart -- and no frame rate is needed to line them up."""
    refs = RefSet(videos=[_video(start=30, fps=30.0, use_soundtrack=True)])
    built = build_graph(BASE, BuildState(refs=refs), ROLES)

    assert _minimax(built)["ref_video_audios.ref_video_audio_0"] == [str(ROLES.injected['video']), 2]


def test_a_soundtrack_needs_no_frame_rate_any_more():
    """It used to be blocked without one, because seconds had to be derived from frames."""
    row = _video(start=30, fps=None, use_soundtrack=True)
    errors, _ = row_warnings(row, target_frames=124)
    assert not any("frame rate is unknown" in e for e in errors)


def test_an_unchecked_soundtrack_omits_the_key(_unused=None):
    refs = RefSet(videos=[_video(use_soundtrack=False)])
    built = build_graph(BASE, BuildState(refs=refs), ROLES)
    assert "ref_video_audios.ref_video_audio_0" not in _minimax(built)


def test_a_trimmed_audio_reference_is_sliced_from_its_mark_for_the_generated_length():
    refs = RefSet(audios=[_audio(start=1.5)])
    built = build_graph(BASE, BuildState(refs=refs, duration_seconds=5.0), ROLES)
    load_id, trim_id = str(ROLES.injected['audio']), str(ROLES.injected['audio'] + 1)

    node = built.graph[trim_id]
    assert node["class_type"] == "TrimAudioDuration"
    assert node["inputs"]["audio"] == [load_id, 0]
    assert node["inputs"]["start_index"] == 1.5
    assert node["inputs"]["duration"] == round(FRAMES_5S / config.FPS, 3)
    assert _minimax(built)["ref_audios.ref_audio_0"] == [trim_id, 0]


def test_an_audio_sections_length_follows_the_duration_parameter():
    refs = RefSet(audios=[_audio(start=1.5)])
    built = build_graph(BASE, BuildState(refs=refs, duration_seconds=10.0), ROLES)
    duration = built.graph[str(ROLES.injected['audio'] + 1)]["inputs"]["duration"]
    assert duration == round(built.frames / config.FPS, 3)


def test_an_unmarked_audio_reference_is_cut_the_same_way_from_zero():
    """Unconditional, so a marked reference and an unmarked one build the same shape of
    graph -- and the truncation the model would do anyway is visible in the workflow."""
    built = build_graph(BASE, BuildState(refs=RefSet(audios=[_audio()]),
                                         duration_seconds=5.0), ROLES)
    trim_id = str(ROLES.injected['audio'] + 1)

    assert built.graph[trim_id]["class_type"] == "TrimAudioDuration"
    assert built.graph[trim_id]["inputs"]["start_index"] == 0
    assert built.graph[trim_id]["inputs"]["duration"] == round(FRAMES_5S / config.FPS, 3)
    assert _minimax(built)["ref_audios.ref_audio_0"] == [trim_id, 0]


def test_several_trimmed_audio_references_keep_their_own_nodes():
    refs = RefSet(audios=[_audio(f"a{j}.wav", start=j)
                          for j in range(config.MAX_REF_AUDIOS)])
    built = build_graph(BASE, BuildState(refs=refs), ROLES)
    for j in range(config.MAX_REF_AUDIOS):
        base = ROLES.injected['audio'] + ROLES.injected['audio_stride'] * j
        assert built.graph[str(base + 1)]["inputs"]["start_index"] == float(j)
        assert _minimax(built)[f"ref_audios.ref_audio_{j}"] == [str(base + 1), 0]


def test_trim_nodes_stay_inside_the_reserved_ids():
    refs = RefSet(
        videos=[_video(f"v{k}.mp4", start=1, use_soundtrack=True)
                for k in range(config.MAX_REF_VIDEOS)],
        audios=[_audio(f"a{j}.wav", start=0) for j in range(config.MAX_REF_AUDIOS)],
    )
    built = build_graph(BASE, BuildState(refs=refs), ROLES)
    injected = set(built.graph) - set(BASE)
    lowest = min(ROLES.injected[k] for k in ("image", "video", "audio"))
    assert injected and all(int(nid) >= lowest for nid in injected)
    assert not (injected & set(BASE))


def test_every_link_resolves_with_trims_in_play():
    refs = RefSet(videos=[_video(start=5, use_soundtrack=True)], audios=[_audio(start=1)])
    graph = build_graph(BASE, BuildState(refs=refs), ROLES).graph
    for node_id, node in graph.items():
        for key, value in node["inputs"].items():
            if isinstance(value, list) and len(value) == 2 and isinstance(value[0], str):
                assert value[0] in graph, f"{node_id}.{key} -> {value[0]}"


def test_labels_mention_where_the_section_starts_and_how_long_it_is():
    refs = RefSet(videos=[_video(start=24, use_soundtrack=False)])
    built = build_graph(BASE, BuildState(refs=refs, duration_seconds=5.0), ROLES)
    label = built.node_label(str(ROLES.injected['video']))
    assert "from 24" in label and f"{FRAMES_5S} frames" in label


# ---------------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------------

def test_a_trim_says_what_will_be_sent():
    _, warnings = row_warnings(_video(start=24), target_frames=124)
    assert any("frames 24 to 148" in w for w in warnings)

    _, warnings = row_warnings(_audio(start=1.0), target_frames=120)
    assert any("1.00s to 6.00s" in w for w in warnings)


def test_a_reference_that_runs_out_after_the_mark_says_so():
    """Not an error -- the model accepts a short reference -- but the fix is to start
    earlier or generate less, and neither is obvious from a window that simply stops."""
    _, warnings = row_warnings(_video(start=250, frames=300), target_frames=124)
    assert any("runs out before" in w and "50 frames" in w for w in warnings)


def test_a_mark_close_to_the_end_leaves_too_little_to_generate_from():
    errors, _ = row_warnings(_video(start=298, frames=300), target_frames=124)
    assert any("only 2 frames" in e for e in errors)


def test_a_long_reference_is_still_reported_as_truncated():
    _, warnings = row_warnings(_video(start=0, frames=300), target_frames=124)
    assert any("truncated" in w for w in warnings)


def test_the_same_video_without_its_soundtrack_is_fine():
    row = _video(start=30, fps=None, use_soundtrack=False)
    errors, _ = row_warnings(row, target_frames=124)
    assert not any("frame rate is unknown" in e for e in errors)


# ---------------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------------

def test_a_mark_survives_a_settings_round_trip():
    row = RefRow(kind=AUDIO, local_path="C:/refs/a.wav")
    row.trim_start = 1.25
    restored = RefSet.from_list([row.to_dict()]).audios[0]
    assert restored.trim_start == 1.25


def test_the_out_point_and_the_switch_an_older_version_wrote_are_dropped():
    """The length comes from the duration parameter and the cut is unconditional, so the
    start is the only part of that trio still worth keeping."""
    restored = RefSet.from_list([{
        "kind": VIDEO, "local_path": "C:/a.mp4",
        "trim_enabled": False, "trim_start": 48, "trim_end": 168,
    }]).videos[0]

    assert restored.trim_start == 48
    assert restored.marked is True
    assert restored.trim_end(124) == 172


def test_a_mark_on_an_image_is_dropped_on_load():
    restored = RefSet.from_list(
        [{"kind": IMAGE, "local_path": "C:/a.png", "trim_start": 5}])
    assert restored.images[0].trim_start == 0.0
    assert restored.images[0].marked is False


@pytest.mark.parametrize("bad", ["abc", None, [1]])
def test_a_corrupt_mark_falls_back_to_the_beginning_rather_than_raising(bad):
    restored = RefSet.from_list(
        [{"kind": AUDIO, "local_path": "C:/a.wav", "trim_start": bad}])
    assert restored.audios[0].trim_start == 0.0
