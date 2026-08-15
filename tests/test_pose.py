"""Pose references: a skeleton of the clip is sent instead of the clip.

Nothing here loads a model or decodes anything. What is under test is the part that is
easy to get wrong and expensive to discover: which section gets rendered, when a cached
render stops describing it, and that the substitution happens on the job snapshot rather
than on the row the user is looking at.
"""

from __future__ import annotations

import importlib.util
import math
import sys
from pathlib import Path

import pytest

#: numpy arrives with the estimator, so the keypoint tests skip on a machine that has not
#: installed it yet. Everything else here is plain Python and always runs.
needs_numpy = pytest.mark.skipif(importlib.util.find_spec("numpy") is None,
                                 reason="numpy comes with the pose dependencies")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmon3 import config, graph_builder, mathmirror, pose, posefigure, scenes  # noqa: E402
from harmon3 import settings as settings_mod                          # noqa: E402
from harmon3.graph_builder import BuildState, build_graph             # noqa: E402
from harmon3.jobs import JobRequest                                   # noqa: E402
from harmon3.refs import AUDIO, IMAGE, VIDEO, RefRow, RefSet, row_warnings  # noqa: E402

WORKFLOW = config.load_workflow()
BASE, ROLES = WORKFLOW.graph, WORKFLOW.roles
SETTINGS = pose.PoseSettings()


def _video(name="clip.mp4", *, start=0, use_pose=True, local="D:/refs/clip.mp4"):
    row = RefRow(kind=VIDEO, local_path=local, comfy_name=name)
    row.fps, row.frame_count = 24.0, 600
    row.trim_start = start
    row.use_pose = use_pose
    return row


# ---------------------------------------------------------------------------------
# What the flag means
# ---------------------------------------------------------------------------------

def test_only_a_video_has_a_skeleton_to_extract():
    assert RefRow(kind=VIDEO, comfy_name="a.mp4").supports_pose is True
    assert RefRow(kind=AUDIO, comfy_name="a.wav").supports_pose is False
    assert RefRow(kind=IMAGE, comfy_name="a.png").supports_pose is False


def test_a_pose_flag_on_anything_else_is_ignored():
    row = RefRow(kind=AUDIO, comfy_name="a.wav")
    row.use_pose = True
    assert row.poses is False


def test_a_pose_flag_on_an_image_is_dropped_on_load():
    restored = RefSet.from_list(
        [{"kind": IMAGE, "local_path": "C:/a.png", "use_pose": True}])
    assert restored.images[0].use_pose is False


def test_the_flag_survives_a_settings_round_trip():
    restored = RefSet.from_list([_video().to_dict()]).videos[0]
    assert restored.use_pose is True


def test_the_rendered_path_is_never_persisted():
    """It points into a cache keyed by content, mark and length -- all of which may have
    moved by the time a scene is loaded again. Re-deriving costs one hash."""
    row = _video()
    row.pose_path = "D:/runs/pose/pose_clip_abc123456789.mp4"
    assert "pose_path" not in row.to_dict()
    assert RefSet.from_list([row.to_dict()]).videos[0].pose_path is None


# ---------------------------------------------------------------------------------
# Which section gets rendered
# ---------------------------------------------------------------------------------

def test_the_section_is_the_mark_and_the_generated_length():
    """Only the frames that will actually be sent are posed. A 226-second reference is
    6,777 frames and a five-second generation needs 124 of them."""
    assert pose.section_for(_video(start=1356), 5.0) == (1356, mathmirror.frames_from_seconds(5.0))
    assert pose.section_for(_video(start=0), 10.0) == (0, mathmirror.frames_from_seconds(10.0))


def test_the_cache_key_moves_with_everything_that_changes_the_frames():
    base = pose.cache_key("digest", SETTINGS, 100, 124)
    assert pose.cache_key("digest", SETTINGS, 101, 124) != base       # a different mark
    assert pose.cache_key("digest", SETTINGS, 100, 248) != base       # a different length
    assert pose.cache_key("other", SETTINGS, 100, 124) != base        # different content
    other_model = pose.PoseSettings(model="vitpose-b")
    assert pose.cache_key("digest", other_model, 100, 124) != base
    coarser = pose.PoseSettings(kpt_thr=0.8)
    assert pose.cache_key("digest", coarser, 100, 124) != base
    other_style = pose.PoseSettings(style="openpose-torso")
    assert pose.cache_key("digest", other_style, 100, 124) != base
    painted = pose.PoseSettings(style=posefigure.STYLE)
    styles = (SETTINGS, other_style, painted)
    assert len({pose.cache_key("digest", s, 100, 124) for s in styles}) == 3


def test_the_cache_key_ignores_where_it_ran():
    """CPU and CUDA draw the same skeleton, so a clip rendered on one is not stale on
    the other."""
    on_cpu = pose.PoseSettings(runtime="cpu")
    on_cuda = pose.PoseSettings(runtime="cuda")
    assert pose.cache_key("d", on_cpu, 0, 124) == pose.cache_key("d", on_cuda, 0, 124)


# ---------------------------------------------------------------------------------
# Clearing what has been rendered
# ---------------------------------------------------------------------------------

@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "POSE_CACHE_DIR", tmp_path / "pose")
    (tmp_path / "pose").mkdir()
    return tmp_path / "pose"


def test_an_empty_cache_reports_nothing_rather_than_failing(tmp_path, monkeypatch):
    """The folder does not exist until the first render, and the panel asks before then."""
    monkeypatch.setattr(config, "POSE_CACHE_DIR", tmp_path / "never-made")
    assert pose.cached_clips() == []
    assert pose.cache_usage() == (0, 0)
    assert pose.clear_cache() == (0, 0, [])


def test_the_cache_counts_the_clips_and_their_bytes(cache_dir):
    (cache_dir / "pose_clip_a1b2c3.mp4").write_bytes(b"x" * 100)
    (cache_dir / "pose_clip_d4e5f6.mp4").write_bytes(b"x" * 50)
    assert pose.cache_usage() == (2, 150)


def test_an_abandoned_part_file_is_swept_up_too(cache_dir):
    """A cancelled render leaves one behind, and nothing will ever come looking for it."""
    (cache_dir / "pose_clip_a1b2c3.part.mp4").write_bytes(b"x" * 10)
    assert len(pose.cached_clips()) == 1

    removed, freed, failures = pose.clear_cache()
    assert (removed, freed, failures) == (1, 10, [])
    assert list(cache_dir.iterdir()) == []


def test_clearing_leaves_anything_that_is_not_a_pose_clip(cache_dir):
    """The folder is under runs/, and a stray file there is not ours to delete."""
    (cache_dir / "pose_clip_a1b2c3.mp4").write_bytes(b"x" * 10)
    keep = cache_dir / "notes.txt"
    keep.write_text("mine")

    removed, _freed, _failures = pose.clear_cache()
    assert removed == 1
    assert keep.is_file()


def test_clearing_forgets_the_rows_pointing_at_what_was_deleted():
    """A row still pointing at a deleted clip would offer a thumbnail for a missing file."""
    row = _video()
    row.pose_path, row.pose_section = "D:/runs/pose/pose_clip_a1b2c3.mp4", (0, 124)
    state = BuildState(refs=RefSet(videos=[row]))

    pose.forget_all(state)
    assert row.pose_path is None
    assert row.pose_section is None


def test_the_cached_file_is_named_after_its_source():
    """A node error is matched back to a row by looking for the row's display name in the
    node label, and the label is built from the filename."""
    path = pose.cached_path("a" * 64, "Womanizer - dance tease (480p).mp4")
    assert path.name.startswith("pose_Womanizer_-_dance_tease_480p_")
    assert path.name.endswith("_aaaaaaaaaaaa.mp4")
    assert path.parent == config.POSE_CACHE_DIR


def test_an_awkward_filename_still_produces_a_usable_name():
    path = pose.cached_path("f" * 64, "**/?.mp4")
    assert path.name == "pose_ref_ffffffffffff.mp4"


# ---------------------------------------------------------------------------------
# Staleness
# ---------------------------------------------------------------------------------

def _rendered(tmp_path, row, duration=5.0):
    clip = tmp_path / "pose.mp4"
    clip.write_bytes(b"a rendered skeleton")
    row.pose_path = str(clip)
    row.pose_section = pose.section_for(row, duration)
    return clip


def test_a_render_that_still_matches_is_kept(tmp_path):
    row = _video(start=100)
    _rendered(tmp_path, row)
    assert pose.forget_stale(row, 5.0) is False
    assert row.pose_path is not None


def test_moving_the_mark_makes_a_render_stale(tmp_path):
    row = _video(start=100)
    _rendered(tmp_path, row)
    row.trim_start = 200
    assert pose.forget_stale(row, 5.0) is True
    assert row.pose_path is None and row.pose_section is None


def test_changing_the_duration_makes_a_render_stale(tmp_path):
    row = _video(start=100)
    _rendered(tmp_path, row)
    assert pose.forget_stale(row, 9.0) is True
    assert row.pose_path is None


def test_a_render_that_has_been_deleted_is_forgotten(tmp_path):
    row = _video(start=100)
    clip = _rendered(tmp_path, row)
    clip.unlink()
    assert pose.forget_stale(row, 5.0) is True


def test_a_row_with_no_render_has_nothing_to_forget():
    assert pose.forget_stale(_video(), 5.0) is False


# ---------------------------------------------------------------------------------
# The substitution
# ---------------------------------------------------------------------------------

def test_the_skeleton_replaces_the_clip_and_the_mark_goes_with_it():
    """The rendered clip *is* the section, so applying the mark again on the server would
    take a section of a section."""
    row = _video(start=1356)
    state = BuildState(refs=RefSet(videos=[row]))
    pose.swap_in(state, {row.uid: "D:/runs/pose/pose_clip_abc123456789.mp4"})

    assert row.local_path == "D:/runs/pose/pose_clip_abc123456789.mp4"
    assert row.trim_start == 0.0
    assert row.comfy_name is None      # a different file needs its own upload


def test_a_row_with_no_render_is_left_alone():
    row = _video(start=1356)
    state = BuildState(refs=RefSet(videos=[row]))
    pose.swap_in(state, {})
    assert row.local_path == "D:/refs/clip.mp4"
    assert row.trim_start == 1356


def test_the_substitution_never_reaches_the_row_on_screen():
    """The same isolation the upload path relies on: the worker rewrites a deep copy, so
    settings and scenes keep recording the user's own file at the user's own mark."""
    row = _video(start=1356)
    state = BuildState(refs=RefSet(videos=[row]))
    request = JobRequest.snapshot(state, {})

    pose.swap_in(request.state, {request.state.refs.videos[0].uid: "D:/runs/pose/p.mp4"})

    assert row.local_path == "D:/refs/clip.mp4"
    assert row.trim_start == 1356
    assert request.state.refs.videos[0].local_path == "D:/runs/pose/p.mp4"


def test_a_posed_row_builds_the_same_graph_shape_as_any_other():
    row = _video(start=1356, name="pose_clip_abc.mp4")
    state = BuildState(refs=RefSet(videos=[row]), duration_seconds=5.0)
    pose.swap_in(state, {row.uid: "D:/runs/pose/pose_clip_abc.mp4"})
    row.comfy_name = "harmon3/pose_clip_abc.mp4"      # as the upload would leave it

    built = build_graph(BASE, state, ROLES)
    loader = built.graph[str(ROLES.injected['video'])]["inputs"]

    assert loader["video"] == "harmon3/pose_clip_abc.mp4"
    assert loader["skip_first_frames"] == 0
    assert loader["frame_load_cap"] == built.frames


def test_a_posed_row_keeps_its_tags():
    """Pose swaps which file is sent, not how many references there are."""
    row = _video()
    built = build_graph(BASE, BuildState(refs=RefSet(videos=[row])), ROLES)
    inputs = built.graph[ROLES.reference]["inputs"]

    assert inputs["ref_videos.ref_video_0"] == [str(ROLES.injected['video']), 0]
    assert inputs["ref_video_audios.ref_video_audio_0"] == [str(ROLES.injected['video']), 2]
    assert built.tags.tag_for(row) == "<Video 1>"


# ---------------------------------------------------------------------------------
# Keypoints
# ---------------------------------------------------------------------------------

@needs_numpy
def test_the_neck_is_the_midpoint_of_the_shoulders():
    """OpenPose has one and COCO does not, and eighteen keypoints is what selects the
    human skeleton when drawing -- seventeen selects an animal."""
    import numpy as np

    keypoints = np.zeros((1, 17, 2), dtype=np.float32)
    scores = np.ones((1, 17), dtype=np.float32)
    keypoints[0, 5] = (100.0, 50.0)      # left shoulder
    keypoints[0, 6] = (60.0, 70.0)       # right shoulder

    converted, converted_scores = pose.coco17_to_openpose18(keypoints, scores)

    assert converted.shape == (1, 18, 2)
    assert tuple(converted[0, 1]) == (80.0, 60.0)
    assert converted_scores.shape == (1, 18)


@needs_numpy
def test_a_guessed_shoulder_makes_a_guessed_neck():
    import numpy as np

    keypoints = np.zeros((1, 17, 2), dtype=np.float32)
    scores = np.ones((1, 17), dtype=np.float32)
    scores[0, 6] = 0.1

    _converted, converted_scores = pose.coco17_to_openpose18(keypoints, scores)
    assert converted_scores[0, 1] == pytest.approx(0.1)


@needs_numpy
def test_the_nose_and_the_hips_land_where_openpose_expects_them():
    import numpy as np

    keypoints = np.zeros((1, 17, 2), dtype=np.float32)
    scores = np.ones((1, 17), dtype=np.float32)
    keypoints[0, 0] = (1.0, 1.0)         # nose      -> 0
    keypoints[0, 11] = (2.0, 2.0)        # left hip  -> 11
    keypoints[0, 12] = (3.0, 3.0)        # right hip -> 8

    converted, _scores = pose.coco17_to_openpose18(keypoints, scores)
    assert tuple(converted[0, 0]) == (1.0, 1.0)
    assert tuple(converted[0, 11]) == (2.0, 2.0)
    assert tuple(converted[0, 8]) == (3.0, 3.0)


@needs_numpy
def test_the_wrong_number_of_keypoints_is_refused():
    import numpy as np

    with pytest.raises(pose.PoseError):
        pose.coco17_to_openpose18(np.zeros((1, 133, 2)), np.zeros((1, 133)))


@needs_numpy
def test_wholebody_keeps_its_face_and_hands_one_slot_along():
    """134 is what draw_skeleton accepts for a wholebody OpenPose figure; 133 raises.

    Past the body the two layouts are the same list displaced by the neck at index 1, so
    the face and hands have to arrive shifted rather than dropped -- the whole point of
    choosing this model.
    """
    import numpy as np

    keypoints = np.zeros((1, 133, 2), dtype=np.float32)
    scores = np.ones((1, 133), dtype=np.float32)
    keypoints[0, 5] = (100.0, 50.0)      # left shoulder
    keypoints[0, 6] = (60.0, 70.0)       # right shoulder
    keypoints[0, 23] = (7.0, 7.0)        # first face point
    keypoints[0, 132] = (9.0, 9.0)       # last right-hand point

    converted, converted_scores = pose.coco133_to_openpose134(keypoints, scores)

    assert converted.shape == (1, 134, 2)
    assert converted_scores.shape == (1, 134)
    assert tuple(converted[0, 1]) == (80.0, 60.0)        # the neck, as for COCO-17
    assert tuple(converted[0, 24]) == (7.0, 7.0)
    assert tuple(converted[0, 133]) == (9.0, 9.0)


# ---------------------------------------------------------------------------------
# Styles. What is joined to what, which is the drawing's business rather than the
# estimator's -- so every style has to work with every model.
# ---------------------------------------------------------------------------------

needs_rtmlib = pytest.mark.skipif(importlib.util.find_spec("rtmlib") is None,
                                  reason="rtmlib comes with the pose dependencies")


@needs_rtmlib
def test_the_standard_style_uses_rtmlibs_own_table():
    """None, not a copy: the default path stays exactly the code it always was."""
    assert pose.skeleton_for(18, config.DEFAULT_POSE_STYLE) is None
    assert pose.skeleton_for(134, config.DEFAULT_POSE_STYLE) is None


@needs_rtmlib
@pytest.mark.parametrize("keypoints", [18, 134])
def test_the_torso_style_hangs_each_hip_off_its_own_shoulder(keypoints):
    _info, links = pose.skeleton_for(keypoints, "openpose-torso")
    drawn = {tuple(link["link"]) for link in links.values()}

    assert ("right_shoulder", "right_hip") in drawn
    assert ("left_shoulder", "left_hip") in drawn
    assert ("neck", "right_hip") not in drawn
    assert ("neck", "left_hip") not in drawn


@needs_rtmlib
@pytest.mark.parametrize("keypoints", [18, 134])
def test_the_torso_style_changes_nothing_else(keypoints):
    from rtmlib.visualization import skeleton as skeleton_mod

    name = {18: "openpose18", 134: "openpose134"}[keypoints]
    stock = skeleton_mod.__dict__[name]["skeleton_info"]
    _info, links = pose.skeleton_for(keypoints, "openpose-torso")

    assert set(links) == set(stock)
    changed = [i for i in links if tuple(links[i]["link"]) != tuple(stock[i]["link"])]
    assert len(changed) == 2
    assert all(links[i]["color"] == stock[i]["color"] for i in links)


@needs_rtmlib
@pytest.mark.parametrize("keypoints", [18, 134])
def test_the_swapped_links_keep_their_place_among_the_body_parts(keypoints):
    """draw_openpose draws links 0-16 as fat limbs and the rest as hairlines, so a torso
    link appended at the end would come out thin beside the leg it joins."""
    from rtmlib.visualization import skeleton as skeleton_mod

    name = {18: "openpose18", 134: "openpose134"}[keypoints]
    stock = skeleton_mod.__dict__[name]["skeleton_info"]
    _info, links = pose.skeleton_for(keypoints, "openpose-torso")

    changed = [i for i in links if tuple(links[i]["link"]) != tuple(stock[i]["link"])]
    assert all(index <= 16 for index in changed)


@needs_rtmlib
@needs_numpy
@pytest.mark.parametrize("keypoints", [18, 134])
def test_the_torso_style_closes_the_pelvis(keypoints):
    """The hips are joined to each other as well as upward, or the legs read as two
    separate things rather than as one body."""
    import numpy as np

    _info, links = pose.skeleton_for(keypoints, "openpose-torso")
    added = {pair for pair, _colour in pose._ADDED_LINKS["openpose-torso"]}
    assert ("right_hip", "left_hip") in added
    # And it is drawn separately rather than smuggled into the table, where its index
    # would put it past the body parts and turn it into a hairline.
    assert ("right_hip", "left_hip") not in {tuple(v["link"]) for v in links.values()}

    canvas = np.zeros((80, 80, 3), np.uint8)
    points = np.zeros((1, keypoints, 2), np.float32)
    scores = np.ones((1, keypoints), np.float32)
    points[0, 8] = (30.0, 50.0)          # right hip
    points[0, 11] = (50.0, 50.0)         # left hip
    drawn = pose._draw_limb(canvas, points[0], scores[0], 8, 11, [0, 255, 140],
                            kpt_thr=0.4, line_width=4)
    assert drawn[50, 40].tolist() != [0, 0, 0]


@needs_rtmlib
@needs_numpy
def test_an_unconfident_or_offscreen_pelvis_is_not_drawn():
    """The same guards draw_openpose applies: a bar to a point clamped at the edge of the
    frame is worse than no bar."""
    import numpy as np

    canvas = np.zeros((80, 80, 3), np.uint8)
    points = np.zeros((1, 18, 2), np.float32)
    points[0, 8], points[0, 11] = (30.0, 50.0), (50.0, 50.0)

    unsure = np.ones((1, 18), np.float32)
    unsure[0, 11] = 0.1
    assert pose._draw_limb(canvas, points[0], unsure[0], 8, 11, [0, 255, 140],
                           kpt_thr=0.4, line_width=4).sum() == 0

    offscreen = points.copy()
    offscreen[0, 11] = (500.0, 50.0)
    assert pose._draw_limb(canvas, offscreen[0], np.ones(18, np.float32), 8, 11,
                           [0, 255, 140], kpt_thr=0.4, line_width=4).sum() == 0


@needs_rtmlib
@needs_numpy
@pytest.mark.parametrize("keypoints", [18, 134])
def test_every_style_draws_every_layout(keypoints):
    """The style is a property of the drawing, so it has to work with whichever model."""
    import numpy as np

    points = np.zeros((1, keypoints, 2), np.float32)
    for index in range(keypoints):
        points[0, index] = (40.0 + index % 7, 30.0 + index % 11)
    scores = np.ones((1, keypoints), np.float32)

    for name in config.POSE_STYLES:
        canvas = pose.draw(np.zeros((120, 120, 3), np.uint8), points, scores,
                           pose.PoseSettings(style=name), radius=3, line_width=4)
        assert canvas.sum() > 0


@needs_rtmlib
def test_a_style_is_refused_for_a_layout_it_cannot_draw():
    with pytest.raises(pose.PoseError):
        pose.skeleton_for(17, "openpose-torso")


def test_the_style_survives_the_settings_round_trip():
    """Applies to whichever model is selected, so it is stored beside them, not inside."""
    chosen = settings_mod.pose_settings({"pose_style": "openpose-torso",
                                         "pose_model": "vitpose-b"})
    assert chosen.style == "openpose-torso"
    assert chosen.model == "vitpose-b"


def test_an_unknown_style_falls_back_rather_than_failing_a_render():
    assert settings_mod.pose_settings({"pose_style": "spaghetti"}).style \
        == config.DEFAULT_POSE_STYLE
    assert settings_mod.pose_settings({}).style == config.DEFAULT_POSE_STYLE


def test_the_painted_style_survives_the_settings_round_trip():
    chosen = settings_mod.pose_settings({"pose_style": posefigure.STYLE})
    assert chosen.style == posefigure.STYLE


def test_every_pose_style_is_labelled_in_the_settings():
    """A style with no entry appears in the list by its bare id, which is not a crash and
    is not something anyone would choose to ship either."""
    labels = pytest.importorskip("harmon3.ui.settings_panel").POSE_STYLE_LABELS
    assert set(config.POSE_STYLES) <= set(labels)


# ---------------------------------------------------------------------------------
# Which way the subject is facing. Front and back are very nearly mirrors of one
# another in two dimensions, and the estimator will place a full face on the back of a
# head without hesitating -- so this is the one thing the figure style has to get right.
# ---------------------------------------------------------------------------------

def _standing(*, back=False, lying=False):
    """A plausible upright OpenPose-18 figure as plain tuples, mirrored when facing away.

    Plain Python rather than numpy on purpose: Facing does no array work, so its tests
    should run on a machine that has never installed the pose dependencies.
    """
    side = -1 if back else 1
    spec = {0: (200, 92), 1: (200, 120),
            2: (200 - 42 * side, 132), 5: (200 + 42 * side, 132),
            3: (200 - 74 * side, 196), 6: (200 + 74 * side, 196),
            4: (200 - 96 * side, 258), 7: (200 + 96 * side, 258),
            8: (200 - 26 * side, 236), 11: (200 + 26 * side, 236),
            9: (200 - 30 * side, 322), 12: (200 + 30 * side, 322),
            10: (200 - 32 * side, 404), 13: (200 + 32 * side, 404),
            14: (200 - 11 * side, 88), 15: (200 + 11 * side, 88),
            16: (200 - 24 * side, 94), 17: (200 + 24 * side, 94)}
    points = [(float(spec[i][0]), float(spec[i][1])) for i in range(18)]
    if lying:                                    # a quarter turn about the middle of it
        points = [(200.0 - (y - 250.0), 250.0 + (x - 200.0)) for x, y in points]
    return points


def _sure(count=18):
    return [1.0] * count


def _settled(points, scores=None, *, frames=20, kpt_thr=0.4):
    facing = posefigure.Facing()
    for _ in range(frames):
        facing.update(points, scores or _sure(len(points)), kpt_thr)
    return facing


def test_the_left_shoulder_on_the_right_reads_as_facing_the_camera():
    facing = _settled(_standing())
    assert facing.frontal is True
    assert facing.score > 0


def test_crossed_shoulders_read_as_facing_away():
    facing = _settled(_standing(back=True))
    assert facing.frontal is False
    assert facing.score < 0


def test_a_figure_lying_down_is_still_read_correctly():
    """Read off image x this would be a coin toss. The cues are projected onto the body's
    own lateral axis instead, which costs three lines and is right at any angle."""
    assert _settled(_standing(lying=True)).frontal is True
    assert _settled(_standing(back=True, lying=True)).frontal is False


def test_a_profile_holds_whichever_way_it_was_last_facing():
    """Near profile the geometry genuinely says nothing, and a figure whose face flickered
    on and off through every turn would be worse to watch than one that waits."""
    edge = [(200.0, y) for _x, y in _standing()]          # every point on the midline
    for started_frontal in (True, False):
        facing = posefigure.Facing(smoothed=0.5 if started_frontal else -0.5,
                                   frontal=started_frontal)
        for _ in range(20):
            facing.update(edge, _sure(), 0.4)
        assert facing.frontal is started_frontal


def test_the_first_usable_frame_seeds_rather_than_easing_in():
    """Easing up from zero would open every back-facing clip with a third of a second of
    face drawn on the back of a head."""
    facing = posefigure.Facing()
    facing.update(_standing(back=True), _sure(), 0.4)
    assert facing.frontal is False


def test_one_disagreeing_frame_does_not_flip_the_figure_round():
    facing = _settled(_standing(back=True))
    facing.update(_standing(), _sure(), 0.4)
    assert facing.frontal is False


def test_a_cue_below_the_threshold_does_not_get_a_vote():
    """The shoulders carry the most weight, so an unconfident pair saying the wrong thing
    is the case that matters: the hips and the ears have to be able to carry it alone."""
    points = _standing()
    for shoulder in (2, 5):                      # mirrored, and not believed
        points[shoulder] = (400.0 - points[shoulder][0], points[shoulder][1])
    scores = _sure()
    scores[2] = scores[5] = 0.1
    assert _settled(points, scores).frontal is True


def test_a_frame_with_nothing_confident_leaves_the_estimate_where_it_was():
    """Not nudged toward zero: absence of evidence is not evidence of a profile. This is
    also what a held frame gets, since a held frame never reaches here at all."""
    facing = posefigure.Facing(smoothed=-0.8, frontal=False)
    facing.update(_standing(), [0.1] * 18, 0.4)
    assert (facing.smoothed, facing.frontal) == (-0.8, False)


# ---------------------------------------------------------------------------------
# Painting the figure. rtmlib's own drawing hardcodes a 3px joint and a 2px face
# whatever the frame size, and gives the feet a colour of black, so this style paints
# its own -- which means the parts it adds are worth an assertion each.
# ---------------------------------------------------------------------------------

needs_cv2 = pytest.mark.skipif(importlib.util.find_spec("cv2") is None,
                               reason="opencv comes with the pose dependencies")

FIGURE = pose.PoseSettings(style=posefigure.STYLE)


def _drawable(count=18, *, back=False):
    """The standing figure as arrays, with the wholebody extras when the layout has them."""
    import numpy as np

    keypoints = np.zeros((1, count, 2), np.float32)
    scores = np.ones((1, count), np.float32)
    keypoints[0, :18] = _standing(back=back)

    if count > posefigure.R_HEEL:
        for big, small, heel, ankle in ((18, 19, 20, 13), (21, 22, 23, 10)):
            x = float(keypoints[0, ankle][0])
            keypoints[0, big] = (x + 10, 430.0)
            keypoints[0, small] = (x - 8, 432.0)
            keypoints[0, heel] = (x, 408.0)
    if count > posefigure.FACE_STOP:
        # A face with the proportions of one: the jaw is what sizes the head, so a ring of
        # evenly spaced points would give a head too small to draw any features on.
        for offset, index in enumerate(posefigure.JAW):
            angle = math.pi * (0.15 + 0.70 * offset / 16)
            keypoints[0, index] = (200 - 24 * math.cos(angle), 90 + 24 * math.sin(angle))
        for start, group in ((186.0, range(41, 46)), (202.0, range(46, 51))):
            for offset, index in enumerate(group):
                keypoints[0, index] = (start + offset * 3, 82.0)
        for offset, index in enumerate(range(51, 55)):
            keypoints[0, index] = (200.0, 88 + offset * 3)
        for offset, index in enumerate(range(55, 60)):
            keypoints[0, index] = (194 + offset * 3, 101.0)
        for centre, group in ((189, range(60, 66)), (211, range(66, 72))):
            for offset, index in enumerate(group):
                keypoints[0, index] = (centre + 3 * math.cos(offset * 1.05),
                                       89 + 2 * math.sin(offset * 1.05))
        for step, group in ((0.52, range(72, 84)), (0.78, range(84, 92))):
            for offset, index in enumerate(group):
                keypoints[0, index] = (200 + 7 * math.cos(offset * step),
                                       109 + 3 * math.sin(offset * step))
        for root, wrist in ((posefigure.L_HAND_ROOT, 7), (posefigure.R_HAND_ROOT, 4)):
            base = keypoints[0, wrist]
            for finger in range(21):
                keypoints[0, root + finger] = (base[0] + finger % 5 * 4,
                                               base[1] + finger // 5 * 5)
        if back:
            scores[0, posefigure.FACE_FIRST:posefigure.FACE_STOP] = 0.15
    return keypoints, scores


def _paint(keypoints, scores, facing, *, size=(480, 400)):
    import numpy as np

    return pose.draw(np.zeros((size[0], size[1], 3), np.uint8), keypoints, scores,
                     FIGURE, radius=4, line_width=6, facing=facing)


@needs_cv2
@needs_numpy
def test_the_face_is_only_drawn_when_the_figure_faces_the_camera():
    """The estimator regresses face keypoints rather than detecting them, so it will hand
    back a full set for the back of a head. Gated on the verdict, not on their confidence."""
    import numpy as np

    def features(back):
        keypoints, scores = _drawable(134, back=back)
        painted = _paint(keypoints, scores,
                         posefigure.Facing(smoothed=-0.9 if back else 0.9, frontal=not back))
        return int((painted == np.array(posefigure.FACE, np.uint8)).all(axis=2).sum())

    assert features(back=False) > 0
    assert features(back=True) == 0


@needs_cv2
@needs_numpy
def test_the_feet_are_drawn_now_that_the_layout_has_them():
    """openpose134 gives the six foot keypoints color=[0, 0, 0] and draw_openpose skips
    any joint whose colour sums to zero, so nothing has ever drawn them."""
    def below_the_ankles(count):
        keypoints, scores = _drawable(count)
        return int((_paint(keypoints, scores, posefigure.Facing())[420:] > 0).sum())

    assert below_the_ankles(134) > 0
    assert below_the_ankles(18) == 0


@needs_cv2
@needs_numpy
@pytest.mark.parametrize("count", [18, 134])
def test_the_painted_style_draws_a_whole_figure(count):
    """The shared crash guard draws a scribble, which proves nothing about a real pose."""
    keypoints, scores = _drawable(count)
    painted = _paint(keypoints, scores, posefigure.Facing())
    lit = (painted.sum(axis=2) > 0)
    assert lit[80:110, 180:220].any()            # a head
    assert lit[150:220, 180:220].any()           # a trunk
    assert lit[300:360, 150:250].any()           # legs


@needs_cv2
@needs_numpy
def test_the_two_sides_of_the_figure_are_told_apart_by_colour():
    """Warm on the subject's right against cool on their left, which is what makes the
    arrangement readable at a glance. OpenPose's own palette is a hue sweep by link index
    and does not separate the sides at all."""
    import numpy as np

    keypoints, scores = _drawable(134)
    painted = _paint(keypoints, scores, posefigure.Facing(smoothed=0.9, frontal=True))
    warm = (painted == np.array(posefigure.R_UPPER_ARM, np.uint8)).all(axis=2)
    cool = (painted == np.array(posefigure.L_UPPER_ARM, np.uint8)).all(axis=2)
    # Facing the camera, the subject's right arm is on the left of the frame.
    assert warm[:, :200].sum() > warm[:, 200:].sum()
    assert cool[:, 200:].sum() > cool[:, :200].sum()


@needs_cv2
@needs_numpy
def test_a_limb_whose_end_is_unconfident_or_offscreen_is_still_skipped():
    """The guards moved down into posefigure.bone with the geometry, so they are worth
    asserting at the new entry point as well as through _draw_limb."""
    import numpy as np

    canvas = np.zeros((80, 80, 3), np.uint8)
    points = np.array([(30.0, 50.0), (50.0, 50.0)], np.float32)

    assert posefigure.bone(canvas, points, np.array([1.0, 0.1], np.float32), 0, 1,
                           posefigure.SPINE, kpt_thr=0.4, line_width=4).sum() == 0
    offscreen = np.array([(30.0, 50.0), (500.0, 50.0)], np.float32)
    assert posefigure.bone(canvas, offscreen, np.array([1.0, 1.0], np.float32), 0, 1,
                           posefigure.SPINE, kpt_thr=0.4, line_width=4).sum() == 0
    assert posefigure.bone(canvas, points, np.array([1.0, 1.0], np.float32), 0, 1,
                           posefigure.SPINE, kpt_thr=0.4, line_width=4).sum() > 0


@needs_cv2
@needs_numpy
def test_a_layout_too_small_to_be_a_figure_is_refused():
    """The same courtesy skeleton_for does for a table it has not got: fail where the
    style is chosen rather than several minutes into a render."""
    import numpy as np

    with pytest.raises(pose.PoseError):
        pose.draw(np.zeros((40, 40, 3), np.uint8), np.zeros((1, 17, 2), np.float32),
                  np.ones((1, 17), np.float32), FIGURE, radius=2, line_width=2)


@needs_numpy
def test_a_layout_with_no_conversion_is_refused():
    import numpy as np

    with pytest.raises(pose.PoseError):
        pose.coco133_to_openpose134(np.zeros((1, 17, 2)), np.zeros((1, 17)))


# ---------------------------------------------------------------------------------
# Encoding constraints
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [(480, 480), (855, 854), (1, 0), (0, 0)])
def test_dimensions_are_rounded_down_to_even(value, expected):
    """H.264 in yuv420p refuses odd dimensions, and a reference that fails to encode
    after two minutes of estimation is a poor way to find that out."""
    assert pose.even(value) == expected


def test_a_render_that_stops_part_way_leaves_nothing_behind(tmp_path, monkeypatch):
    """The dangerous failure: a truncated clip sitting at the name the cache looks for
    would be picked up as finished, and the next run would send six frames of skeleton
    without saying anything about it."""
    destination = tmp_path / "pose_clip_abc123456789.mp4"

    class Boom(Exception):
        pass

    def explode(*_args, **_kwargs):
        # Stand in for a cancel: the frame loop raises once it is asked to stop.
        destination.with_suffix(".part.mp4").write_bytes(b"half a clip")
        raise Boom("stopped")

    monkeypatch.setattr(pose, "Estimator", lambda settings: _FakeEstimator())
    monkeypatch.setattr(pose, "_render_frames", explode)

    with pytest.raises(Boom):
        pose.render(_a_tiny_video(tmp_path), 0, 4, SETTINGS, destination)

    assert not destination.exists()
    assert not destination.with_suffix(".part.mp4").exists()


class _FakeEstimator:
    explanation = "fake"

    def close(self):
        pass


def _a_tiny_video(tmp_path) -> Path:
    """Four black frames -- enough for av.open to report a video stream."""
    import av as _av

    path = tmp_path / "tiny.mp4"
    with _av.open(str(path), "w") as container:
        stream = container.add_stream("libx264", rate=24)
        stream.width, stream.height = 32, 32
        stream.pix_fmt = "yuv420p"
        for _ in range(4):
            frame = _av.VideoFrame(32, 32, "yuv420p")
            for plane in frame.planes:
                plane.update(bytes(plane.buffer_size))
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return path


# ---------------------------------------------------------------------------------
# Diagnostics and the rest of the app
# ---------------------------------------------------------------------------------

def _on_server(**kw):
    """A posed row with no local file, so `local_missing` does not muddy the warnings."""
    row = RefRow(kind=VIDEO, comfy_name="clip.mp4", **kw)
    row.fps, row.frame_count = 24.0, 600
    row.use_pose = True
    return row


def test_a_posed_row_says_so_but_never_blocks_the_queue():
    """Queueing renders whatever is missing before it uploads, so "not rendered yet" is a
    statement about now, not about readiness."""
    errors, warnings = row_warnings(_on_server(), target_frames=124)
    assert errors == []
    assert any("skeleton" in w for w in warnings)
    assert any("POSE? thumbnail" in w for w in warnings)


def test_a_row_whose_skeleton_is_ready_stops_explaining_how_to_get_one():
    row = _on_server()
    row.pose_path = "D:/runs/pose/p.mp4"
    _errors, warnings = row_warnings(row, target_frames=124)
    assert any("skeleton" in w for w in warnings)
    assert not any("POSE? thumbnail" in w for w in warnings)


def test_toggling_pose_marks_a_scene_modified():
    """signature_of normalises each reference down to a tuple; a field missing from it is
    a field the modified marker cannot see."""
    off = [_video(use_pose=False).to_dict()]
    on = [_video(use_pose=True).to_dict()]
    assert (scenes.signature_of("p", 5.0, 1, False, off)
            != scenes.signature_of("p", 5.0, 1, False, on))


def test_the_pose_settings_survive_a_round_trip_and_reject_nonsense():
    good = settings_mod.pose_settings(
        {"pose_model": "vitpose-b", "pose_runtime": "cpu", "pose_kpt_thr": 0.6})
    assert (good.model, good.runtime, good.kpt_thr) == ("vitpose-b", "cpu", 0.6)

    fallback = settings_mod.pose_settings(
        {"pose_model": "nope", "pose_runtime": "quantum", "pose_kpt_thr": "abc"})
    assert fallback.model == config.DEFAULT_POSE_MODEL
    assert fallback.runtime == config.DEFAULT_POSE_RUNTIME
    assert fallback.kpt_thr == config.DEFAULT_POSE_KPT_THR


def test_pose_settings_are_saved_and_reloaded(tmp_path):
    path = tmp_path / "settings.json"
    data = dict(settings_mod.DEFAULTS)
    data["pose_model"] = "vitpose-b"
    settings_mod.save_settings(data, path)
    assert settings_mod.load_settings(path)["pose_model"] == "vitpose-b"


def test_the_history_record_says_whether_a_skeleton_was_sent():
    from harmon3 import history, refs as refs_mod

    refset = RefSet(videos=[_video(), _video("b.mp4", use_pose=False, local="D:/b.mp4")])
    snapshot = history.refs_snapshot(refset, refs_mod.compute_tags(refset))
    assert [entry["use_pose"] for entry in snapshot] == [True, False]


def test_every_named_pose_model_is_reachable_from_the_settings():
    for name in config.POSE_MODELS:
        assert settings_mod.pose_settings({"pose_model": name}).model == name
        assert pose.model_path(name).parent == config.POSE_MODELS_DIR


def test_an_unknown_model_is_refused_rather_than_guessed_at():
    with pytest.raises(pose.PoseError):
        pose.model_path("vitpose-xxl")


def test_the_graph_builder_knows_nothing_about_pose():
    """The whole point of swapping on the snapshot: the estimator is invisible from here.
    A pose branch in the builder would be a second place for the trim to be applied."""
    source = Path(graph_builder.__file__).read_text(encoding="utf-8")
    assert "pose" not in source.lower()
