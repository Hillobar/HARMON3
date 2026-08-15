"""Where files live and where sections are marked.

References own the files: they are added in the reference panel and nowhere else. The
result frame is the surface for looking at one and marking where the section that reaches
the model starts -- so a reference stays a name plus an in point, and the frame keeps its
hands off the list.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6.QtWidgets")

from harmon3 import config                                      # noqa: E402
from harmon3.refs import AUDIO, IMAGE, VIDEO, RefRow, RefSet     # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    import os

    from PySide6.QtWidgets import QApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture
def files(tmp_path):
    """Placeholders with bytes in them; an empty file is refused outright."""
    made = {}
    for kind, name in ((IMAGE, "still.png"), (VIDEO, "clip.mp4"), (AUDIO, "roar.wav")):
        path = tmp_path / name
        path.write_bytes(b"placeholder bytes")
        made[kind] = str(path)
    made["pose"] = str(tmp_path / "pose.mp4")
    Path(made["pose"]).write_bytes(b"stick figures")
    return made


@pytest.fixture(scope="module")
def window(qapp):
    from harmon3.ui.main_window import MainWindow

    made = MainWindow("http://127.0.0.1:8188", "marking-test")
    made._save_settings = lambda: None          # never write the user's settings file
    made.reachable = True
    yield made

    made._save_timer.stop()
    made.ws_client.stop()
    made.ws_thread.quit()
    made.ws_thread.wait(2000)
    made.job_thread.quit()
    made.job_thread.wait(2000)


@pytest.fixture
def clean(window):
    window.state.refs = RefSet()
    window.ref_panel.set_refset(window.state.refs)
    window.extracting = window.submitting = False
    window.runs.clear()
    window.player.clear()
    return window


def _add(window, kind, path):
    """Add a reference the way the panel does, and hand back its row."""
    window.ref_panel.lists[kind].add_paths([path])
    return window.ref_panel.lists[kind].rows[-1].row


# ---------------------------------------------------------------------------------
# Files are added in the reference panel, and only there
# ---------------------------------------------------------------------------------

def test_the_result_frame_takes_no_drops(clean):
    """Dropping is the reference panel's job; the frame is for looking and marking."""
    assert clean.player.acceptDrops() is False
    assert not hasattr(clean.player, "files_dropped")


def test_the_result_frame_cannot_add_references(clean):
    assert not hasattr(clean.player, "send_button")
    assert not hasattr(clean, "workbench_row")
    assert not hasattr(clean, "_on_send_requested")


def test_the_reference_panel_still_accepts_drops(clean):
    assert clean.ref_panel.acceptDrops() is True


def test_a_reference_added_to_the_panel_is_in_the_state(clean, files):
    _add(clean, VIDEO, files[VIDEO])
    assert [Path(r.local_path).name for r in clean._collect_state().refs.videos] == ["clip.mp4"]


# ---------------------------------------------------------------------------------
# Clicking one opens it in the result frame, without playing it
# ---------------------------------------------------------------------------------

def test_clicking_a_reference_opens_it_in_the_result_frame(clean, files):
    row = _add(clean, VIDEO, files[VIDEO])
    clean._on_ref_preview(row, "source", False)

    assert clean._shown_row() is row
    assert clean.player.shown_media()[0].path == files[VIDEO]


def test_opening_one_does_not_start_it_playing(clean, files):
    """A reference list you click through should not start talking at you."""
    from PySide6.QtMultimedia import QMediaPlayer

    row = _add(clean, VIDEO, files[VIDEO])
    clean._on_ref_preview(row, "source", False)

    pane = clean.player.media_view.playing_pane()
    assert pane is not None
    assert pane.player.playbackState() != QMediaPlayer.PlayingState


def test_audio_opens_without_playing_either(clean, files):
    from PySide6.QtMultimedia import QMediaPlayer

    row = _add(clean, AUDIO, files[AUDIO])
    clean._on_ref_preview(row, "source", False)

    pane = clean.player.media_view.playing_pane()
    assert pane.player.playbackState() != QMediaPlayer.PlayingState


def test_the_transport_is_there_to_play_it_deliberately(clean, files):
    row = _add(clean, VIDEO, files[VIDEO])
    clean._on_ref_preview(row, "source", False)

    assert clean.player.play_button.isEnabled() is True
    assert clean.player.stop_button.isHidden() is False


# ---------------------------------------------------------------------------------
# The marks live in the result frame and define what is sent
# ---------------------------------------------------------------------------------

def test_the_mark_is_set_in_the_result_frame(clean, files):
    row = _add(clean, VIDEO, files[VIDEO])
    row.fps, row.frame_count, row.duration_s = 24.0, 240, 10.0
    clean._on_ref_preview(row, "source", False)

    assert clean.player.timeline.row is row
    clean.player.timeline._set_start(2000)
    assert row.trim_start == 48


def test_audio_is_marked_in_seconds(clean, files):
    row = _add(clean, AUDIO, files[AUDIO])
    row.duration_s = 10.0
    clean._on_ref_preview(row, "source", False)

    clean.player.timeline._set_start(1500)
    assert row.trim_start == 1.5


def test_the_reference_row_is_just_a_name_and_the_mark(clean, files):
    row = _add(clean, VIDEO, files[VIDEO])
    row.fps, row.frame_count = 24.0, 240
    row.trim_start = 48

    stored = row.to_dict()
    assert stored["local_path"] == files[VIDEO]
    assert stored["trim_start"] == 48
    assert row.window_summary(120) == "48-168f (5.00s)"


def test_the_mark_is_what_reaches_the_model(clean, files):
    """The loader decodes from there for the generated length; nothing is cut on disk."""
    from harmon3.graph_builder import BuildState, build_graph

    row = RefRow(kind=VIDEO, comfy_name="clip.mp4")
    row.fps, row.frame_count = 24.0, 240
    row.trim_start = 48

    workflow = config.load_workflow()
    built = build_graph(workflow.graph, BuildState(refs=RefSet(videos=[row])), workflow.roles)
    loader = built.graph[str(workflow.roles.injected["video"])]["inputs"]
    assert loader["skip_first_frames"] == 48
    assert loader["frame_load_cap"] == built.frames


def test_the_mark_survives_a_settings_round_trip(files):
    row = RefRow(kind=VIDEO, local_path=files[VIDEO])
    row.trim_start = 48

    restored = RefSet.from_list([row.to_dict()]).videos[0]
    assert restored.trim_start == 48
    assert restored.marked is True


# ---------------------------------------------------------------------------------
# Pose: a skeleton is rendered here, before the run goes out
# ---------------------------------------------------------------------------------

@pytest.fixture
def no_worker(clean):
    """Intercept the pose request instead of letting a real render start.

    The runner lives on its own thread and would go looking for weights; what is under
    test is which jobs the window asks for, not the estimator.
    """
    clean._pose_requested.disconnect()
    captured = []
    clean._pose_requested.connect(lambda jobs, settings: captured.append((jobs, settings)))
    yield captured
    clean._pose_requested.disconnect()
    clean._pose_requested.connect(clean.pose_runner.render_all)
    clean.posing = False
    clean.submitting = False
    clean._pose_then_submit = False


def test_queueing_renders_a_missing_skeleton_before_it_submits(clean, files, no_worker,
                                                               monkeypatch):
    """The graph has to name a file that exists, so the local pass comes first."""
    from harmon3.ui import main_window as main_window_mod

    row = _add(clean, VIDEO, files[VIDEO])
    row.use_pose = True
    monkeypatch.setattr(main_window_mod.pose_mod, "resolve",
                        lambda r, d, s: (Path("D:/runs/pose/x.mp4"), False))

    assert clean._start_pose_pass(then_submit=True) is True
    assert clean.posing is True
    jobs, _settings = no_worker[0]
    assert [job.uid for job in jobs] == [row.uid]
    assert jobs[0].start == int(row.trim_start)
    assert jobs[0].length == clean.params_panel.frames()


def test_a_skeleton_that_is_already_rendered_costs_nothing(clean, files, no_worker,
                                                           monkeypatch):
    from harmon3.ui import main_window as main_window_mod

    row = _add(clean, VIDEO, files[VIDEO])
    row.use_pose = True
    cached = Path(files[VIDEO])          # any file that exists will do
    monkeypatch.setattr(main_window_mod.pose_mod, "resolve", lambda r, d, s: (cached, True))

    assert clean._start_pose_pass(then_submit=True) is False
    assert clean.posing is False
    assert no_worker == []
    assert row.pose_path == str(cached)


def test_an_unposed_reference_queues_without_a_pose_phase(clean, files, no_worker):
    _add(clean, VIDEO, files[VIDEO])
    assert clean._start_pose_pass(then_submit=True) is False
    assert no_worker == []


def test_cancelling_the_pose_pass_stops_it_here_rather_than_at_the_server(clean,
                                                                         monkeypatch):
    """The one thing the app can actually stop on the spot. Cancel during a run has to go
    to ComfyUI and wait; cancel during a pose pass is a flag the worker reads."""
    interrupted, stopped = [], []
    monkeypatch.setattr(clean.pose_runner, "cancel", lambda: stopped.append(True))
    monkeypatch.setattr(clean.job_runner, "interrupt", lambda *a: interrupted.append(True))

    clean.posing = True
    clean._on_cancel_clicked()
    clean.posing = False

    assert stopped == [True]
    assert interrupted == []


def test_a_pose_pass_that_failed_never_reaches_the_server(clean, monkeypatch):
    """A graph naming a skeleton that was never rendered would fail on the server several
    minutes later, which is a worse way to find out."""
    submitted = []
    monkeypatch.setattr(clean, "_submit_now", lambda: submitted.append(True))

    clean.posing, clean._pose_then_submit, clean.submitting = True, True, True
    clean._on_pose_finished(False)

    assert submitted == []
    assert clean.submitting is False
    assert "not queued" in clean.stage_label.text()


def test_a_pose_pass_that_worked_goes_straight_on_to_the_run(clean, monkeypatch):
    submitted = []
    monkeypatch.setattr(clean, "_submit_now", lambda: submitted.append(True))

    clean.posing, clean._pose_then_submit = True, True
    clean._on_pose_finished(True)

    assert submitted == [True]
    assert clean.posing is False


def test_moving_the_mark_forgets_the_skeleton_that_was_rendered(clean, files, tmp_path):
    row = _add(clean, VIDEO, files[VIDEO])
    row.fps, row.frame_count = 24.0, 600
    row.use_pose = True
    clip = tmp_path / "p.mp4"
    clip.write_bytes(b"skeleton")
    row.pose_path = str(clip)
    row.pose_section = (0, clean.params_panel.frames())

    row.trim_start = 200
    clean._refresh_derived()

    assert row.pose_path is None


def test_ticking_pose_puts_a_way_to_render_it_on_screen(clean, files):
    """The slot appears as soon as Pose is ticked, not once a render exists -- showing it
    only when there was something to show left no way to get the first one."""
    row = _add(clean, VIDEO, files[VIDEO])
    widget = next(w for w in clean.ref_panel.all_row_widgets() if w.row is row)
    assert widget.pose_thumb.isHidden() is True

    widget.pose_box.setChecked(True)

    assert widget.pose_thumb.isHidden() is False
    assert widget.pose_thumb.text() == "POSE?"          # nothing rendered yet
    assert "Click to render" in widget.pose_thumb.toolTip()


def test_clicking_the_pose_thumbnail_with_nothing_rendered_asks_for_one(clean, files,
                                                                       monkeypatch):
    row = _add(clean, VIDEO, files[VIDEO])
    row.use_pose = True
    asked = []
    monkeypatch.setattr(clean, "_on_pose_preview", lambda r: asked.append(r))

    widget = next(w for w in clean.ref_panel.all_row_widgets() if w.row is row)
    widget.refresh_pose_thumbnail()
    widget.pose_thumb.clicked.emit("pose", False)       # the click, not the handler

    assert asked == [row]


def test_clicking_it_once_there_is_one_shows_the_skeleton(clean, files):
    row = _add(clean, VIDEO, files[VIDEO])
    row.use_pose = True
    row.pose_path = files["pose"]

    clean._on_ref_preview(row, "pose", False)
    shown = clean.player.shown_media()

    assert [item.path for item in shown] == [files["pose"]]
    assert "pose" in shown[0].caption


# ---------------------------------------------------------------------------------
# What a reference no longer carries
# ---------------------------------------------------------------------------------

def test_pose_is_a_toggle_on_the_row_and_nothing_in_the_result_frame(clean, files):
    """It came back, but not where it was. Extraction used to be a button in the result
    frame driving a second ComfyUI graph; it is now a per-reference flag and a local
    pass, so the frame stays a place for looking rather than a control panel."""
    row = _add(clean, VIDEO, files[VIDEO])
    clean._on_ref_preview(row, "source", False)

    assert not hasattr(clean.player, "pose_button")
    assert not hasattr(clean.player, "pose_requested")

    widget = next(w for w in clean.ref_panel.all_row_widgets() if w.row is row)
    assert widget.pose_box is not None
    assert row.use_pose is False
    widget.pose_box.setChecked(True)
    assert row.use_pose is True


def test_a_still_has_nothing_to_pose(clean, files):
    row = _add(clean, IMAGE, files[IMAGE])
    widget = next(w for w in clean.ref_panel.all_row_widgets() if w.row is row)
    assert widget.pose_box is None
    assert row.poses is False


def test_an_unreadable_reference_says_so_on_its_own_row(clean, tmp_path):
    """The frame does not narrate any more: the row that owns the file carries it."""
    empty = tmp_path / "truncated.mp4"
    empty.write_bytes(b"")
    row = _add(clean, VIDEO, str(empty))
    clean._on_ref_preview(row, "source", False)
    clean._refresh_derived()

    assert row.unreadable_reason == "the file is empty"
    widget = next(w for w in clean.ref_panel.all_row_widgets() if w.row is row)
    assert "cannot be read" in widget.warning_label.text()


def test_the_frame_says_nothing_it_does_not_have_to(clean, files):
    """No caption, no path, no "sending this much" -- the reference list already says
    which file it is, and the track draws the section. The line is for failures."""
    row = _add(clean, VIDEO, files[VIDEO])
    clean._on_ref_preview(row, "source", False)

    assert not hasattr(clean.player, "action_label")
    assert clean.player.status_label.isHidden() is True


def test_the_track_carries_the_section_instead(clean, files):
    row = _add(clean, VIDEO, files[VIDEO])
    row.fps, row.frame_count, row.duration_s = 24.0, 600, 25.0
    row.trim_start = 48
    clean.params_panel.duration_spin.setValue(5.0)
    clean._on_ref_preview(row, "source", False)
    clean._refresh_derived()

    frames = clean.params_panel.frames()
    assert clean.player.timeline.row is row
    assert clean.player.timeline.window_ms() == (2000, round(2000 + frames / 24 * 1000))

    widget = next(w for w in clean.ref_panel.all_row_widgets() if w.row is row)
    assert f"from 48-{48 + frames}f" in widget.detail_label.fullText()
