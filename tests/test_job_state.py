"""Isolation between the UI's live state and the snapshot handed to the job worker.

The worker rewrites reference filenames while the user keeps editing, so anything shared
between the two would let a mid-flight edit change the graph being submitted -- or, in
the case of the upload cache, corrupt the settings file being written at the same moment.
"""

from __future__ import annotations

import json
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmon3 import prompt, settings as settings_mod           # noqa: E402
from harmon3.graph_builder import BuildState                   # noqa: E402
from harmon3.jobs import JobRequest                            # noqa: E402
from harmon3.refs import IMAGE, VIDEO, RefRow, RefSet          # noqa: E402


def _state():
    return BuildState(
        prompt_sections=prompt.from_legacy("original"),
        refs=RefSet(
            images=[RefRow(kind=IMAGE, local_path="C:/refs/a.png")],
            videos=[RefRow(kind=VIDEO, local_path="C:/refs/v.mp4")],
        ),
    )


def test_snapshot_copies_state_rows_and_cache():
    state = _state()
    cache = {"sha": "harmon3/a.png"}
    request = JobRequest.snapshot(state, cache)

    assert request.state is not state
    assert request.state.refs is not state.refs
    assert request.state.refs.images[0] is not state.refs.images[0]
    assert request.upload_cache is not cache


def test_worker_side_mutations_do_not_reach_the_ui_state():
    """Exactly what _resolve_uploads does to its copy."""
    state = _state()
    cache = {}
    request = JobRequest.snapshot(state, cache)

    request.state.refs.images[0].comfy_name = "harmon3/a_deadbeef.png"
    request.state.prompt_sections["summary"] = "rewritten by the worker"
    request.upload_cache["sha"] = "harmon3/a_deadbeef.png"

    assert state.refs.images[0].comfy_name is None
    assert state.prompt_sections == prompt.from_legacy("original")
    assert cache == {}


def test_snapshot_preserves_everything_the_builder_needs():
    state = _state()
    state.seed = 12345
    state.duration_seconds = 3.5
    state.refs.videos[0].use_soundtrack = False
    request = JobRequest.snapshot(state, {})

    assert request.state.seed == 12345
    assert request.state.duration_seconds == 3.5
    assert request.state.refs.videos[0].use_soundtrack is False
    assert [r.local_path for r in request.state.refs.all_rows()] == \
           [r.local_path for r in state.refs.all_rows()]


def test_local_rows_do_not_persist_their_derived_server_name():
    """Otherwise editing a reference file in place would keep submitting the old bytes.

    The server name encodes the file's hash, so it is only valid for the exact contents
    that were uploaded. It is recomputed on every submit and cached by hash instead.
    """
    row = RefRow(kind=IMAGE, local_path="C:/refs/a.png")
    row.comfy_name = "harmon3/a_deadbeef.png"
    assert "comfy_name" not in row.to_dict()
    assert RefSet.from_list([row.to_dict()]).images[0].comfy_name is None


def test_server_rows_do_persist_their_name():
    """A row that only ever existed on the server has nothing to recompute from."""
    row = RefRow(kind=IMAGE, comfy_name="already_there.png")
    assert row.to_dict()["comfy_name"] == "already_there.png"
    assert RefSet.from_list([row.to_dict()]).images[0].comfy_name == "already_there.png"


def test_save_settings_survives_a_concurrently_mutated_cache(tmp_path):
    """json.dump with indent uses the pure-Python encoder, which iterates as it writes.

    A cache being updated by the job thread mid-save would otherwise raise
    "dictionary changed size during iteration" and abandon the write.
    """
    live = {str(i): "v" for i in range(3000)}
    data = dict(settings_mod.DEFAULTS)
    data["upload_cache"] = live

    stop = threading.Event()

    def churn():
        counter = 0
        while not stop.is_set():
            live[f"x{counter}"] = "y"
            counter += 1
            if counter > 30000:
                counter = 0
                live.clear()

    worker = threading.Thread(target=churn, daemon=True)
    worker.start()
    try:
        target = tmp_path / "settings.json"
        for _ in range(25):
            settings_mod.save_settings(data, target)
    finally:
        stop.set()
        worker.join(timeout=2)

    assert isinstance(json.loads(target.read_text(encoding="utf-8")), dict)
    assert not list(tmp_path.glob(".settings-*.tmp"))


def test_save_settings_never_raises(tmp_path, monkeypatch):
    """It runs from closeEvent, where an exception would skip the thread shutdown."""
    def explode(*_args, **_kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(settings_mod.json, "dump", explode)
    settings_mod.save_settings(dict(settings_mod.DEFAULTS), tmp_path / "settings.json")
    assert not list(tmp_path.glob(".settings-*.tmp"))


def test_settings_round_trip_keeps_reference_structure(tmp_path):
    state = _state()
    state.refs.videos[0].use_soundtrack = False
    data = settings_mod.capture_from_state(dict(settings_mod.DEFAULTS), state)

    target = tmp_path / "settings.json"
    settings_mod.save_settings(data, target)
    restored_state = _state()
    settings_mod.apply_to_state(restored_state, settings_mod.load_settings(target))

    assert restored_state.prompt_text == state.prompt_text
    assert len(restored_state.refs.images) == 1
    assert restored_state.refs.videos[0].use_soundtrack is False
