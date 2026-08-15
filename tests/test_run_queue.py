"""Several runs in flight at once, and cancelling them in the order they were submitted.

ComfyUI queues what it is given and works through it in order, so there was never any
reason for this app to refuse a second run while a first one was going. What it does need
is to know which of its own runs the server is on -- which is the head of the queue, since
the server is a FIFO -- and to take them off as they end.

Two halves: the queue itself (no Qt) and the window driving it.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmon3 import history                                      # noqa: E402
from harmon3.history import RunRecord                            # noqa: E402
from harmon3.runqueue import QueuedRun, RunQueue                 # noqa: E402


def _run(prompt_id: str, **kwargs) -> QueuedRun:
    return QueuedRun(
        prompt_id=prompt_id,
        record=RunRecord(prompt_id=prompt_id, submitted_at="2026-08-12T10:00:00"),
        **kwargs,
    )


# ---------------------------------------------------------------------------------
# The queue
# ---------------------------------------------------------------------------------

def test_an_empty_queue_has_nothing_active():
    queue = RunQueue()
    assert not queue
    assert queue.active is None
    assert queue.active_id is None


def test_the_oldest_submitted_run_is_the_active_one():
    """ComfyUI is a FIFO, so the head of the list is what it is working on."""
    queue = RunQueue()
    for prompt_id in ("a", "b", "c"):
        queue.add(_run(prompt_id))

    assert queue.active_id == "a"
    assert [run.prompt_id for run in queue.waiting] == ["b", "c"]
    assert len(queue) == 3


def test_finishing_the_active_run_promotes_the_next():
    queue = RunQueue()
    for prompt_id in ("a", "b", "c"):
        queue.add(_run(prompt_id))

    queue.remove("a")
    assert queue.active_id == "b"
    queue.remove("b")
    assert queue.active_id == "c"
    queue.remove("c")
    assert queue.active is None


def test_a_run_can_end_out_of_order_without_disturbing_the_rest():
    """Cancelling one that never started takes it out from the middle."""
    queue = RunQueue()
    for prompt_id in ("a", "b", "c"):
        queue.add(_run(prompt_id))

    queue.remove("b")
    assert [run.prompt_id for run in queue] == ["a", "c"]
    assert queue.active_id == "a"


def test_removing_something_that_was_never_queued_is_harmless():
    queue = RunQueue()
    queue.add(_run("a"))
    assert queue.remove("elsewhere") is None
    assert queue.active_id == "a"


def test_holds_is_what_tells_our_events_from_another_clients():
    queue = RunQueue()
    queue.add(_run("a"))
    assert queue.holds("a") is True
    assert queue.holds("someone-elses") is False


def test_started_is_recorded_separately_from_being_next():
    """Being at the head means the server will get to it, not that it has."""
    queue = RunQueue()
    queue.add(_run("a"))
    assert queue.active.started is False
    queue.active.started = True
    assert queue.active.started is True


# ---------------------------------------------------------------------------------
# The window
# ---------------------------------------------------------------------------------

pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def qapp():
    import os

    from PySide6.QtWidgets import QApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def window(qapp, tmp_path_factory):
    from harmon3 import config
    from harmon3.ui.main_window import MainWindow

    home = tmp_path_factory.mktemp("queue-home")
    original_state, original_runs = config.UI_STATE_PATH, config.RUNS_JSONL
    config.UI_STATE_PATH = home / "ui_state.ini"
    config.RUNS_JSONL = home / "runs.jsonl"

    made = MainWindow("http://127.0.0.1:8188", "queue-test")
    made._save_settings = lambda: None
    made.history_store.path = home / "runs.jsonl"
    made.reachable = True
    yield made

    made._save_timer.stop()
    made.ws_client.stop()
    made.ws_thread.quit()
    made.ws_thread.wait(2000)
    made.job_thread.quit()
    made.job_thread.wait(2000)
    config.UI_STATE_PATH, config.RUNS_JSONL = original_state, original_runs


class _Built:
    """The parts of a BuiltGraph the window reads off a submission."""

    def __init__(self):
        self.graph = {}
        self.labels = {}
        self.width = self.height = self.frames = 0
        self.pruned = []

        class _Tags:
            def tag_for(self, row):
                return ""

            def soundtrack_tag_for(self, row):
                return ""

        self.tags = _Tags()


@pytest.fixture
def idle(window):
    window.runs.clear()
    window.submitting = window.posing = False
    window._settle()
    return window


def _submit(window, prompt_id: str) -> None:
    window._on_submitted(prompt_id, _Built())


# -- queueing ---------------------------------------------------------------------

def test_a_run_in_flight_no_longer_blocks_the_queue_button(idle):
    _submit(idle, "one")
    assert idle.queue_button.isEnabled() is True


def test_the_app_still_sends_one_submission_at_a_time(idle):
    """The job thread uploads references serially; a second press mid-upload is refused."""
    idle.submitting = True
    idle._refresh_derived()
    assert idle.queue_button.isEnabled() is False


def test_each_submission_joins_the_back_of_the_queue(idle):
    for prompt_id in ("one", "two", "three"):
        _submit(idle, prompt_id)

    assert [run.prompt_id for run in idle.runs] == ["one", "two", "three"]
    assert idle.active_prompt_id == "one"


def test_a_queued_run_says_so_in_history(idle):
    _submit(idle, "one")
    _submit(idle, "two")

    assert idle.runs.find("one").record.status == history.STATUS_QUEUED
    assert idle.runs.find("two").record.status == history.STATUS_QUEUED


def test_the_server_starting_one_is_what_makes_it_running(idle):
    _submit(idle, "one")
    idle._on_execution_start("one")

    assert idle.runs.find("one").record.status == history.STATUS_RUNNING
    assert idle.runs.find("one").started is True


def test_the_depth_of_the_queue_is_on_the_cancel_button(idle):
    _submit(idle, "one")
    assert idle.cancel_button.text() == "Cancel"

    _submit(idle, "two")
    _submit(idle, "three")
    assert idle.cancel_button.text() == "Cancel (3)"


def test_nothing_outstanding_means_nothing_to_cancel(idle):
    assert idle.cancel_button.isEnabled() is False
    _submit(idle, "one")
    assert idle.cancel_button.isEnabled() is True


# -- cancelling -------------------------------------------------------------------

@pytest.fixture
def asked(idle):
    """What the window asked the job worker to cancel."""
    seen = []
    idle._interrupt_requested.connect(lambda pid, executing: seen.append((pid, executing)))
    return seen


def test_cancel_asks_the_worker_to_interrupt_the_oldest(idle, asked):
    for prompt_id in ("one", "two", "three"):
        _submit(idle, prompt_id)
    idle._on_execution_start("one")

    idle._on_cancel_clicked()

    # It has started, so stopping it means interrupting the server.
    assert asked == [("one", True)]


def test_a_run_that_has_not_started_is_deleted_rather_than_interrupted(idle, asked):
    """/interrupt takes down whatever is executing, which may be another client's job."""
    _submit(idle, "one")

    idle._on_cancel_clicked()

    assert asked == [("one", False)]


def test_cancelling_works_back_through_the_queue_in_order(idle):
    for prompt_id in ("one", "two", "three"):
        _submit(idle, prompt_id)

    # None of them has started, so each cancel closes its run out here and the next
    # becomes the oldest.
    idle._on_cancel_clicked()
    assert [run.prompt_id for run in idle.runs] == ["two", "three"]
    idle._on_cancel_clicked()
    assert [run.prompt_id for run in idle.runs] == ["three"]
    idle._on_cancel_clicked()
    assert not idle.runs


def test_a_run_the_server_never_began_is_closed_out_here(idle):
    """It emits no execution_interrupted, so nothing else would ever finish it."""
    _submit(idle, "one")

    idle._on_cancel_clicked()
    assert not idle.runs


def test_cancelling_a_started_run_waits_for_the_server_to_confirm(idle):
    """Interrupting is asynchronous; the run ends when execution_interrupted arrives."""
    _submit(idle, "one")
    idle._on_execution_start("one")

    idle._on_cancel_clicked()
    assert idle.runs.holds("one")

    idle._on_execution_interrupted("one", {})
    assert not idle.runs
    assert idle.runs.find("one") is None


# -- finishing --------------------------------------------------------------------

def test_a_finished_run_leaves_the_queue_and_the_next_takes_over(idle):
    _submit(idle, "one")
    _submit(idle, "two")
    idle._on_execution_start("one")

    idle.fetch_requested_for = "one"
    idle.runs.find("one").record.local_path = "runs/videos/one.mp4"
    idle._on_execution_success("one")

    assert idle.active_prompt_id == "two"
    assert len(idle.runs) == 1


def test_a_failed_run_does_not_take_the_queue_down_with_it(idle, monkeypatch):
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "exec", lambda self: None)
    _submit(idle, "one")
    _submit(idle, "two")

    idle._on_execution_error("one", {"exception_message": "boom"})

    assert idle.active_prompt_id == "two"
    assert idle.runs.find("two") is not None


def test_the_bar_says_how_many_are_left_rather_than_going_idle(idle):
    _submit(idle, "one")
    _submit(idle, "two")
    idle._on_execution_start("one")
    idle._on_execution_interrupted("one", {})

    assert "queued" in idle.progress_bar.format()
    assert idle.cancel_button.isEnabled() is True


def test_the_bar_goes_idle_once_the_queue_empties(idle):
    _submit(idle, "one")
    idle._on_execution_start("one")
    idle._on_execution_interrupted("one", {})

    assert idle.progress_bar.format() == "cancelled"
    assert idle.cancel_button.isEnabled() is False


def test_a_result_mid_batch_loads_without_taking_the_frame(idle, monkeypatch, tmp_path):
    """The run still going is the one with something to show; this one waits in its tab."""
    loaded = []
    monkeypatch.setattr(type(idle.player), "load",
                        lambda self, path, **kwargs: loaded.append(kwargs))
    video = tmp_path / "one.mp4"
    video.write_bytes(b"")

    _submit(idle, "one")
    _submit(idle, "two")
    idle._on_downloaded("one", str(video))

    assert loaded == [{"autoplay": False, "raise_tab": False}]


def test_the_last_result_of_a_batch_does_raise_the_tab(idle, monkeypatch, tmp_path):
    loaded = []
    monkeypatch.setattr(type(idle.player), "load",
                        lambda self, path, **kwargs: loaded.append(kwargs))
    video = tmp_path / "one.mp4"
    video.write_bytes(b"")

    _submit(idle, "one")
    idle._on_downloaded("one", str(video))

    assert loaded == [{"autoplay": True, "raise_tab": True}]


def test_a_failed_submission_leaves_the_running_preview_alone(idle, monkeypatch):
    """Queueing a second run and being refused must not freeze the first run's preview."""
    from harmon3.jobs import JobFailure
    from PySide6.QtWidgets import QMessageBox

    monkeypatch.setattr(QMessageBox, "exec", lambda self: None)
    stopped = []
    monkeypatch.setattr(type(idle.player), "end_live_preview",
                        lambda self: stopped.append(True))

    _submit(idle, "one")
    idle._on_execution_start("one")
    idle._on_submit_failed(JobFailure(message="the second one was refused"))

    assert stopped == []
    assert idle.runs.holds("one")


def test_a_run_ending_does_freeze_its_preview(idle):
    _submit(idle, "one")
    idle._on_execution_start("one")
    idle._on_execution_interrupted("one", {})

    assert idle.player.live_preview._timer.isActive() is False


def test_an_event_for_somebody_elses_prompt_is_ignored(idle):
    _submit(idle, "one")
    idle._on_execution_interrupted("a-browser-tab-on-the-same-server", {})
    assert idle.runs.holds("one")
