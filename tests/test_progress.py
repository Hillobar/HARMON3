"""Turning sampler steps into a time remaining."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmon3.progress import EtaEstimator, StageTracker, format_duration      # noqa: E402


def _run(steps, total=20, pace=2.0, start=100.0):
    """Feed `steps` progress messages `pace` seconds apart."""
    eta = EtaEstimator()
    for i in range(1, steps + 1):
        eta.note_step(i, total, start + i * pace)
    return eta


def test_no_estimate_before_anything_has_happened():
    eta = EtaEstimator()
    assert eta.remaining_seconds() is None
    assert eta.per_step_seconds() is None
    assert eta.describe("sampling") == "sampling"


def test_the_first_step_alone_gives_no_estimate():
    """That interval starts before the model is loaded; timing it would be nonsense."""
    eta = EtaEstimator()
    eta.note_step(1, 20, 100.0)
    assert eta.remaining_seconds() is None
    assert "estimating" in eta.describe("sampling")


def test_the_model_load_before_step_one_is_not_counted():
    eta = EtaEstimator()
    eta.note_step(1, 20, 100.0)     # 100s of model loading preceded this
    eta.note_step(2, 20, 102.0)
    eta.note_step(3, 20, 104.0)
    assert eta.per_step_seconds() == 2.0        # not skewed by the load
    assert eta.remaining_seconds() == 34.0      # 17 steps left at 2s


def test_estimate_from_a_steady_pace():
    eta = _run(5, total=20, pace=3.0)
    assert eta.per_step_seconds() == 3.0
    assert eta.remaining_seconds() == 45.0      # 15 steps left


def test_estimate_reaches_zero_on_the_last_step():
    eta = _run(20, total=20, pace=2.0)
    assert eta.remaining_seconds() == 0.0
    assert "done" in eta.describe("sampling")


def test_the_average_only_covers_the_recent_window():
    eta = EtaEstimator(window=4)
    now = 0.0
    for _ in range(6):                 # slow steps that should age out
        now += 10.0
        eta.note_step(eta.step + 1, 40, now)
    for _ in range(4):                 # then fast ones
        now += 1.0
        eta.note_step(eta.step + 1, 40, now)
    assert eta.per_step_seconds() == 1.0


def test_skipped_steps_are_divided_out():
    """ComfyUI can report step 5 straight after step 1 on a fast sampler."""
    eta = EtaEstimator()
    eta.note_step(1, 20, 0.0)
    eta.note_step(5, 20, 8.0)
    assert eta.per_step_seconds() == 2.0


def test_a_measured_step_time_wins_over_the_inferred_one():
    """The preview node times the sampler itself, which beats message arrival times."""
    eta = _run(5, total=20, pace=3.0)
    assert eta.per_step_seconds() == 3.0

    eta.note_measured_step_ms(1500.0)
    assert eta.per_step_seconds() == 1.5
    assert eta.remaining_seconds() == 22.5


def test_a_nonsense_measurement_is_ignored():
    eta = _run(5, total=20, pace=3.0)
    for bad in (None, "", 0, -5, "abc"):
        eta.note_measured_step_ms(bad)
        assert eta.per_step_seconds() == 3.0


def test_reset_clears_everything():
    eta = _run(5)
    eta.note_measured_step_ms(1000)
    eta.reset()
    assert eta.remaining_seconds() is None
    assert eta.per_step_seconds() is None
    assert (eta.step, eta.total) == (0, 0)


def test_describe_reads_as_a_caption():
    eta = _run(5, total=20, pace=3.0)
    assert eta.describe("sampling") == "sampling  5/20  45s left"
    assert eta.describe() == "5/20  45s left"


def test_out_of_order_or_repeated_steps_do_not_corrupt_the_average():
    eta = _run(5, total=20, pace=3.0)
    pace = eta.per_step_seconds()
    eta.note_step(5, 20, 999.0)       # same step again
    eta.note_step(4, 20, 1000.0)      # and a step backwards
    assert eta.per_step_seconds() == pace


def test_zero_total_yields_no_estimate():
    eta = EtaEstimator()
    eta.note_step(1, 0, 0.0)
    eta.note_step(2, 0, 2.0)
    assert eta.remaining_seconds() is None


# ---------------------------------------------------------------------------------
# Folding a staged sampler's several passes back into one
# ---------------------------------------------------------------------------------

def _stage(tracker, first, last, maximum):
    """Feed one stage's worth of messages and hand back the last pair reported."""
    result = (0, 0)
    for step in range(first, last + 1):
        result = tracker.note(step, maximum)
    return result


def test_a_single_pass_sampler_is_reported_unchanged():
    tracker = StageTracker()
    tracker.reset(20)
    assert _stage(tracker, 1, 20, 20) == (20, 20)


def test_two_stages_read_as_one_walk_across_the_run():
    """The staged sampler counts 1..11 then 1..9 for a twenty-step run."""
    tracker = StageTracker()
    tracker.reset(20)

    assert _stage(tracker, 1, 11, 11) == (11, 20)
    # The second stage starts over at one; the bar must not.
    assert tracker.note(1, 9) == (12, 20)
    assert _stage(tracker, 2, 9, 9) == (20, 20)


def test_the_position_never_goes_backwards_across_a_boundary():
    tracker = StageTracker()
    tracker.reset(20)
    seen = [_stage(tracker, 1, 11, 11)[0]]
    for step in range(1, 10):
        seen.append(tracker.note(step, 9)[0])
    assert seen == sorted(seen)


def test_the_total_holds_still_from_the_first_stage():
    """Without the run's own step count the first stage would claim the run is 11 long."""
    tracker = StageTracker()
    tracker.reset(20)
    assert tracker.note(1, 11)[1] == 20


def test_without_a_known_step_count_the_total_grows_but_never_trails_the_position():
    tracker = StageTracker()
    tracker.reset()                      # e.g. a graph whose step count is not ours
    assert tracker.note(5, 11) == (5, 11)
    tracker.note(11, 11)
    assert tracker.note(1, 9) == (12, 20)


def test_a_position_past_the_expected_end_does_not_overflow_the_bar():
    """A requeued graph may have been built with a different step count."""
    tracker = StageTracker()
    tracker.reset(10)
    position, total = tracker.note(15, 15)
    assert position <= total


def test_reset_forgets_the_previous_run():
    tracker = StageTracker()
    tracker.reset(20)
    _stage(tracker, 1, 11, 11)
    tracker.reset(20)
    assert tracker.note(1, 11) == (1, 20)


def test_format_duration():
    assert format_duration(0) == "0s"
    assert format_duration(45) == "45s"
    assert format_duration(59.4) == "59s"
    assert format_duration(60) == "1:00"
    assert format_duration(200) == "3:20"
    assert format_duration(3600) == "1:00:00"
    assert format_duration(3750) == "1:02:30"
    assert format_duration(-5) == "0s"
