"""Turning sampler steps into a time remaining.

Two sources feed this. Every workflow emits ComfyUI's ``progress`` messages, from which
the pace is measured here; the Model Preview Override node additionally reports its own
averaged step time, measured at the sampler itself, which is preferred when present.

Both sources count *within a sampling pass*, and the staged sampler makes several passes
per run -- one per resolution stage, each with a fresh progress bar. ``StageTracker`` folds
those back into one count, so the bar crosses the window once rather than snapping back to
the start at every stage boundary.

No Qt, no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field

#: How many recent steps the rolling average covers. Matches the preview node's window.
WINDOW = 8


@dataclass
class StageTracker:
    """Fold a sampler that restarts its count per stage into one run-wide position.

    The staged sampler samples each resolution stage separately, so for a 20-step run it
    reports 1..11 and then 1..9. The stages partition the step count exactly, which is what
    makes ``expected`` -- the run's configured step count -- the right total to show
    throughout; without it the first stage would claim the run is 11 steps long.

    A single-pass sampler never goes backwards, so it lands in the same code path with
    ``done`` staying at zero.
    """

    #: Steps completed in the stages that have already finished.
    done: int = 0
    #: The last position reported within the current stage.
    seen: int = 0
    #: The run's own step count, when it is known.
    expected: int = 0

    def reset(self, expected: int = 0) -> None:
        self.done = 0
        self.seen = 0
        self.expected = max(0, int(expected or 0))

    def note(self, value: int, maximum: int) -> tuple[int, int]:
        """Take one stage-local (position, total) and return the run-wide pair."""
        if value < self.seen:
            # The count went backwards, so a new stage has begun and everything the
            # previous one covered is now behind us.
            self.done += self.seen
        self.seen = value

        position = self.done + value
        # Falling back to what has been seen so far keeps the bar honest when the step
        # count is unknown: it may grow, but it never runs past its own end.
        total = self.expected or (self.done + maximum)
        return min(position, total), max(total, position)


@dataclass
class EtaEstimator:
    window: int = WINDOW

    _durations: list[float] = field(default_factory=list)
    _last_step: int | None = None
    _last_time: float | None = None
    _measured_step_s: float | None = None
    _step: int = 0
    _total: int = 0

    def reset(self) -> None:
        self._durations.clear()
        self._last_step = None
        self._last_time = None
        self._measured_step_s = None
        self._step = 0
        self._total = 0

    # -- inputs --------------------------------------------------------------------

    def note_step(self, step: int, total: int, now: float) -> None:
        """Record a sampler step observed at monotonic time ``now``."""
        self._step, self._total = step, total

        if self._last_step is not None and self._last_time is not None:
            advanced = step - self._last_step
            elapsed = now - self._last_time
            if advanced > 0 and elapsed > 0:
                self._durations.append(elapsed / advanced)
                del self._durations[: -self.window]
        # The very first interval is deliberately not measured: it starts before the
        # model is loaded, so it would poison the average with tens of seconds.

        self._last_step, self._last_time = step, now

    def note_measured_step_ms(self, milliseconds) -> None:
        """Adopt a step time measured at the sampler, which beats anything inferred here."""
        try:
            value = float(milliseconds)
        except (TypeError, ValueError):
            return
        if value > 0:
            self._measured_step_s = value / 1000.0

    # -- outputs -------------------------------------------------------------------

    @property
    def step(self) -> int:
        return self._step

    @property
    def total(self) -> int:
        return self._total

    def per_step_seconds(self) -> float | None:
        if self._measured_step_s:
            return self._measured_step_s
        if self._durations:
            return sum(self._durations) / len(self._durations)
        return None

    def remaining_seconds(self) -> float | None:
        pace = self.per_step_seconds()
        if pace is None or self._total <= 0:
            return None
        remaining = max(0, self._total - self._step)
        return remaining * pace

    def describe(self, stage: str = "") -> str:
        """The progress bar's caption: where we are, and how long is left."""
        parts = []
        if stage:
            parts.append(stage)
        if self._total > 0:
            parts.append(f"{self._step}/{self._total}")

        remaining = self.remaining_seconds()
        if remaining is not None:
            parts.append("done" if remaining < 1 else f"{format_duration(remaining)} left")
        elif self._total > 0:
            parts.append("estimating...")

        return "  ".join(parts) if parts else "working"


def format_duration(seconds: float) -> str:
    """A compact, readable duration: 45s, 3:20, 1:02:30."""
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}:{secs:02d}"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}:{minutes:02d}:{secs:02d}"
