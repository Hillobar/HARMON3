"""The pose pass, on its own thread.

Rendering a skeleton takes tens of seconds. That is far too long to sit on the GUI thread,
and it does not belong on the job thread either: while ``JobRunner.submit_job`` is blocked,
that thread's event loop is not spinning, so nothing posted to it -- including a cancel --
can run. Local work with a stop button needs a thread whose loop is free, so it gets one.

Two entry points, both `@Slot`s reached by queued connection, results back as signals.
``cancel()`` is the exception: it is a plain attribute write from the GUI thread, checked
between frames, because a queued cancel would sit behind the very work it is cancelling.
``ComfyWsClient.stop()`` does the same thing for the same reason.

No widgets are touched here.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from .. import pose as pose_mod

log = logging.getLogger(__name__)


@dataclass
class PoseJob:
    """One reference to render, resolved by the caller so the worker decides nothing."""

    uid: int
    source: str
    display_name: str
    start: int
    length: int
    destination: str
    #: The frame size to write at, or None for the source's own.
    canvas: tuple[int, int] | None = None
    #: Draw the skeleton, or pass the frames through at the new size.
    skeleton: bool = True


class PoseRunner(QObject):
    """Downloads the weights if they are missing, then renders each job in turn."""

    #: done, total, what is being done -- suitable for a determinate bar and a label.
    progress = Signal(int, int, str)
    #: row uid, the finished clip's path
    rendered = Signal(int, str)
    #: row uid (0 for a failure that belongs to no single row), what went wrong
    failed = Signal(int, str)
    #: Every job in the batch has been dealt with, successfully or not.
    finished = Signal(bool)          # True when nothing failed and nothing was cancelled
    #: Which execution provider the last render actually used, in words.
    provider_known = Signal(str)

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        #: Written from the GUI thread, read here between frames. Plain attribute on
        #: purpose: a queued slot could not run while a render is in progress.
        self._stop = threading.Event()
        self._reported_provider = ""

    # -- control -------------------------------------------------------------------

    def cancel(self) -> None:
        """Ask the current batch to stop. Safe to call from any thread."""
        self._stop.set()

    def _cancelled(self) -> bool:
        return self._stop.is_set()

    # -- work ----------------------------------------------------------------------

    @Slot(object, object)
    def render_all(self, jobs, settings) -> None:
        """Render every job, reporting as it goes. One batch at a time."""
        self._stop.clear()
        jobs = list(jobs or [])
        if not jobs:
            self.finished.emit(True)
            return

        try:
            # Only for jobs that draw one. A batch of pure rescales must not stall on a
            # gigabyte of ONNX it is never going to open.
            if any(job.skeleton for job in jobs):
                self._ensure_weights(settings)
        except Exception as exc:
            self.failed.emit(0, f"the pose model could not be downloaded: {exc}")
            self.finished.emit(False)
            return

        if self._cancelled():
            self.finished.emit(False)
            return

        ok = True
        total = sum(max(1, job.length) for job in jobs)
        done_before = 0

        for job in jobs:
            if self._cancelled():
                ok = False
                break
            label = f"{'pose' if job.skeleton else 'resize'}: {job.display_name}"

            def on_frame(done: int, _total: int, _base=done_before, _label=label) -> None:
                self.progress.emit(_base + done, total, _label)

            try:
                result = pose_mod.render(
                    job.source, job.start, job.length, settings, job.destination,
                    on_frame=on_frame, should_stop=self._cancelled,
                    canvas=job.canvas, skeleton=job.skeleton)
            except pose_mod.PoseError as exc:
                # Cancellation arrives as a PoseError from inside the frame loop, which is
                # a stop rather than a failure and should not be reported as one.
                if self._cancelled():
                    ok = False
                    break
                log.info("Pose render failed for %s: %s", job.display_name, exc)
                self.failed.emit(job.uid, str(exc))
                ok = False
                continue
            except Exception as exc:
                log.exception("Pose render blew up for %s", job.display_name)
                self.failed.emit(job.uid, f"the pose pass failed: {exc}")
                ok = False
                continue

            done_before += max(1, job.length)
            self._note_provider(result)
            if job.skeleton and result.held_badly:
                self.failed.emit(job.uid, (
                    f"no person was found in {result.held} of {result.frames} frames, so "
                    "the skeleton holds still through them - try a different section"))
                ok = False
            self.rendered.emit(job.uid, str(result.path))

        self.finished.emit(ok and not self._cancelled())

    def _note_provider(self, result) -> None:
        """Say which execution provider ran, once, rather than on every clip."""
        if result.provider and result.provider != self._reported_provider:
            self._reported_provider = result.provider
            self.provider_known.emit(result.provider)

    def _ensure_weights(self, settings) -> None:
        """Fetch anything missing before the first frame, with its own progress."""
        for label, url, destination in pose_mod.missing_models(settings):
            self.progress.emit(0, 1, f"downloading the {label} weights")

            def on_bytes(done: int, total: int, _label=label) -> None:
                if total:
                    self.progress.emit(done, total,
                                       f"downloading the {_label} weights "
                                       f"({done >> 20}/{total >> 20} MB)")

            pose_mod.download(url, Path(destination),
                              on_progress=on_bytes, should_stop=self._cancelled)
