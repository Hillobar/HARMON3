"""The runs this app has handed to ComfyUI and not yet seen the end of.

ComfyUI's own queue is a FIFO, so the oldest run this app submitted is the one the server
is working on. That single fact is what the whole class rests on: there is no need to ask
the server which of ours is running, and no window in which the app disagrees with it. The
``started`` flag only records whether the server has *said* so yet, which matters when
cancelling -- ``/interrupt`` stops whatever is executing, and on a shared server that might
belong to somebody else until our own run has actually begun.

Runs are removed as they finish, so the head of the list is always the live one and an
empty queue means idle. Cancelling takes the head, which is what makes repeated presses
clear a batch in the order it was submitted.

No Qt, no network.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class QueuedRun:
    """One submitted run, from the moment the server accepted it until it ends."""

    prompt_id: str
    #: Its history entry, updated in place as the run progresses.
    record: object = None
    #: The BuiltGraph that was sent, kept for its node labels when an error names a node.
    built: object = None
    #: True once the server has said this one is executing.
    started: bool = False
    #: When it started, for the elapsed readout. None until it does.
    started_at: float | None = None


@dataclass
class RunQueue:
    """Everything outstanding, oldest first."""

    runs: list = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.runs)

    def __bool__(self) -> bool:
        return bool(self.runs)

    def __iter__(self):
        return iter(self.runs)

    def add(self, run: QueuedRun) -> QueuedRun:
        self.runs.append(run)
        return run

    def find(self, prompt_id: str) -> QueuedRun | None:
        for run in self.runs:
            if run.prompt_id == prompt_id:
                return run
        return None

    def remove(self, prompt_id: str) -> QueuedRun | None:
        run = self.find(prompt_id)
        if run is not None:
            self.runs.remove(run)
        return run

    def clear(self) -> list:
        gone, self.runs = self.runs, []
        return gone

    @property
    def active(self) -> QueuedRun | None:
        """The one the server is working on: the oldest we have not seen the end of."""
        return self.runs[0] if self.runs else None

    @property
    def active_id(self) -> str | None:
        run = self.active
        return run.prompt_id if run is not None else None

    @property
    def waiting(self) -> list:
        """The ones behind the active run, still in submission order."""
        return self.runs[1:]

    def holds(self, prompt_id: str) -> bool:
        return self.find(prompt_id) is not None
