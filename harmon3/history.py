"""Run history: an append-only JSONL log plus a local cache of the produced videos.

Append-only means a crash mid-write can damage at most the record being added, never the
ones already on disk. Each record stores the exact graph that was submitted, which makes
re-queue a verbatim resubmission rather than a re-derivation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

from . import config

log = logging.getLogger(__name__)

#: Accepted by the server and waiting behind something else.
STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_SUCCESS = "success"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

MAX_RECORDS = 200


@dataclass
class RunRecord:
    prompt_id: str
    submitted_at: str
    status: str = STATUS_QUEUED
    finished_at: str | None = None

    prompt_text: str = ""
    #: The prompt's sections, so restoring a run puts each part back in its own box.
    #: Records written before the prompt was split have only prompt_text.
    prompt_sections: dict = field(default_factory=dict)
    aspect_ratio: str = ""
    megapixels: float = 0.0
    width: int = 0
    height: int = 0
    duration_seconds: float = 0.0
    frames: int = 0
    seed: int = 0
    steps: int = 0
    #: Empty on records written before these were exposed; a falsy value means "unknown",
    #: and restoring one leaves the current setting standing.
    sampler_name: str = ""
    scheduler: str = ""
    schedule: str = ""
    upscale_method: str = ""
    shift_video: float = 0.0
    ref_image_size: str = ""

    #: [{kind, name, tag, soundtrack_tag}]
    refs: list = field(default_factory=list)
    output: dict | None = None
    local_path: str | None = None
    elapsed_s: float | None = None
    error: str | None = None
    graph: dict | None = None

    @property
    def submitted_dt(self) -> datetime | None:
        try:
            return datetime.fromisoformat(self.submitted_at)
        except (TypeError, ValueError):
            return None

    def summary(self) -> str:
        """A one-line preview, built from what was actually written.

        The stored prompt carries every section including the empty ones, which would
        otherwise fill the history column with N/A.
        """
        from . import prompt as prompt_mod

        sections = self.prompt_sections or prompt_mod.parse(self.prompt_text)
        filled = prompt_mod.normalise(sections)
        text = " ".join(
            " ".join(filled[name].split()) for name in prompt_mod.filled_names(filled))
        return (text[:60] + "...") if len(text) > 60 else (text or "(no prompt)")

    def video_path(self) -> Path | None:
        if not self.local_path:
            return None
        path = Path(self.local_path)
        if not path.is_absolute():
            path = config.HOME / path
        return path if path.is_file() else None

    def to_json(self) -> str:
        return json.dumps(asdict(self), ensure_ascii=False)

    @classmethod
    def from_dict(cls, data: dict) -> "RunRecord":
        known = {k: v for k, v in data.items() if k in cls.__dataclass_fields__}
        known.setdefault("prompt_id", "")
        known.setdefault("submitted_at", "")
        return cls(**known)


class HistoryStore:
    """Reads the whole log at startup, appends single records thereafter."""

    def __init__(self, path: Path | None = None, video_dir: Path | None = None):
        self.path = path or config.RUNS_JSONL
        self.video_dir = video_dir or config.VIDEO_CACHE_DIR
        self.records: list[RunRecord] = []

    def load(self) -> list[RunRecord]:
        self.records = []
        if not self.path.is_file():
            return self.records

        with self.path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    self.records.append(RunRecord.from_dict(json.loads(line)))
                except (ValueError, TypeError) as exc:
                    # One damaged line must not cost the user the rest of their history.
                    log.warning("Skipping malformed history line %d: %s", line_no, exc)

        # A run still marked running or queued is a leftover from a crash or a forced quit:
        # whatever became of it, this app was not watching, so it cannot claim it finished.
        for record in self.records:
            if record.status in (STATUS_RUNNING, STATUS_QUEUED):
                record.status = STATUS_FAILED
                record.error = record.error or "Interrupted - HARMON3 closed while this run was active"

        self.records = self.records[-MAX_RECORDS:]
        return self.records

    def append(self, record: RunRecord) -> None:
        self.records.append(record)
        self._write_line(record)

    def update(self, record: RunRecord) -> None:
        """Persist a changed record by rewriting the log.

        Records change at most twice (queued, then finished), so a rewrite of a few
        hundred short lines is cheaper than any indexing scheme would be to maintain.
        """
        self.records = self.records[-MAX_RECORDS:]
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = self.path.with_suffix(".jsonl.tmp")
        try:
            with temp_path.open("w", encoding="utf-8") as fh:
                for entry in self.records:
                    fh.write(entry.to_json() + "\n")
            temp_path.replace(self.path)
        except OSError as exc:
            log.error("Could not rewrite history: %s", exc)
            temp_path.unlink(missing_ok=True)

    def find(self, prompt_id: str) -> RunRecord | None:
        for record in reversed(self.records):
            if record.prompt_id == prompt_id:
                return record
        return None

    def video_path_for(self, prompt_id: str) -> Path:
        return self.video_dir / f"{prompt_id}.mp4"

    def relative(self, path: Path) -> str:
        try:
            return str(path.relative_to(config.HOME))
        except ValueError:
            return str(path)

    def _write_line(self, record: RunRecord) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with self.path.open("a", encoding="utf-8") as fh:
                fh.write(record.to_json() + "\n")
        except OSError as exc:
            log.error("Could not append to history: %s", exc)


def refs_snapshot(refset, tags) -> list[dict]:
    """Freeze the reference list and its tag assignment into the record."""
    snapshot = []
    for row in refset.all_rows():
        snapshot.append({
            "kind": row.kind,
            "name": row.display_name,
            "comfy_name": row.comfy_name,
            "tag": tags.tag_for(row),
            "soundtrack_tag": tags.soundtrack_tag_for(row),
            # What the model actually received: a posed row sent a skeleton, and a record
            # that did not say so would restore into something that behaves differently.
            "use_pose": bool(getattr(row, "use_pose", False)),
        })
    return snapshot
