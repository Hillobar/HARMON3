"""Named, reusable scenes, grouped into projects.

A scene captures what makes a shot *that shot* -- its references, prompt, length and seed
settings -- so it can be reloaded, rerun or adjusted later. Resolution is deliberately
not part of it: it is a render setting, and the same scene is often wanted at draft size
first and full size afterwards.

A *project* is the finished video those scenes are pieces of. It is a name and a running
order, and it is carried **on the scenes themselves** -- ``project`` and ``project_index``
-- rather than by subfolders or a membership list:

* a scene file stays self-describing, so copying one to a colleague or into version
  control brings its place in the sequence with it;
* moving a scene between projects is a field change rather than a file move that can
  half-fail;
* and the rule that one damaged file cannot take the catalogue with it still holds, which
  a central membership list would quietly undo.

The one thing that cannot live on a scene is a project with no scenes in it yet -- there
is nothing to write it on. So ``projects.json`` beside the scenes records *names only*,
never membership. Losing it loses empty projects and nothing else.

No Qt, no network.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import tempfile
import unicodedata
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import config, prompt as prompt_mod
from .refs import RefSet

log = logging.getLogger(__name__)

SCHEMA_VERSION = 1
MAX_NAME_LENGTH = 80
MAX_DESCRIPTION_LENGTH = 200
MAX_PROJECT_LENGTH = 80

#: A scene belonging to no project. The empty string rather than None, so a scene written
#: by an older build -- which has no such field at all -- reads as ungrouped rather than
#: as something needing migration.
UNGROUPED = ""

#: Where the names of projects that have no scenes yet are kept. Names only; membership is
#: always read off the scenes.
PROJECTS_FILENAME = "projects.json"


def clean_project(name) -> str:
    """Collapse a project name to a single trimmed line within the length limit."""
    return " ".join(str(name or "").split())[:MAX_PROJECT_LENGTH]

_SLUG_STRIP = re.compile(r"[^a-z0-9]+")
#: Reserved on Windows regardless of extension.
_RESERVED_STEMS = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def clean_description(text) -> str:
    """Collapse a description to a single trimmed line within the length limit."""
    return " ".join(str(text or "").split())[:MAX_DESCRIPTION_LENGTH]


def slugify(name: str) -> str:
    """A filesystem-safe stem for a scene name."""
    normalised = unicodedata.normalize("NFKD", name or "")
    ascii_only = normalised.encode("ascii", "ignore").decode("ascii").lower()
    slug = _SLUG_STRIP.sub("-", ascii_only).strip("-")[:60]
    if not slug or slug in _RESERVED_STEMS:
        slug = f"scene-{slug}" if slug else "scene"
    return slug


@dataclass
class Scene:
    """One saved shot definition."""

    name: str
    #: A short human note about what this scene is for. Owned by the scene rather than
    #: the editor, so it is saved the moment it is edited and never goes "modified".
    description: str = ""
    #: The finished video this scene is a piece of, and where it falls in it. Both are
    #: deliberately outside `signature()`: reordering a project is not an edit to any of
    #: the scenes in it, and marking them all modified would be nonsense.
    project: str = UNGROUPED
    project_index: int = 0
    #: The prompt's named sections. prompt_text is derived from them.
    prompt_sections: dict = field(default_factory=lambda: prompt_mod.empty_sections())
    duration_seconds: float = config.DEFAULT_DURATION
    seed: int = config.DEFAULT_SEED
    randomize_seed: bool = True
    #: RefRow.to_dict() entries, in presentation order
    refs: list = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""

    #: Where this scene lives. Assigned by the store; never serialised.
    path: Path | None = field(default=None, compare=False, repr=False)

    # -- conversion ----------------------------------------------------------------

    @classmethod
    def from_state(cls, name: str, state, randomize_seed: bool,
                   description: str = "") -> "Scene":  # noqa: D401
        return cls(
            name=name.strip()[:MAX_NAME_LENGTH] or "Untitled scene",
            description=clean_description(description),
            prompt_sections=dict(state.prompt_sections),
            duration_seconds=float(state.duration_seconds),
            seed=int(state.seed),
            randomize_seed=bool(randomize_seed),
            refs=state.refs.to_list(),
        )

    @property
    def prompt_text(self) -> str:
        return prompt_mod.combine(self.prompt_sections)

    def apply_to_state(self, state) -> None:
        """Load this scene into an editor state, leaving render settings alone."""
        state.prompt_sections = dict(self.prompt_sections)
        state.duration_seconds = float(self.duration_seconds)
        state.seed = int(self.seed)
        state.refs = RefSet.from_list(self.refs)

    def capture_from_state(self, state, randomize_seed: bool) -> None:
        """Overwrite this scene's definition with the editor's current contents."""
        self.prompt_sections = dict(state.prompt_sections)
        self.duration_seconds = float(state.duration_seconds)
        self.seed = int(state.seed)
        self.randomize_seed = bool(randomize_seed)
        self.refs = state.refs.to_list()

    # -- comparison ----------------------------------------------------------------

    def signature(self) -> tuple:
        """What "unchanged" means, for the modified marker.

        Compares only the fields a scene owns, so switching resolution -- which a scene
        deliberately does not capture -- never makes one look edited.
        """
        return signature_of(
            self.prompt_text, self.duration_seconds, self.seed,
            self.randomize_seed, self.refs,
        )

    def matches_state(self, state, randomize_seed: bool) -> bool:
        return self.signature() == signature_of(
            state.prompt_text, state.duration_seconds, state.seed,
            randomize_seed, state.refs.to_list(),
        )

    # -- description ---------------------------------------------------------------

    def ref_counts(self) -> dict[str, int]:
        counts = {"image": 0, "video": 0, "audio": 0}
        for entry in self.refs:
            kind = entry.get("kind")
            if kind in counts:
                counts[kind] += 1
        return counts

    def ref_summary(self) -> str:
        counts = self.ref_counts()
        parts = [f"{n} {kind}" for kind, n in counts.items() if n]
        return ", ".join(parts) if parts else "no references"

    def prompt_summary(self, limit: int = 90) -> str:
        """A one-line preview for the catalogue.

        Built from the sections that were actually written, so an empty scene does not
        read as a wall of N/A.
        """
        filled = prompt_mod.normalise(self.prompt_sections)
        text = " ".join(
            " ".join(filled[name].split())
            for name in prompt_mod.filled_names(filled)
        ).strip()
        if not text:
            return "(no prompt)"
        return text[: limit - 3] + "..." if len(text) > limit else text

    def missing_local_files(self) -> list[str]:
        """Referenced files that are no longer on disk.

        Worth knowing before a reload: a scene saved weeks ago can easily outlive the
        folder its references came from.
        """
        missing = []
        for entry in self.refs:
            path = entry.get("local_path")
            if path and not Path(path).is_file():
                missing.append(path)
        return missing

    # -- serialisation -------------------------------------------------------------

    def to_dict(self) -> dict:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "description": self.description,
            "project": self.project,
            "project_index": self.project_index,
            "prompt_sections": dict(self.prompt_sections),
            "duration_seconds": self.duration_seconds,
            "seed": self.seed,
            "randomize_seed": self.randomize_seed,
            "refs": self.refs,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict, path: Path | None = None) -> "Scene":
        if not isinstance(data, dict):
            raise ValueError("a scene file must contain a JSON object")

        name = str(data.get("name") or (path.stem if path else "")).strip()
        if not name:
            raise ValueError("a scene needs a name")

        try:
            index = int(data.get("project_index", 0))
        except (TypeError, ValueError):    # hand-edited, or written by something else
            index = 0

        refs = data.get("refs")
        scene = cls(
            name=name[:MAX_NAME_LENGTH],
            description=clean_description(data.get("description")),
            project=clean_project(data.get("project")),
            project_index=max(0, index),
            prompt_sections=(
                prompt_mod.normalise(data["prompt_sections"])
                if isinstance(data.get("prompt_sections"), dict)
                # A scene saved before the prompt was split into sections.
                else prompt_mod.from_legacy(str(data.get("prompt_text") or ""))
            ),
            duration_seconds=float(data.get("duration_seconds", config.DEFAULT_DURATION)),
            seed=int(data.get("seed", config.DEFAULT_SEED)),
            randomize_seed=bool(data.get("randomize_seed", True)),
            refs=refs if isinstance(refs, list) else [],
            created_at=str(data.get("created_at") or ""),
            updated_at=str(data.get("updated_at") or ""),
        )
        scene.path = path
        return scene


def signature_of(prompt_text, duration_seconds, seed, randomize_seed, refs) -> tuple:
    """Normalised comparison key shared by Scene.signature and the editor state.

    Two things are deliberately excluded, because treating either as an edit would make
    the modified marker cry wolf:

      * reference fields that are discovered rather than chosen (probe results, the
        upload name resolved from a file's contents);
      * the seed value itself when the scene randomizes -- every queue rolls a new one,
        so for a randomized scene "randomize" *is* the setting, not the number.
    """
    normalised_refs = tuple(
        (
            entry.get("kind"),
            entry.get("local_path"),
            entry.get("comfy_name"),
            bool(entry.get("use_soundtrack", True)),
            bool(entry.get("use_pose", False)),
            int(entry.get("scale_percent", 100)),
        )
        for entry in refs or []
    )
    randomize = bool(randomize_seed)
    return (
        prompt_text or "",
        round(float(duration_seconds), 4),
        None if randomize else int(seed),
        randomize,
        normalised_refs,
    )


class SceneStore:
    """The catalogue: one JSON file per scene in a directory."""

    def __init__(self, directory: Path | None = None):
        self.dir = directory or config.SCENES_DIR
        self.scenes: list[Scene] = []
        #: Project names in display order. Only the empty ones actually *need* recording;
        #: the rest are here so that order survives too.
        self.projects: list[str] = []

    def load_all(self) -> list[Scene]:
        self.scenes = []
        if not self.dir.is_dir():
            self.projects = []
            return self.scenes

        for path in sorted(self.dir.glob("*.json")):
            if path.name == PROJECTS_FILENAME:
                continue
            try:
                with path.open("r", encoding="utf-8") as fh:
                    self.scenes.append(Scene.from_dict(json.load(fh), path))
            except (OSError, ValueError) as exc:
                # One damaged file must not cost the user the rest of the catalogue.
                log.warning("Skipping unreadable scene %s: %s", path.name, exc)

        self._load_projects()
        self._sort()
        return self.scenes

    # -- projects ------------------------------------------------------------------

    def project_names(self) -> list[str]:
        """Every project, in display order, including any that have no scenes yet."""
        return list(self.projects)

    def scenes_in(self, project: str) -> list[Scene]:
        """The scenes of one project, in running order."""
        project = clean_project(project)
        members = [s for s in self.scenes if s.project == project]
        members.sort(key=lambda s: (s.project_index, s.name.casefold()))
        return members

    def ungrouped(self) -> list[Scene]:
        return self.scenes_in(UNGROUPED)

    def create_project(self, name: str) -> str:
        """Register a project, returning the name actually used ("" if it was blank).

        Recorded even with nothing in it, because a project you cannot see is a project
        you cannot drag a scene onto -- which is the only way scenes get into one.
        """
        name = clean_project(name)
        if not name:
            return UNGROUPED
        name = self._unique_project_name(name)
        self.projects.append(name)
        self._save_projects()
        return name

    def rename_project(self, old: str, new: str) -> str:
        """Rename a project and everything filed under it. Returns the name used."""
        old, new = clean_project(old), clean_project(new)
        if not old or not new or old == new:
            return old
        new = self._unique_project_name(new, exclude=old)

        for scene in self.scenes_in(old):
            scene.project = new
            self._write(scene)
        self.projects = [new if p == old else p for p in self.projects]
        if new not in self.projects:
            self.projects.append(new)
        self._save_projects()
        self._sort()
        return new

    def delete_project(self, name: str) -> int:
        """Drop a project, returning its scenes to ungrouped. Never deletes a scene.

        A project is a way of arranging work, not a container that owns it -- so removing
        the arrangement must not be able to take hours of work with it.
        """
        name = clean_project(name)
        if not name:
            return 0
        released = self.scenes_in(name)
        for scene in released:
            self.set_project(scene, UNGROUPED, save=True)
        self.projects = [p for p in self.projects if p != name]
        self._save_projects()
        return len(released)

    def set_project(self, scene: Scene, project: str, index: int | None = None,
                    save: bool = True) -> Scene:
        """Move a scene into a project at ``index``, or onto the end.

        The whole destination is renumbered afterwards rather than the moved scene being
        given a fractional or duplicate index, so the order on disk always reads 0, 1, 2
        and a file edited by hand cannot leave two scenes fighting over one slot.
        """
        project = clean_project(project)
        if project and project not in self.projects:
            self.projects.append(project)
            self._save_projects()

        source = scene.project
        remaining = [s for s in self.scenes_in(project) if s is not scene]
        position = len(remaining) if index is None else max(0, min(int(index), len(remaining)))
        remaining.insert(position, scene)

        scene.project = project
        # Keyed by identity: Scene is a comparable dataclass, so two scenes that happen to
        # hold the same shot are equal -- and unhashable besides.
        touched = {id(scene): scene}
        for order, member in enumerate(remaining):
            if member.project_index != order:
                member.project_index = order
                touched[id(member)] = member

        if source != project:
            touched.update(self._renumber(source))

        if save:
            for member in touched.values():
                self._write_quietly(member)
        self._sort()
        return scene

    def move_within_project(self, scene: Scene, delta: int) -> Scene:
        """Nudge a scene up or down its project's running order."""
        members = self.scenes_in(scene.project)
        try:
            position = members.index(scene)
        except ValueError:
            return scene
        return self.set_project(scene, scene.project, position + delta)

    def _renumber(self, project: str) -> dict:
        """Close the gap a departing scene left. Returns the scenes that changed."""
        touched = {}
        for order, member in enumerate(self.scenes_in(project)):
            if member.project_index != order:
                member.project_index = order
                touched[id(member)] = member
        return touched

    def _unique_project_name(self, base: str, exclude: str | None = None) -> str:
        existing = {p.casefold() for p in self.projects if p != exclude}
        existing |= {s.project.casefold() for s in self.scenes
                     if s.project and s.project != exclude}
        if base.casefold() not in existing:
            return base
        for n in range(2, 1000):
            candidate = clean_project(f"{base} {n}")
            if candidate.casefold() not in existing:
                return candidate
        return base

    def _projects_path(self) -> Path:
        return self.dir / PROJECTS_FILENAME

    def _load_projects(self) -> None:
        """Read the registry, then fold in every project the scenes themselves name.

        The scenes are the authority on membership, so a project mentioned by a scene but
        missing from the registry is not an error -- it is what happens when a scene file
        arrives from somewhere else, and it should simply appear.
        """
        recorded: list[str] = []
        try:
            with self._projects_path().open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            names = data.get("projects") if isinstance(data, dict) else data
            for name in names if isinstance(names, list) else []:
                cleaned = clean_project(name)
                if cleaned and cleaned not in recorded:
                    recorded.append(cleaned)
        except (OSError, ValueError) as exc:
            if self._projects_path().exists():
                log.warning("Could not read %s: %s", PROJECTS_FILENAME, exc)

        for scene in self.scenes:
            if scene.project and scene.project not in recorded:
                recorded.append(scene.project)
        self.projects = recorded

    def _save_projects(self) -> None:
        """Write the registry. Best effort: losing it costs empty projects and no work."""
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
            handle, temp_name = tempfile.mkstemp(
                dir=str(self.dir), prefix=".projects-", suffix=".tmp")
            with os.fdopen(handle, "w", encoding="utf-8") as fh:
                json.dump({"schema_version": SCHEMA_VERSION, "projects": self.projects},
                          fh, indent=2, ensure_ascii=False)
            os.replace(temp_name, self._projects_path())
        except OSError as exc:
            log.warning("Could not write %s: %s", PROJECTS_FILENAME, exc)

    def _write_quietly(self, scene: Scene) -> None:
        """Persist one scene of a reorder, without letting one failure abandon the rest.

        A drag touches every scene below the drop. Raising on the third would leave the
        first two renumbered on disk and the rest not, which is worse than a warning and
        a catalogue that is right again after the next load.
        """
        if scene.path is None:
            scene.path = self._free_path(scene.name)
        scene.updated_at = datetime.now().isoformat(timespec="seconds")
        try:
            self._write(scene)
        except SceneWriteError as exc:
            log.warning("Could not record the new order of %s: %s", scene.name, exc)

    def save(self, scene: Scene) -> Scene:
        """Write a scene, assigning it a file the first time."""
        now = datetime.now().isoformat(timespec="seconds")
        scene.created_at = scene.created_at or now
        scene.updated_at = now

        if scene.path is None:
            scene.path = self._free_path(scene.name)
        self._write(scene)

        if scene not in self.scenes:
            self.scenes.append(scene)
        self._sort()
        return scene

    def set_details(self, scene: Scene, new_name: str, description: str) -> Scene:
        """Change a scene's name and description, moving its file if the name changed.

        Both are properties of the scene rather than of the editor, so this writes
        immediately -- there is nothing here that could sit unsaved.
        """
        new_name = new_name.strip()[:MAX_NAME_LENGTH] or scene.name
        renamed = new_name != scene.name

        old_path = scene.path
        scene.name = new_name
        scene.description = clean_description(description)
        if renamed or scene.path is None:
            scene.path = self._free_path(new_name, exclude=old_path)
        scene.updated_at = datetime.now().isoformat(timespec="seconds")
        self._write(scene)

        if old_path and old_path != scene.path:
            try:
                old_path.unlink(missing_ok=True)
            except OSError as exc:
                log.warning("Could not remove the old scene file %s: %s", old_path, exc)

        self._sort()
        return scene

    def rename(self, scene: Scene, new_name: str) -> Scene:
        return self.set_details(scene, new_name, scene.description)

    def duplicate(self, scene: Scene) -> Scene:
        copy = Scene.from_dict(scene.to_dict())
        copy.name = self._unique_name(f"{scene.name} copy")
        copy.created_at = ""
        copy.path = None
        self.save(copy)
        # Directly after its original rather than at the end: a duplicate is nearly always
        # the next take of that shot, not the last shot of the video.
        if copy.project:
            self.set_project(copy, copy.project, scene.project_index + 1)
        return copy

    def delete(self, scene: Scene) -> None:
        if scene.path:
            try:
                scene.path.unlink(missing_ok=True)
            except OSError as exc:
                log.error("Could not delete %s: %s", scene.path, exc)
                return
        if scene in self.scenes:
            self.scenes.remove(scene)
        # Close the gap, so the order on disk stays 0, 1, 2 rather than growing holes
        # that a later insert would land in unpredictably.
        for member in self._renumber(scene.project).values():
            self._write_quietly(member)

    def move_to(self, new_dir: Path) -> tuple[int, list[str]]:
        """Move every scene file into ``new_dir``. Returns (moved count, problems).

        A file whose name is already taken in the destination is left alone and reported
        rather than silently overwriting whatever is there.
        """
        problems: list[str] = []
        if new_dir == self.dir or not self.dir.is_dir():
            self.dir = new_dir
            return 0, problems

        try:
            new_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return 0, [f"Could not create {new_dir}: {exc}"]

        moved = 0
        for path in sorted(self.dir.glob("*.json")):
            target = new_dir / path.name
            if target.exists():
                problems.append(f"{path.name} already exists there and was left behind")
                continue
            try:
                shutil.move(str(path), str(target))
                moved += 1
            except OSError as exc:
                problems.append(f"{path.name}: {exc}")

        self.dir = new_dir
        return moved, problems

    def count_files(self) -> int:
        """Scene files on disk. The project registry sits among them and is not one."""
        if not self.dir.is_dir():
            return 0
        return len([p for p in self.dir.glob("*.json") if p.name != PROJECTS_FILENAME])

    def writable(self) -> str | None:
        """None if the folder can be written to, otherwise why not."""
        try:
            self.dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return f"cannot create it ({exc.strerror or exc})"

        probe = self.dir / ".harmon3-write-test"
        try:
            probe.write_text("", encoding="utf-8")
            probe.unlink()
        except OSError as exc:
            return f"cannot write to it ({exc.strerror or exc})"
        return None

    def name_exists(self, name: str, exclude: Scene | None = None) -> bool:
        target = (name or "").strip().casefold()
        return any(
            scene is not exclude and scene.name.casefold() == target
            for scene in self.scenes
        )

    def find(self, name: str) -> Scene | None:
        target = (name or "").strip().casefold()
        for scene in self.scenes:
            if scene.name.casefold() == target:
                return scene
        return None

    # -- internals -----------------------------------------------------------------

    def _sort(self) -> None:
        self.scenes.sort(key=lambda s: s.name.casefold())

    def _unique_name(self, base: str) -> str:
        if not self.name_exists(base):
            return base[:MAX_NAME_LENGTH]
        for n in range(2, 1000):
            candidate = f"{base} {n}"[:MAX_NAME_LENGTH]
            if not self.name_exists(candidate):
                return candidate
        return base[:MAX_NAME_LENGTH]

    def _free_path(self, name: str, exclude: Path | None = None) -> Path:
        """A file path not already taken by a different scene.

        Two scenes can legitimately slugify to the same stem ("Take 1" and "take-1"), so
        the stem is suffixed until it is free.
        """
        stem = slugify(name)
        taken = {s.path for s in self.scenes if s.path and s.path != exclude}

        candidate = self.dir / f"{stem}.json"
        counter = 2
        while candidate in taken or (candidate.exists() and candidate != exclude):
            candidate = self.dir / f"{stem}-{counter}.json"
            counter += 1
        return candidate

    def _write(self, scene: Scene) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        handle, temp_name = tempfile.mkstemp(
            dir=str(self.dir), prefix=".scene-", suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as fh:
                json.dump(scene.to_dict(), fh, indent=2, ensure_ascii=False)
            os.replace(temp_name, scene.path)
        except Exception as exc:
            log.error("Could not write %s: %s", scene.path, exc)
            Path(temp_name).unlink(missing_ok=True)
            raise SceneWriteError(str(exc)) from exc


class SceneWriteError(Exception):
    """A scene could not be written to disk."""
