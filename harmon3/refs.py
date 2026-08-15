"""Reference rows and the <Picture i> / <Video k> / <Audio j> tag algorithm.

The tag ordinals are not cosmetic: they are how the prompt addresses each reference, and
they are assigned by MiniMaxH3ReferenceToVideo.execute() purely from the *order and kind*
of the connected inputs. Toggling one soundtrack checkbox renumbers every standalone
<Audio j> that follows it, so this module also provides the diff and the simultaneous
rewrite the UI uses to keep an existing prompt pointing at the same assets.

No Qt, no network.
"""

from __future__ import annotations

import itertools
import mimetypes
import re
from dataclasses import dataclass, field
from pathlib import Path

from . import config

IMAGE = "image"
VIDEO = "video"
AUDIO = "audio"

KIND_LIMITS = {
    IMAGE: config.MAX_REF_IMAGES,
    VIDEO: config.MAX_REF_VIDEOS,
    AUDIO: config.MAX_REF_AUDIOS,
}

KIND_LABELS = {IMAGE: "image", VIDEO: "video", AUDIO: "audio"}

#: Extensions offered in the file dialogs. ComfyUI accepts anything whose MIME major type
#: matches, so these are a convenience filter rather than a hard constraint.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif", ".tiff", ".tif")
VIDEO_EXTENSIONS = (".mp4", ".webm", ".mkv", ".mov", ".avi", ".m4v")
AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac", ".opus")

EXTENSIONS_BY_KIND = {
    IMAGE: IMAGE_EXTENSIONS,
    VIDEO: VIDEO_EXTENSIONS,
    AUDIO: AUDIO_EXTENSIONS,
}


def kind_for_path(path) -> str | None:
    """Which kind of reference a file would be, by extension, or None if it is not one.

    Shared by every drop target so a file lands in the same place wherever it is dropped.
    """
    suffix = Path(str(path)).suffix.lower()
    for kind, extensions in EXTENSIONS_BY_KIND.items():
        if suffix in extensions:
            return kind
    return None

_uid_counter = itertools.count(1)

TAG_RE = re.compile(r"<\s*(Picture|Video|Audio)\s+(\d+)\s*>", re.IGNORECASE)

_CANONICAL_TAG_NAME = {"picture": "Picture", "video": "Video", "audio": "Audio"}

#: Subjects are matched separately from the three reference kinds -- see
#: ``subjects_in_prompt`` for why they are not simply added to ``TAG_RE``.
SUBJECT_RE = re.compile(r"<\s*(Subject)\s+(\d+)\s*>", re.IGNORECASE)


@dataclass
class RefRow:
    """One reference slot.

    A row is either a *local file* (``local_path`` set, needs uploading) or a *server
    file* (``comfy_name`` set with no local path -- a filename that already exists in
    ComfyUI's input directory). The server-file mode is what lets the app reproduce the
    shipped workflow's two baked LoadImage nodes without uploading anything.
    """

    kind: str
    local_path: str | None = None
    comfy_name: str | None = None
    use_soundtrack: bool = True          # videos only
    #: Send a skeleton rendition of this clip instead of the clip itself. Videos only.
    #: The estimator runs locally before the job is queued; see ``harmon3.pose``. The row
    #: keeps naming the user's own file either way -- the substitution happens on the job
    #: snapshot, so nothing persisted here is a derivative.
    use_pose: bool = False
    #: How much of this image is sent, as a percentage of its own size. Images only.
    #: A ceiling rather than a setting: the node never enlarges a reference, so an image
    #: shrunk on the way in stays shrunk -- see ``harmon3.scaling``. Like the pose clip,
    #: the rescaled copy is produced locally and substituted on the job snapshot, so the
    #: row itself always names the user's own file at its own full size.
    scale_percent: int = 100
    uid: int = field(default_factory=lambda: next(_uid_counter))

    #: Where the section sent to the model starts, in the unit its slicing node actually
    #: takes -- frames for a video, seconds for audio -- so nothing is lost rounding
    #: between them. Videos and audio only; 0 is the beginning of the file.
    #:
    #: There is no end and no on/off switch. Every video and audio reference is cut to the
    #: generated length, because that is what the model keeps of one either way, so the
    #: only thing left to decide is where the cut begins. See ``trim_length``.
    trim_start: float = 0.0

    #: Probe results, filled in asynchronously by the UI. None means "not probed yet".
    duration_s: float | None = None
    fps: float | None = None
    frame_count: int | None = None
    has_audio: bool | None = None
    width: int | None = None
    height: int | None = None

    #: The rendered pose clip for the section currently marked, once there is one, and the
    #: (start, length) it was rendered for. Neither is persisted: they point into a cache
    #: keyed by content, mark and length, and a stale pointer loaded from a scene would be
    #: worse than no pointer at all.
    #:
    #: The section is kept beside the path so staleness is a tuple comparison. Deriving it
    #: from the cache key instead would mean hashing the source file, and this is checked
    #: every time anything in the editor changes.
    pose_path: str | None = None
    pose_section: tuple[int, int] | None = None

    #: Set when the probe reports that nothing can decode this file. None means "not
    #: checked yet". A file that cannot be read here cannot be read by ComfyUI either, so
    #: this stops it before it is uploaded and fails several minutes later on the server.
    unreadable_reason: str | None = None

    #: For server-file rows: whether the named file is actually present on the server.
    #: None means "not checked yet". The workflow ships filenames that may not exist on
    #: the machine running ComfyUI, and a missing one is a 400 at submit time.
    server_missing: bool | None = None

    def __post_init__(self):
        if self.kind not in KIND_LIMITS:
            raise ValueError(f"Unknown reference kind {self.kind!r}")
        if self.local_path is None and self.comfy_name is None:
            raise ValueError("A reference row needs either a local path or a server filename")

    @property
    def needs_upload(self) -> bool:
        return self.local_path is not None

    @property
    def supports_trim(self) -> bool:
        """Only things with a timeline can be windowed."""
        return self.kind in (VIDEO, AUDIO)

    @property
    def supports_pose(self) -> bool:
        """Only a clip of a person has a skeleton to extract."""
        return self.kind == VIDEO

    @property
    def poses(self) -> bool:
        """Whether this row sends a skeleton instead of itself."""
        return self.supports_pose and self.use_pose

    @property
    def supports_scale(self) -> bool:
        """Images and videos, which mean different things by it.

        An image's ceiling is a share of *its own size*, because that is what the node
        caps. A video's is a share of the fixed canvas the node fits every clip to -- a
        share of the source would do nothing at all for the top of the slider's travel,
        since a 4K clip and a 1080p one are flattened onto the same 1344x768.
        """
        return self.kind in (IMAGE, VIDEO)

    @property
    def scales(self) -> bool:
        """Whether this row sends a rescaled copy rather than the file itself."""
        return self.supports_scale and int(self.scale_percent) < 100

    def scaled_size(self) -> tuple[int, int] | None:
        """The size this row will actually be sent at, once its dimensions are known."""
        from . import scaling

        if not (self.supports_scale and self.width and self.height):
            return None
        if self.kind == VIDEO:
            return (scaling.video_target_size(self.width, self.height, self.scale_percent)
                    if self.scales else scaling.video_canvas(self.width, self.height))
        if not self.scales:
            return int(self.width), int(self.height)
        return scaling.target_size(self.width, self.height, self.scale_percent)

    @property
    def marked(self) -> bool:
        """Whether this reference starts anywhere other than at its own beginning.

        Not a switch for whether it is cut -- everything with a timeline is -- only
        whether there is a mark worth drawing, saying or rewinding to.
        """
        return self.supports_trim and self.trim_start > 0

    def trim_length(self, target_frames: int) -> float:
        """How much is taken, in this reference's own unit.

        Not stored, and not markable: MiniMaxH3ReferenceToVideo truncates every reference
        to the generated length, so an out point beyond that could only ask for material
        that would be discarded, and one short of it would make the reference run out
        before the clip does. The length parameter is the out point.
        """
        if self.kind == AUDIO:
            return target_frames / config.FPS
        return float(target_frames)

    def trim_span(self, target_frames: int) -> tuple[float, float]:
        """(start, length) in this reference's own unit -- what the slicing node takes."""
        return self.trim_start, self.trim_length(target_frames)

    def trim_end(self, target_frames: int) -> float:
        """Where the section ends -- or where the reference does, if it runs out first.

        Clamped, because this is what gets drawn and read out: asking a loader for frames
        past the end of a file is harmless, but showing them as though they existed is not.
        """
        end = self.trim_start + self.trim_length(target_frames)
        limit = self.trim_limit()
        return min(end, limit) if limit > 0 else end

    def trim_seconds(self, target_frames: int) -> tuple[float, float] | None:
        """The section as (start, length) in seconds, or None if it cannot be expressed.

        A video's section is counted in frames of the *source*; turning it into seconds
        needs that file's frame rate, which comes from probing it. Without one there is
        nothing to convert with, and the caller has to say so rather than guess.
        """
        start, length = self.trim_span(target_frames)
        if self.kind == AUDIO:
            return start, length
        if not self.fps:
            return None
        return start / self.fps, length / self.fps

    def window_summary(self, target_frames: int) -> str:
        """The section in this reference's own unit: "24-148f (5.17s)" or "1.50-6.67s".

        Empty for an unmarked reference. Every one of them is cut to the same length, so
        saying "0-124f" on every row would be noise on the rows that say nothing.
        """
        if not self.marked:
            return ""
        end = self.trim_end(target_frames)
        if self.kind == VIDEO:
            # The clamped length, not the asked-for one: a clip that runs out early
            # should not read as though it had supplied the whole section.
            tail = f" ({(end - self.trim_start) / self.fps:.2f}s)" if self.fps else ""
            return f"{int(self.trim_start)}-{int(end)}f{tail}"
        return f"{self.trim_start:.2f}-{end:.2f}s"

    def trim_summary(self, target_frames: int) -> str:
        """The same section labelled, short enough to sit on the reference's own row."""
        if not self.marked:
            return ""
        return f"from {self.window_summary(target_frames)}"

    def trim_limit(self) -> float:
        """The furthest into this reference anything can reach, once probed."""
        if self.kind == VIDEO:
            return float(self.frame_count or 0)
        return float(self.duration_s or 0.0)

    def graph_name(self) -> str | None:
        """The filename the model should actually receive for this reference."""
        return self.comfy_name

    @property
    def local_missing(self) -> bool:
        """True when this row names a local file that is no longer there.

        Reloading a scene saved weeks ago is the case that makes this worth checking:
        the references may have been moved or deleted in the meantime.
        """
        return bool(self.local_path) and not Path(self.local_path).is_file()

    @property
    def display_name(self) -> str:
        if self.local_path:
            return Path(self.local_path).name
        return self.comfy_name or "?"

    def server_location(self) -> tuple[str, str]:
        """Split ``comfy_name`` into the (filename, subfolder) pair /view expects.

        /view takes the basename from ``filename`` and the directory from its own
        ``subfolder`` parameter, so "harmon3/x.png" has to be taken apart first.
        """
        name = (self.comfy_name or "").replace("\\", "/")
        subfolder, _, filename = name.rpartition("/")
        return filename, subfolder

    @property
    def mime_type(self) -> str:
        guessed, _ = mimetypes.guess_type(self.display_name)
        if guessed:
            return guessed
        return {IMAGE: "image/png", VIDEO: "video/mp4", AUDIO: "audio/wav"}[self.kind]

    def to_dict(self) -> dict:
        data = {
            "kind": self.kind,
            "local_path": self.local_path,
            "use_soundtrack": self.use_soundtrack,
            "use_pose": self.use_pose,
            "scale_percent": self.scale_percent,
            "trim_start": self.trim_start,
        }
        # `pose_path` is deliberately absent: it points into a cache keyed by this file's
        # contents, the mark and the generated length, any of which may have moved since.
        # Re-deriving it costs one hash; trusting a stale one costs a wrong run.
        # A local file's server name is derived from its current contents, so persisting
        # it would pin the row to whatever bytes happened to be uploaded first -- editing
        # the file in place would then silently keep submitting the old version. Only a
        # genuine server-file row (no local path) carries its name across sessions.
        if self.local_path is None:
            data["comfy_name"] = self.comfy_name
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "RefRow":
        row = cls(
            kind=data["kind"],
            local_path=data.get("local_path"),
            comfy_name=data.get("comfy_name"),
            use_soundtrack=bool(data.get("use_soundtrack", True)),
        )
        row.use_pose = bool(data.get("use_pose", False)) and row.supports_pose
        if row.supports_scale:
            from . import scaling
            row.scale_percent = scaling.clamp_percent(data.get("scale_percent", 100))
        # "trim_enabled" and "trim_end", written by older versions, are read and dropped:
        # the length comes from the generated length and the cut is unconditional, so
        # where it starts is the only part of that trio still worth keeping.
        if row.supports_trim:
            try:
                row.trim_start = max(0.0, float(data.get("trim_start", 0.0)))
            except (TypeError, ValueError):
                row.trim_start = 0.0

        return row


@dataclass
class RefSet:
    """The three ordered reference lists, which together determine every tag ordinal."""

    images: list[RefRow] = field(default_factory=list)
    videos: list[RefRow] = field(default_factory=list)
    audios: list[RefRow] = field(default_factory=list)

    def list_for(self, kind: str) -> list[RefRow]:
        return {IMAGE: self.images, VIDEO: self.videos, AUDIO: self.audios}[kind]

    def all_rows(self) -> list[RefRow]:
        return [*self.images, *self.videos, *self.audios]

    def is_empty(self) -> bool:
        return not (self.images or self.videos or self.audios)

    def can_add(self, kind: str) -> bool:
        return len(self.list_for(kind)) < KIND_LIMITS[kind]

    def to_list(self) -> list[dict]:
        return [row.to_dict() for row in self.all_rows()]

    @classmethod
    def from_list(cls, data) -> "RefSet":
        refset = cls()
        for entry in data or []:
            try:
                row = RefRow.from_dict(entry)
            except (KeyError, ValueError):
                continue
            target = refset.list_for(row.kind)
            if len(target) < KIND_LIMITS[row.kind]:
                target.append(row)
        return refset


@dataclass
class TagAssignment:
    """Which tag each row owns, plus the presentation order the model will see."""

    #: row uid -> its own tag (``<Picture 1>`` / ``<Video 1>`` / ``<Audio 2>``)
    by_uid: dict[int, str] = field(default_factory=dict)
    #: video row uid -> its soundtrack's ``<Audio j>`` tag, when the checkbox is on
    soundtrack_by_uid: dict[int, str] = field(default_factory=dict)
    #: every tag in the order the model presents them to the text encoder
    order: list[str] = field(default_factory=list)

    def tag_for(self, row: RefRow) -> str:
        return self.by_uid.get(row.uid, "")

    def soundtrack_tag_for(self, row: RefRow) -> str | None:
        return self.soundtrack_by_uid.get(row.uid)

    def all_tags(self) -> set[str]:
        return set(self.by_uid.values()) | set(self.soundtrack_by_uid.values())


def compute_tags(refset: RefSet) -> TagAssignment:
    """Assign tags exactly as MiniMaxH3ReferenceToVideo.execute() presents references.

    Order: every image, then each video -- with its soundtrack's ``<Audio j>`` label
    emitted *before* the video's own ``<Video k>`` label -- then every standalone audio.
    Ordinals are 1-based per type.
    """
    assignment = TagAssignment()
    picture_n = video_n = audio_n = 0

    for row in refset.images:
        picture_n += 1
        tag = f"<Picture {picture_n}>"
        assignment.by_uid[row.uid] = tag
        assignment.order.append(tag)

    for row in refset.videos:
        if row.use_soundtrack:
            audio_n += 1
            soundtrack = f"<Audio {audio_n}>"
            assignment.soundtrack_by_uid[row.uid] = soundtrack
            assignment.order.append(soundtrack)
        video_n += 1
        tag = f"<Video {video_n}>"
        assignment.by_uid[row.uid] = tag
        assignment.order.append(tag)

    for row in refset.audios:
        audio_n += 1
        tag = f"<Audio {audio_n}>"
        assignment.by_uid[row.uid] = tag
        assignment.order.append(tag)

    return assignment


def tag_migration(old: TagAssignment, new: TagAssignment) -> dict[str, str]:
    """Map old tag -> new tag for every row that kept its identity but changed ordinal.

    Rows are matched by uid, so this only reports genuine renumbering. Identity mappings
    are omitted, making an empty result mean "nothing moved".
    """
    migration: dict[str, str] = {}

    for uid, old_tag in old.by_uid.items():
        new_tag = new.by_uid.get(uid)
        if new_tag and new_tag != old_tag:
            migration[old_tag] = new_tag

    for uid, old_tag in old.soundtrack_by_uid.items():
        new_tag = new.soundtrack_by_uid.get(uid)
        if new_tag and new_tag != old_tag:
            migration[old_tag] = new_tag

    return migration


def remap_prompt(prompt: str, migration: dict[str, str]) -> str:
    """Apply a tag migration to a prompt in a single simultaneous pass.

    Chained ``str.replace`` calls corrupt overlapping migrations -- rewriting 1->2 and
    then 2->3 turns the original ``<Audio 1>`` into ``<Audio 3>``. Scanning once and
    substituting from the original text avoids that entirely.
    """
    if not migration:
        return prompt

    # Normalise the lookup so "< audio  1 >" matches "<Audio 1>".
    normalised = {_normalise_tag(k): v for k, v in migration.items()}

    def substitute(match: re.Match) -> str:
        canonical = _normalise_tag(match.group(0))
        return normalised.get(canonical, match.group(0))

    return TAG_RE.sub(substitute, prompt)


def _normalise_tag(raw: str) -> str:
    match = TAG_RE.fullmatch(raw.strip())
    if not match:
        return raw
    name = _CANONICAL_TAG_NAME[match.group(1).lower()]
    return f"<{name} {int(match.group(2))}>"


def tags_in_prompt(prompt: str) -> list[str]:
    """Every reference tag mentioned in the prompt, canonicalised, in order."""
    return [_normalise_tag(m.group(0)) for m in TAG_RE.finditer(prompt)]


def subjects_in_prompt(text: str) -> list[str]:
    """Every ``<Subject N>`` mentioned, canonicalised, in first-seen order.

    Kept out of ``TAG_RE`` on purpose. That regex and ``TagAssignment`` model the ordinals
    the node assigns from the order and kind of *connected inputs*; a subject has no row,
    no uid and no ordinal -- it is a name invented inside the prose. Folded in, it would
    make ``unknown_tags`` flag every ``<Subject 1>`` forever, because no assignment can
    ever contain one, and the real warning would be lost in the noise.
    """
    seen: list[str] = []
    for match in SUBJECT_RE.finditer(text or ""):
        tag = f"<Subject {int(match.group(2))}>"
        if tag not in seen:
            seen.append(tag)
    return seen


def defined_subjects(definitions: str) -> list[str]:
    """The subjects ``subject_definitions`` introduces, for the chips to offer elsewhere.

    Only the subjects named here are worth offering: a chip for one the prompt has not
    defined would hand out a label the model has never been told about.
    """
    return subjects_in_prompt(definitions)


def unknown_tags(prompt: str, assignment: TagAssignment) -> list[str]:
    """Tags the prompt references that no loaded reference provides.

    Non-blocking: the model tolerates them and the user may simply be mid-edit.
    """
    known = assignment.all_tags()
    seen: list[str] = []
    for tag in tags_in_prompt(prompt):
        if tag not in known and tag not in seen:
            seen.append(tag)
    return seen


def unused_tags(prompt: str, assignment: TagAssignment) -> list[str]:
    """References that are loaded but never mentioned in the prompt."""
    mentioned = set(tags_in_prompt(prompt))
    return [tag for tag in assignment.order if tag not in mentioned]


def row_warnings(row: RefRow, target_frames: int) -> tuple[list[str], list[str]]:
    """Return (errors, warnings) for a probed reference row.

    Errors block the queue; warnings are advisory. Probe fields left as None simply
    produce no diagnostics, so an unprobed row is never blocked.
    """
    errors: list[str] = []
    warnings: list[str] = []

    if row.server_missing:
        errors.append(
            "not present in ComfyUI's input folder - pick a local file with the ... button"
        )

    if row.local_missing:
        errors.append(f"file no longer exists: {row.local_path}")

    if row.unreadable_reason:
        errors.append(f"this file cannot be read as {KIND_LABELS[row.kind]} "
                      f"- {row.unreadable_reason}")

    if row.marked and row.kind == AUDIO:
        warnings.append(f"only {row.trim_start:.2f}s to "
                        f"{row.trim_end(target_frames):.2f}s is sent to the model")

    if row.kind != VIDEO:
        return errors, warnings

    if row.marked:
        warnings.append(f"starts at frame {int(row.trim_start)} - frames "
                        f"{int(row.trim_start)} to {int(row.trim_end(target_frames))} "
                        f"are sent to the model")
        # No frame-rate check here any more: the loader windows the frames and the sound
        # from the same two inputs, using the source's own frame time, so the two cannot
        # drift apart and nothing has to be converted on this side.

    if row.poses:
        # Advisory, never blocking: queueing renders whatever is missing before it
        # uploads, so "not rendered yet" is a statement about now, not about readiness.
        warnings.append(
            "a skeleton of this clip is sent instead of the clip"
            if row.pose_path else
            "a skeleton of this clip is sent instead of the clip - it is rendered when "
            "you queue, or click the POSE? thumbnail to see it first"
        )

    if row.frame_count is not None:
        # What is left of the file after the mark is what the model can have; a section
        # that starts late has less of the reference behind it than the file suggests.
        available = max(0, row.frame_count - int(row.trim_start))
        # MiniMaxH3ReferenceToVideo truncates to the generated length, then trims down to
        # the nearest n % 17 == 5, and raises below 5 frames.
        usable = min(available, target_frames)
        while usable % config.FRAME_MOD != config.FRAME_REM and usable > 0:
            usable -= 1
        if available < config.MIN_REF_VIDEO_FRAMES or usable < config.MIN_REF_VIDEO_FRAMES:
            errors.append(
                f"only {available} frames - the model needs at least "
                f"{config.MIN_REF_VIDEO_FRAMES}"
            )
        elif available > target_frames:
            warnings.append(
                f"{available} frames will be truncated to {usable} "
                f"({usable / config.FPS:.1f} s)"
            )
        elif available < target_frames:
            # Not an error: the model accepts a short reference. It is worth saying,
            # because the fix is to start earlier or generate less, and neither is obvious
            # from a window that simply stops before the clip does.
            warnings.append(
                f"only {available} frames left here, so this reference runs out before "
                f"the {target_frames}-frame clip does"
            )

    if row.fps is not None and abs(row.fps - config.FPS) > 0.5:
        warnings.append(
            f"source is {row.fps:.3g} fps but the model assumes {config.FPS} fps, "
            "so this reference will play at the wrong speed"
        )

    if row.use_soundtrack and row.has_audio is False:
        warnings.append(
            "no audio track, so its <Audio> tag will not be emitted - "
            "uncheck 'use soundtrack' to keep the remaining audio ordinals correct"
        )

    return errors, warnings
