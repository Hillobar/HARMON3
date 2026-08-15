"""Exporting exactly what ``MiniMaxH3ReferenceToVideo`` is about to be given.

Every reference the model receives has been through several hands by the time it gets
there: a section cut from a clip, a skeleton drawn from it, an image rescaled, an EXIF
rotation applied, and then the node's own sizing on the far side.
Each of those is easy to reason about on its own and hard to hold all at once, and when a
result comes back wrong the first question is always the same -- *what did it actually
see?*

So this writes the answer to disk: the real files, under the tag each one is addressed by
in the prompt, the prompt itself as the node receives it, and a manifest saying what the
node will do to each file after it arrives.

The files are taken from the **job snapshot**, after the pose and scaling substitutions,
which is the same object the uploader works from -- not from the rows on screen. That is
the point: a posed row on screen still names the user's own clip, and exporting *that*
would show the one thing the model never sees.

No Qt, no network.
"""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from . import config, mathmirror, prompt as prompt_mod, refs as refs_mod, scaling
from .refs import AUDIO, IMAGE, VIDEO

log = logging.getLogger(__name__)

MANIFEST_NAME = "manifest.json"
PROMPT_NAME = "prompt.txt"
README_NAME = "README.txt"

#: The tag kinds, in the order the node numbers them, for a stable filename prefix.
_KIND_ORDER = {IMAGE: 1, VIDEO: 2, AUDIO: 3}


class BundleError(RuntimeError):
    """The bundle could not be written, phrased for the user."""


@dataclass
class BundleResult:
    directory: Path
    copied: list[str] = field(default_factory=list)
    #: Rows that should have had a file and did not, with why.
    missing: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing


def _slug(text: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "_" for c in str(text))[:60]


def _tagged_name(tag: str, row, index: int) -> str:
    """``01_Picture_1_hero.png`` -- sorts in tag order and still names the file."""
    stem = Path(row.local_path or row.comfy_name or "reference").stem
    return f"{_KIND_ORDER.get(row.kind, 9)}{index:02d}_{_slug(tag.strip('<>'))}_{_slug(stem)}"


def _prepare(directory: Path) -> None:
    """Empty the bundle folder, having first checked it is one.

    A folder with no manifest in it is not something this wrote, and clearing it because
    someone pointed the app at their pictures directory would be unforgivable. An empty
    or absent folder is fine; anything else has to prove itself.
    """
    if directory.exists():
        if not directory.is_dir():
            raise BundleError(f"{directory} is a file, not a folder")
        contents = list(directory.iterdir())
        if contents and not (directory / MANIFEST_NAME).is_file():
            raise BundleError(
                f"{directory} has things in it that this did not write - "
                "move it aside, or empty it yourself")
        for item in contents:
            try:
                shutil.rmtree(item) if item.is_dir() else item.unlink()
            except OSError as exc:
                raise BundleError(f"could not clear {item.name}: {exc}") from exc
    directory.mkdir(parents=True, exist_ok=True)


def measure(path: Path, kind: str) -> tuple[int, int] | None:
    """The real dimensions of a file in the bundle, or None if they cannot be read.

    Measured rather than taken off the row, which carries the *original*: by this point a
    reference may have been rescaled, posed or rotated, and the row's numbers
    describe the file that went in rather than the one going out. The whole value of the
    manifest is that these two are allowed to differ and it says so.
    """
    size = None
    try:
        if kind == IMAGE:
            from . import imaging

            size = imaging.oriented_size(path)
        elif kind == VIDEO:
            import av

            with av.open(str(path)) as container:
                if container.streams.video:
                    stream = container.streams.video[0]
                    size = (stream.codec_context.width, stream.codec_context.height)
    except Exception as exc:                  # a diagnostic must not fail on a bad file
        log.info("Could not measure %s: %s", path.name, exc)

    # PyAV will open very nearly anything and report a 0x0 stream for it, and "0x0" in the
    # manifest reads as a measurement rather than as a failure to take one.
    return size if size and all(size) else None


def _node_view(kind: str, size, tag: str, ref_image_size: str, output,
               frames: int) -> dict:
    """What the node will make of this file once it has it.

    Mirrors of its own arithmetic, which is the part nobody can see from the outside and
    the part that answers most questions about a result. An image is capped by
    ``ref_image_size``; a clip is fitted to a fixed canvas that ignores it entirely.
    """
    width, height = size or (0, 0)
    view: dict = {"tag": tag, "kind": kind}

    if kind == IMAGE:
        encoded = scaling.node_image_size(width, height, ref_image_size, output)
        view["encoded_size"] = f"{encoded[0]}x{encoded[1]}" if all(encoded) else "unknown"
        view["reference_tokens"] = scaling.latent_tokens(*encoded) if all(encoded) else None
        view["sized_by"] = f"ref_image_size = {ref_image_size}"
    elif kind == VIDEO:
        canvas = scaling.video_canvas(width, height)
        view["encoded_size"] = f"{canvas[0]}x{canvas[1]}" if all(canvas) else "unknown"
        view["reference_tokens_per_latent_frame"] = (
            scaling.latent_tokens(*canvas) if all(canvas) else None)
        view["sized_by"] = ("a fixed ~1344x768 canvas; reference videos never consult "
                            "ref_image_size")
        view["frames_sent"] = frames
    else:
        view["encoded_size"] = "n/a"
        view["sized_by"] = "audio is resampled to the audio VAE's rate"
    return view


def _entry(live, sent, tag: str, exported: Path | None, ref_image_size: str, output,
           frames: int) -> dict:
    size = measure(exported, sent.kind) if exported else None
    original = ((int(live.width), int(live.height))
                if live.width and live.height else None)

    entry = {
        "tag": tag,
        "kind": sent.kind,
        "file_in_bundle": exported.name if exported else None,
        "sent_size": f"{size[0]}x{size[1]}" if size else "unknown",
        "original_size": f"{original[0]}x{original[1]}" if original else "unknown",
        "source_on_disk": live.local_path,
        "prepared_copy": sent.local_path if sent.local_path != live.local_path else None,
        "already_on_server": sent.comfy_name if sent.local_path is None else None,
    }
    if sent.kind in (VIDEO, AUDIO):
        # On the *sent* row: a prepared clip already starts at the mark, so it submits
        # zero, and saying otherwise here would double the cut in the reader's head.
        entry["starts_at"] = sent.trim_start
    if live.kind == IMAGE:
        entry["size_ceiling_percent"] = live.scale_percent
    if live.kind == VIDEO:
        entry["size_ceiling_percent"] = live.scale_percent
        entry["posed"] = bool(live.poses)

    entry["node_will_encode"] = _node_view(sent.kind, size, tag, ref_image_size,
                                           output, frames)
    return entry


def export(state, snapshot_state, directory: Path | None = None,
           notes: list[str] | None = None) -> BundleResult:
    """Write the bundle. ``snapshot_state`` is the submit copy, after every substitution.

    Both states are wanted. The snapshot says what is *sent*; the live state is what the
    tags were computed from, and they have to agree -- a tag is assigned from the order
    and kind of the rows, which the substitutions never change.

    Read from the state rather than from a built graph. The two now agree -- the builder
    writes width, height and length as literals -- but the state is still the source they
    are both derived from, and reading it keeps this independent of which node the
    workflow happens to hold them on.
    """
    from .graph_builder import clean_ref_image_size
    directory = Path(directory or config.BUNDLE_DIR)
    _prepare(directory)

    tags = refs_mod.compute_tags(state.refs)
    frames = mathmirror.frames_from_seconds(
        mathmirror.clamp_duration(state.duration_seconds))
    output = state.resolution
    ref_image_size = clean_ref_image_size(state.ref_image_size)

    result = BundleResult(directory=directory)
    entries = []

    live_rows = state.refs.all_rows()
    sent_rows = snapshot_state.refs.all_rows()
    for index, (row, sent) in enumerate(zip(live_rows, sent_rows), start=1):
        tag = tags.tag_for(row) or "(untagged)"
        exported = None

        if sent.local_path:
            source = Path(sent.local_path)
            if source.is_file():
                exported = directory / f"{_tagged_name(tag, sent, index)}{source.suffix}"
                try:
                    shutil.copy2(source, exported)
                    result.copied.append(exported.name)
                except OSError as exc:
                    result.missing.append(f"{tag}: could not copy {source.name} ({exc})")
                    exported = None
            else:
                result.missing.append(f"{tag}: {source.name} is not on disk")
        elif sent.comfy_name:
            # A server-file row has nothing local to copy; saying so is the useful part.
            result.missing.append(
                f"{tag}: lives on the server as {sent.comfy_name}, nothing to export")

        entries.append(_entry(row, sent, tag, exported, ref_image_size, output, frames))

        # The soundtrack of a posed or rescaled clip travels inside the clip itself, so
        # there is no separate file for it -- but it does carry its own tag.
        soundtrack = tags.soundtrack_tag_for(row)
        if soundtrack:
            entries.append({
                "tag": soundtrack,
                "kind": "audio",
                "file_in_bundle": exported.name if exported else None,
                "note": "the soundtrack of the video above, sent inside the same file",
            })

    (directory / PROMPT_NAME).write_text(state.prompt_text, encoding="utf-8")

    manifest = {
        "written_at": datetime.now().isoformat(timespec="seconds"),
        "generation": {
            "width": output[0],
            "height": output[1],
            "length_frames": frames,
            "seconds": round(mathmirror.true_seconds(frames), 3),
            "ref_image_size": ref_image_size,
        },
        "prompt_file": PROMPT_NAME,
        "prompt_sections": prompt_mod.normalise(state.prompt_sections),
        "references": entries,
        "not_exported": result.missing,
    }
    if notes:
        manifest["notes"] = list(notes)

    (directory / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    (directory / README_NAME).write_text(_readme(), encoding="utf-8")

    log.info("Wrote a reference bundle to %s", directory)
    return result


def _readme() -> str:
    return (
        "What MiniMaxH3ReferenceToVideo is given\n"
        "======================================\n\n"
        "These are the actual files, after everything this app does to a reference:\n"
        "a section cut from a clip, a skeleton drawn from it, an image rescaled, an\n"
        "EXIF rotation applied. They are named for the prompt tag the model addresses\n"
        "each one by, so <Picture 1> is the file whose name says Picture_1.\n\n"
        f"{PROMPT_NAME} is the prompt exactly as the node receives it.\n\n"
        f"{MANIFEST_NAME} adds what happens next, on the far side of the upload:\n"
        "'sent_size' is the file here, and 'node_will_encode' is what the node resizes\n"
        "it to before the VAE sees it. Those two differ, and the second is the one that\n"
        "decides what the model actually looks at.\n\n"
        "Nothing here is read back by the app. Delete the folder whenever you like.\n"
    )
