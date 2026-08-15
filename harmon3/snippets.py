"""The output format, as fragments that can be clicked into place.

The six sections are written to a spec nobody memorises: four label kinds, a fixed
vocabulary of bracketed task types and relationship markers, and shot lines whose
timestamps have a shape. Retyping any of that is where the errors come from -- a prompt
saved in this repo says ``<Subjkect 1>``, which the model reads as a subject it has never
been told about, so the shot quietly loses the person it was built around. There is no
error message for that; the video simply comes back wrong.

So the vocabulary is data here, and the editor turns it into chips whose captions *are*
the tokens they insert. That is what lets one row do two jobs: the row under
``retention_analysis`` is the way to insert a marker and, at the same time, the list of
which seven markers exist.

The guide in ./API is the authority. When it changes, this table changes and nothing else
does -- and ``tests/test_snippets.py`` checks the fixed vocabularies still appear in it.

No Qt, no network.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from . import config, prompt as prompt_mod

#: What a chip inserts where the user has to write something themselves. Inserted *and*
#: selected, so the next keystroke replaces it -- a placeholder the user has to go and
#: find is worse than no placeholder at all.
HOLE = "..."


@dataclass(frozen=True)
class Snippet:
    """One clickable fragment of the format."""

    #: What the chip says. For a bare token this is the token itself, so the row reads as
    #: a reference card as well as a set of buttons.
    label: str
    #: What lands in the editor. A newline anywhere in it makes this a block insert: the
    #: fragment is a *line* of the format, not a phrase inside one.
    text: str
    #: Selected after insertion, if it occurs in ``text``.
    select: str | None = None
    #: Hover text, usually the guide's own reason for the token existing.
    hint: str = ""
    #: Set on the six summary chips, which merge into the bracketed prefix rather than
    #: inserting at the caret. See ``merge_task_type``.
    task_type: str = ""
    #: Whether this is a *line* of the format rather than a phrase inside one, and so is
    #: separated by newlines instead of spaces. Not inferable from the text: a shot line
    #: is one line long and still must not be glued onto the end of the previous shot.
    block: bool = False

    @property
    def is_block(self) -> bool:
        return self.block or "\n" in self.text


#: The bracketed prefixes ``summary`` may carry, in the guide's own order.
TASK_TYPES = (
    "keyframe completion",
    "reference generation",
    "video editing",
    "video continuation",
    "audio reuse",
    "audio reference",
)

_TASK_TYPE_HINTS = {
    "keyframe completion": "An image is a concrete frame of the target video - first "
                           "frame, keyframe, last frame.",
    "reference generation": "An asset guides a character, scene, style, action or camera "
                            "move without being a concrete frame or the video being edited.",
    "video editing": "An existing source video is directly modified.",
    "video continuation": "New content continues or extends an existing source video.",
    "audio reuse": "The same audio signal is reused, in full or in part.",
    "audio reference": "Only the style, timbre, words, texture or beat of the audio is "
                       "referenced, not the signal.",
}

#: Relationship markers for visible content: <Subject N>, <Picture N>, <Video N>.
VISIBLE_MARKERS = (
    "fully_preserved",
    "partially_preserved",
    "attribute_transfer",
    "weak_reference",
)

#: Relationship markers for <Audio N>. ``weak_reference`` is in both lists, with the same
#: meaning; the other six are not interchangeable, which is why the row shows them apart.
AUDIO_MARKERS = (
    "fully_copy",
    "partially_copy",
    "reference",
    "weak_reference",
)

_MARKER_HINTS = {
    "fully_preserved": "The referenced content's defined role is fully kept.",
    "partially_preserved": "Still used, but some defined characteristics change.",
    "attribute_transfer": "Characteristics move to a different identifiable subject.",
    "weak_reference": "Only broad similarity in style, category or atmosphere.",
    "fully_copy": "The whole source audio is the target's whole final track.",
    "partially_copy": "Part of the timeline or some layers are copied.",
    "reference": "Timbre, rhythm, style or content is referenced, not copied.",
}

#: The multi-line starting point behind each section's `skeleton` chip. Written as lines of
#: the real format with holes in them, so filling one in is editing rather than composing.
SKELETONS = {
    "subject_definitions": (
        f"<Subject 1> is the {HOLE} in <Picture 1>, with {HOLE}.\n"
        f"<Video 1> is the source video for the target video edit.\n"
        f"<Audio 1> is the voice-timbre reference for <Subject 1> (S1)."
    ),
    "summary": f"[reference generation] The target video shows <Subject 1> {HOLE}.",
    "retention_analysis": (
        f"<Subject 1> (appears in [Shot 1]): fully_preserved - {HOLE}\n"
        f"<Audio 1>: reference - {HOLE}"
    ),
    "detailed_description": (
        f"The target video is in a {HOLE} style with {HOLE} lighting.\n"
        f"[Shot 1] {HOLE}\n"
        f"[Shot 2] At 00:00.000, the shot cuts to {HOLE}"
    ),
    "overall_soundscape": (
        f"Quiet {HOLE} room tone and a low {HOLE} hum continue throughout the video."
    ),
    "non_diegetic_music": (
        f"A restrained {HOLE} score at a {HOLE} tempo, with {HOLE} underneath and no swell."
    ),
}


def _skeleton(name: str) -> Snippet:
    return Snippet("skeleton", SKELETONS[name], HOLE,
                   "A starting point for this whole section, in the right shape.",
                   block=True)


def _task_type(value: str) -> Snippet:
    return Snippet(value, "", None, _TASK_TYPE_HINTS[value], task_type=value)


def _marker(value: str) -> Snippet:
    return Snippet(value, value, None, _MARKER_HINTS[value])


#: section name -> what its chip row contains, in order. A bare ``str`` is a caption
#: rather than a chip, so a row can group its markers without splitting the table in two.
CATALOGUE: dict[str, tuple] = {
    "subject_definitions": (
        _skeleton("subject_definitions"),
        Snippet("<Subject N>", "", None,
                "The next unused subject number, computed from what is already defined."),
        Snippet("subject",f"<Subject 1> is the {HOLE} in <Picture 1>, with {HOLE}.", HOLE,
                "Reusable visible content: a person, a place, a garment, a style, a move.",
                block=True),
        Snippet("two sources",
                f"<Subject 1> is the {HOLE} whose appearance comes from <Picture 1> and "
                f"whose {HOLE} comes from <Video 1>.", HOLE,
                "One subject may be defined by several assets; say what each provides.",
                block=True),
        Snippet("picture as frame",
                f"<Picture 1> is the first frame of [Shot 1], showing {HOLE}.", HOLE,
                "Only when the image IS a frame. An image that just defines a character "
                "is cited inside that subject's line instead.", block=True),
        Snippet("storyboard",
                "<Picture 1> is a storyboard reference for [Shot 1] and [Shot 2], "
                "defining their viewpoint, subject placement, and shot order.", None,
                "For an image that plans shots rather than appearing in one.", block=True),
        Snippet("video source", "<Video 1> is the source video for the target video edit.",
                None,
                "<Video N> is for whole-video relationships - editing, continuing, or "
                "following its structure. Anything visible reused from it is a subject.",
                block=True),
        Snippet("voice ref", "<Audio 1> is the voice-timbre reference for <Subject 1> (S1).",
                None,
                "When an audio maps to a speaker, reuse that speaker's ID here; never "
                "assign a new one in a definition.", block=True),
    ),
    "summary": (
        _skeleton("summary"),
        "task type:",
        *(_task_type(value) for value in TASK_TYPES),
        Snippet("edit opener", "The target video is an edited version of <Video 1>.", None,
                "How a video-editing summary begins, after the prefix."),
    ),
    "retention_analysis": (
        _skeleton("retention_analysis"),
        Snippet("subject line",
                f"<Subject 1> (appears in [Shot 1], [Shot 3]): fully_preserved - {HOLE}",
                HOLE, "One line per label, saying where it appears and how it survives.",
                block=True),
        Snippet("picture line",
                f"<Picture 1> ([Shot 1] first frame): fully_preserved - {HOLE}", HOLE,
                block=True),
        Snippet("video line",
                f"<Video 1> (cut and pacing structure): weak_reference - {HOLE}", HOLE,
                block=True),
        Snippet("audio line", f"<Audio 1>: fully_copy - {HOLE}", HOLE, block=True),
        "visible:",
        *(_marker(value) for value in VISIBLE_MARKERS),
        "audio:",
        *(_marker(value) for value in AUDIO_MARKERS if value not in VISIBLE_MARKERS),
    ),
    "detailed_description": (
        _skeleton("detailed_description"),
        Snippet("style opening",
                f"The target video is in a {HOLE} style with {HOLE} lighting and a {HOLE} "
                "color palette.", HOLE,
                "One or two sentences of style, before [Shot 1] rather than inside it.",
                block=True),
        Snippet("[Shot 1]", "[Shot 1] ", None, "The opening shot carries no timestamp.",
                block=True),
        Snippet("[Shot N] At ...", "", None,
                "The next shot, numbered from what you have already written. Later shots "
                "mark their cut time as MM:SS.mmm.", block=True),
        Snippet("(S1)", "(S1)", "1",
                "A speaker ID, numbered by the order voices are first heard. Reuse the "
                "same ID everywhere that voice speaks."),
        Snippet("<d>", f"<d>[English] {HOLE}</d>", HOLE,
                "Dialogue and lyrics, in their original language."),
        Snippet("speaks", f"<Subject 1> (S1) says, <d>[English] {HOLE}</d>", HOLE,
                "A subject who speaks keeps both labels: the subject, and the speaker."),
        Snippet("frame anchor", "the shot begins from <Picture 1>", None,
                "Natural phrasing for a concrete frame anchor."),
        Snippet("<scenetrans>", "<scenetrans>", None,
                "Dialogue or sound carrying across a cut."),
        Snippet("<cutoff>", "<cutoff>", None,
                "Speech truncated by the video ending."),
        Snippet("[unclear]", "[unclear]", None,
                "For an unintelligible span of reused dialogue. Never guess at it."),
    ),
    "overall_soundscape": (
        _skeleton("overall_soundscape"),
        Snippet("ambience",
                f"Quiet {HOLE} room tone and a low {HOLE} hum continue throughout the "
                "video.", HOLE,
                "Ambience and physical sound across the whole video. Anything tied to one "
                "shot stays in detailed_description."),
        Snippet("copied ambience",
                "The copied ambience layer from <Audio 1> continues throughout the target "
                "video.", None,
                "State a copy or reference relationship in whichever section carries that "
                "audible layer."),
        Snippet(prompt_mod.EMPTY, prompt_mod.EMPTY, None, "When there is none."),
    ),
    "non_diegetic_music": (
        _skeleton("non_diegetic_music"),
        Snippet("score",
                f"A restrained {HOLE} score at a {HOLE} tempo, with {HOLE} underneath and "
                "no swell.", HOLE,
                "Music only the audience hears. Give instrumentation, tempo and whether it "
                "develops."),
        Snippet("reused score",
                "<Audio 2> is directly reused as the complete audience-only score.", None),
        Snippet(prompt_mod.EMPTY, prompt_mod.EMPTY, None, "When there is none."),
    ),
}

#: The two chips whose text is computed from the section rather than fixed, matched by
#: label when a click comes in. Both would otherwise hand out a number that already exists.
NEXT_SHOT_LABEL = "[Shot N] At ..."
NEXT_SUBJECT_LABEL = "<Subject N>"

#: The timestamp a new shot line starts with, and the part of it worth selecting.
SHOT_TIMESTAMP = "00:00.000"

_TASK_PREFIX_RE = re.compile(r"^\s*\[([^\]\n]*)\]")
_SHOT_RE = re.compile(r"\[\s*Shot\s+(\d+)\s*\]", re.IGNORECASE)


def next_shot_number(text: str) -> int:
    """One past the highest ``[Shot N]`` already written, or 1.

    Highest rather than count: shots are sometimes drafted out of order, and offering
    ``[Shot 2]`` when a ``[Shot 5]`` already exists would produce a duplicate.
    """
    numbers = [int(match.group(1)) for match in _SHOT_RE.finditer(text or "")]
    return max(numbers) + 1 if numbers else 1


def next_shot_snippet(text: str) -> Snippet:
    """The `[Shot N] At ...` chip, numbered for what is already in the section."""
    number = next_shot_number(text)
    return Snippet(
        NEXT_SHOT_LABEL,
        f"[Shot {number}] At {SHOT_TIMESTAMP}, ",
        SHOT_TIMESTAMP,
        "The next shot, with its cut time selected. Later shots mark MM:SS.mmm.",
        block=True,
    )


def next_subject_number(definitions: str) -> int:
    """One past the highest ``<Subject N>`` defined, or 1."""
    from . import refs as refs_mod

    numbers = [int(m.group(2)) for m in refs_mod.SUBJECT_RE.finditer(definitions or "")]
    return max(numbers) + 1 if numbers else 1


def merge_task_type(text: str, task_type: str) -> tuple[str, int, int] | None:
    """The summary's ``[...]`` prefix with one more type in it, and the span it replaces.

    ``None`` when the prefix already carries this type. The guide's rule -- combine with
    " + " and never repeat -- cannot be expressed by pasting a token beside the caret, so
    these chips edit the prefix instead of inserting into it.

    A span rather than a whole new string, because the caller rewrites it through a cursor:
    replacing the section's text outright would call ``setPlainText`` and take the user's
    entire undo history with it.
    """
    text = text or ""
    match = _TASK_PREFIX_RE.match(text)
    if match is None:
        return f"[{task_type}] ", 0, 0

    existing = [part.strip() for part in match.group(1).split("+")]
    existing = [part for part in existing if part]
    if task_type in existing:
        return None
    return f"[{' + '.join(existing + [task_type])}]", match.start(), match.end()


def load_guide() -> str | None:
    """The format spec, or ``None`` if it is not there.

    ``errors="replace"`` because the guide carries arrows and en-dashes, and a machine with
    an odd default encoding must not be able to turn a help button into a traceback.
    """
    try:
        return config.GUIDE_PATH.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return None


def next_subject_snippet(definitions: str) -> Snippet:
    """The `<Subject N>` chip, numbered for what is already defined."""
    return Snippet(
        NEXT_SUBJECT_LABEL,
        f"<Subject {next_subject_number(definitions)}>",
        None,
        "The next unused subject number. Define it here, then click it from the other "
        "sections rather than typing it again.",
    )


def rows_for(section: str) -> tuple:
    """What to put under a section's header. Unknown sections simply get nothing."""
    return CATALOGUE.get(section, ())
