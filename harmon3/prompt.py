"""The prompt as a set of named sections.

The model receives one string. The editor splits that into labelled boxes so each part of
a shot can be written and revised on its own, and they are joined back together in a fixed
order on the way out:

    subject_definitions:
    whatever was typed

    summary:
    whatever was typed

An empty box still appears, carrying ``N/A``, so the model always sees the same structure.

No Qt, no network.
"""

from __future__ import annotations

#: The sections, in the order they are written to the prompt.
SECTIONS = (
    "subject_definitions",
    "summary",
    "retention_analysis",
    "detailed_description",
    "overall_soundscape",
    "non_diegetic_music",
)

#: What an empty section contributes.
EMPTY = "N/A"

#: Where a prompt written before the split ends up.
LEGACY_SECTION = "detailed_description"


def empty_sections() -> dict:
    return {name: "" for name in SECTIONS}


def normalise(sections) -> dict:
    """A full set of sections, whatever subset or junk was handed in."""
    source = sections if isinstance(sections, dict) else {}
    result = {}
    for name in SECTIONS:
        value = source.get(name, "")
        result[name] = value if isinstance(value, str) else str(value or "")
    return result


def combine(sections) -> str:
    """Join the sections into the single prompt the model receives."""
    filled = normalise(sections)
    return "\n\n".join(
        f"{name}:\n{filled[name].strip() or EMPTY}" for name in SECTIONS
    )


def parse(text: str) -> dict:
    """Split a combined prompt back into its sections.

    Used when reloading something that only kept the joined string, such as an older run
    record. Anything that does not carry the section headings is treated as a prompt from
    before the split.
    """
    lines = (text or "").splitlines()
    headings = {f"{name}:": name for name in SECTIONS}

    sections, current, buffer = empty_sections(), None, []
    found = False
    for line in lines:
        name = headings.get(line.strip())
        if name is not None:
            if current is not None:
                sections[current] = "\n".join(buffer).strip()
            current, buffer, found = name, [], True
            continue
        if current is not None:
            buffer.append(line)
    if current is not None:
        sections[current] = "\n".join(buffer).strip()

    if not found:
        return from_legacy(text)

    # An empty box was written out as N/A; it should come back empty.
    return {name: "" if value.strip() == EMPTY else value
            for name, value in sections.items()}


def from_legacy(text: str) -> dict:
    """Carry a prompt written before the split into the section it best matches."""
    sections = empty_sections()
    sections[LEGACY_SECTION] = text or ""
    return sections


def is_empty(sections) -> bool:
    return not any(value.strip() for value in normalise(sections).values())


def filled_names(sections) -> list:
    return [name for name, value in normalise(sections).items() if value.strip()]


def title(name: str) -> str:
    """The heading shown in the editor. The key itself is what the model sees."""
    return name
