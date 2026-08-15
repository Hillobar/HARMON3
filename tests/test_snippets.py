"""The format table, checked against the guide it was copied from.

This feature exists because labels get mistyped, so the one thing it must never do is ship
a mistyped label of its own. Most of what is below is that check, from several angles.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmon3 import config, prompt, refs, snippets


def _snippets():
    for name, row in snippets.CATALOGUE.items():
        for entry in row:
            if not isinstance(entry, str):
                yield name, entry


# --------------------------------------------------------------------------------- shape

def test_the_catalogue_covers_every_section_in_order():
    assert tuple(snippets.CATALOGUE) == prompt.SECTIONS


def test_every_placeholder_is_in_the_text_it_belongs_to():
    """A `select` that does not occur is a hole the caret silently fails to land on."""
    for name, snippet in _snippets():
        if snippet.select is not None:
            assert snippet.select in snippet.text, f"{name}/{snippet.label}"


def test_no_section_offers_the_same_label_twice():
    for name, row in snippets.CATALOGUE.items():
        labels = [e.label for e in row if not isinstance(e, str)]
        assert len(labels) == len(set(labels)), name


def test_only_the_computed_chips_ship_without_text():
    """Everything else inserts itself literally; an empty text would insert nothing."""
    computed = {snippets.NEXT_SHOT_LABEL, snippets.NEXT_SUBJECT_LABEL}
    for name, snippet in _snippets():
        if not snippet.text and not snippet.task_type:
            assert snippet.label in computed, f"{name}/{snippet.label}"


# ---------------------------------------------------------------------------- the labels

#: Anything shaped like one of the four labels, however misspelt.
_LOOSE_LABEL = re.compile(r"<\s*([A-Za-z]+)\s+(\d+)\s*>")


def test_no_snippet_contains_a_misspelt_label():
    """The point of the whole feature. <Subjkect 1> is in a prompt saved in this repo."""
    for name, snippet in _snippets():
        for match in _LOOSE_LABEL.finditer(snippet.text):
            assert (refs.TAG_RE.fullmatch(match.group(0))
                    or refs.SUBJECT_RE.fullmatch(match.group(0))), \
                f"{name}/{snippet.label}: {match.group(0)}"


def test_no_skeleton_contains_a_misspelt_label():
    for name, skeleton in snippets.SKELETONS.items():
        for match in _LOOSE_LABEL.finditer(skeleton):
            assert (refs.TAG_RE.fullmatch(match.group(0))
                    or refs.SUBJECT_RE.fullmatch(match.group(0))), \
                f"{name}: {match.group(0)}"


def test_every_label_is_written_canonically():
    """`<Subject  1>` reads to the model as a different label from `<Subject 1>`."""
    for name, snippet in _snippets():
        for match in _LOOSE_LABEL.finditer(snippet.text):
            kind, number = match.group(1), match.group(2)
            assert match.group(0) == f"<{kind} {int(number)}>", f"{name}/{snippet.label}"


def test_a_skeleton_cannot_be_mistaken_for_a_section_heading():
    """`parse` splits on a line that is a section name and a colon; a skeleton must not be."""
    for name, skeleton in snippets.SKELETONS.items():
        combined = prompt.combine({name: skeleton})
        assert prompt.parse(combined)[name] == skeleton


# ------------------------------------------------------------------- against the guide

def _guide():
    text = snippets.load_guide()
    if text is None:
        pytest.skip("the guide is not present in this checkout")
    return text


@pytest.mark.parametrize("value", snippets.TASK_TYPES)
def test_every_task_type_appears_in_the_guide(value):
    """Turns the guide drifting out from under this table into a failing test."""
    assert value in _guide()


@pytest.mark.parametrize(
    "value", sorted(set(snippets.VISIBLE_MARKERS + snippets.AUDIO_MARKERS)))
def test_every_relationship_marker_appears_in_the_guide(value):
    assert value in _guide()


def test_the_guide_is_optional(monkeypatch, tmp_path):
    monkeypatch.setattr(config, "GUIDE_PATH", tmp_path / "nothing.md")
    assert snippets.load_guide() is None


# ------------------------------------------------------------------------- task prefixes

def test_a_summary_with_no_prefix_gets_one():
    assert snippets.merge_task_type("", "video editing") == ("[video editing] ", 0, 0)


def test_a_second_task_type_joins_the_prefix():
    replacement, start, end = snippets.merge_task_type(
        "[video editing] The target video", "audio reuse")
    assert replacement == "[video editing + audio reuse]"
    assert (start, end) == (0, len("[video editing]"))


def test_a_task_type_already_present_changes_nothing():
    assert snippets.merge_task_type("[audio reuse] x", "audio reuse") is None


def test_a_loosely_written_prefix_is_tidied_rather_than_doubled():
    replacement, _start, _end = snippets.merge_task_type(
        "[ video editing ]  x", "audio reuse")
    assert replacement == "[video editing + audio reuse]"


def test_prose_that_merely_contains_brackets_is_not_treated_as_a_prefix():
    """Only a leading bracket is the prefix; one mid-sentence is part of the summary."""
    replacement, start, end = snippets.merge_task_type(
        "The target video [really] shows", "video editing")
    assert (replacement, start, end) == ("[video editing] ", 0, 0)


# ------------------------------------------------------------------------- the numbering

@pytest.mark.parametrize("text, expected", [
    ("", 1),
    ("[Shot 1] a", 2),
    ("[Shot 1] a [Shot 2] b", 3),
    ("[Shot 3] a [Shot 1] b", 4),          # drafted out of order; must not repeat 2
    ("[Shot 12] a", 13),
    ("[ Shot 4 ] a", 5),
    ("Shot 9 without brackets", 1),
])
def test_the_next_shot_number(text, expected):
    assert snippets.next_shot_number(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("", 1),
    ("<Subject 1> is the woman", 2),
    ("<Subject 1> and <Subject 3>", 4),    # gaps are left alone rather than filled
    ("<subject 2> lowercase", 3),
])
def test_the_next_subject_number(text, expected):
    assert snippets.next_subject_number(text) == expected


def test_the_computed_chips_carry_their_number():
    assert snippets.next_shot_snippet("[Shot 4] x").text.startswith("[Shot 5] At ")
    assert snippets.next_subject_snippet("<Subject 2>").text == "<Subject 3>"


def test_the_new_shot_line_selects_its_timestamp():
    snippet = snippets.next_shot_snippet("")
    assert snippet.select == snippets.SHOT_TIMESTAMP
    assert snippet.select in snippet.text


# ---------------------------------------------------------------------------- subjects

@pytest.mark.parametrize("text", [
    "<Subject 1>", "<subject 1>", "<Subject  1>", "< Subject 1 >", "<SUBJECT 01>",
])
def test_a_subject_is_recognised_however_it_is_written(text):
    assert refs.subjects_in_prompt(text) == ["<Subject 1>"]


def test_subjects_come_back_in_order_without_repeats():
    text = "<Subject 2> meets <Subject 1>, then <Subject 2> leaves"
    assert refs.subjects_in_prompt(text) == ["<Subject 2>", "<Subject 1>"]


def test_subjects_stay_out_of_the_reference_tags():
    """Folding them into TAG_RE would make the lint flag every subject, forever."""
    assert refs.tags_in_prompt("<Subject 1> in <Picture 1>") == ["<Picture 1>"]
    assert refs.unknown_tags("<Subject 1>", refs.TagAssignment()) == []


def test_defined_subjects_reads_only_what_it_is_given():
    assert refs.defined_subjects("<Subject 1> is the woman in <Picture 1>.") \
        == ["<Subject 1>"]
    assert refs.defined_subjects("") == []
