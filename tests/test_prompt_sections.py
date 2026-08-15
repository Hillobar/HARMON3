"""Joining the prompt's named sections into the one string the model receives."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from harmon3 import prompt                                      # noqa: E402


def test_the_sections_and_their_order():
    assert prompt.SECTIONS == (
        "subject_definitions",
        "summary",
        "retention_analysis",
        "detailed_description",
        "overall_soundscape",
        "non_diegetic_music",
    )


def test_the_combined_shape():
    combined = prompt.combine({"subject_definitions": "a hero", "summary": "a wide shot"})
    assert combined == (
        "subject_definitions:\n"
        "a hero\n"
        "\n"
        "summary:\n"
        "a wide shot\n"
        "\n"
        "retention_analysis:\n"
        "N/A\n"
        "\n"
        "detailed_description:\n"
        "N/A\n"
        "\n"
        "overall_soundscape:\n"
        "N/A\n"
        "\n"
        "non_diegetic_music:\n"
        "N/A"
    )


def test_every_section_appears_even_when_all_are_empty():
    combined = prompt.combine({})
    for name in prompt.SECTIONS:
        assert f"{name}:\nN/A" in combined
    assert combined.count("N/A") == len(prompt.SECTIONS)


def test_sections_keep_their_order_regardless_of_the_dict():
    reversed_input = {name: name for name in reversed(prompt.SECTIONS)}
    positions = [prompt.combine(reversed_input).index(f"{n}:") for n in prompt.SECTIONS]
    assert positions == sorted(positions)


def test_whitespace_only_counts_as_empty():
    assert "summary:\nN/A" in prompt.combine({"summary": "   \n\t "})


def test_surrounding_whitespace_is_trimmed_but_the_body_is_kept():
    combined = prompt.combine({"summary": "  line one\nline two  "})
    assert "summary:\nline one\nline two" in combined


def test_multiline_text_survives():
    combined = prompt.combine({"detailed_description": "CUT 1: a\nCUT 2: b"})
    assert "detailed_description:\nCUT 1: a\nCUT 2: b" in combined


def test_unknown_keys_are_ignored():
    combined = prompt.combine({"summary": "kept", "nonsense": "dropped"})
    assert "kept" in combined and "dropped" not in combined


@pytest.mark.parametrize("junk", [None, "", [], 42, {"summary": None}, {"summary": 7}])
def test_junk_input_still_produces_the_full_structure(junk):
    combined = prompt.combine(junk)
    assert all(f"{name}:" in combined for name in prompt.SECTIONS)


def test_empty_sections_helper():
    assert prompt.empty_sections() == {name: "" for name in prompt.SECTIONS}
    assert prompt.is_empty(prompt.empty_sections())
    assert not prompt.is_empty({"summary": "x"})


def test_filled_names_reports_what_was_written():
    assert prompt.filled_names({"summary": "x", "non_diegetic_music": "  "}) == ["summary"]


# ---------------------------------------------------------------------------------
# Round tripping
# ---------------------------------------------------------------------------------

def test_a_combined_prompt_parses_back_into_its_sections():
    original = {name: f"text for {name}" for name in prompt.SECTIONS}
    assert prompt.parse(prompt.combine(original)) == original


def test_parsing_restores_empty_boxes_as_empty():
    sections = {"summary": "only this"}
    restored = prompt.parse(prompt.combine(sections))
    assert restored["summary"] == "only this"
    assert restored["detailed_description"] == ""      # not the literal "N/A"


def test_parsing_keeps_multiline_bodies_together():
    restored = prompt.parse(prompt.combine({"detailed_description": "CUT 1: a\n\nCUT 2: b"}))
    assert restored["detailed_description"] == "CUT 1: a\n\nCUT 2: b"


def test_text_without_headings_is_treated_as_a_prompt_from_before_the_split():
    restored = prompt.parse("just a plain old prompt")
    assert restored[prompt.LEGACY_SECTION] == "just a plain old prompt"
    assert prompt.filled_names(restored) == [prompt.LEGACY_SECTION]


def test_a_heading_like_line_inside_a_body_does_not_start_a_section():
    """Only an exact heading on its own line counts."""
    restored = prompt.parse(prompt.combine({"summary": "see summary: below"}))
    assert restored["summary"] == "see summary: below"


def test_from_legacy_puts_everything_in_one_box():
    sections = prompt.from_legacy("an old prompt")
    assert sections[prompt.LEGACY_SECTION] == "an old prompt"
    assert prompt.filled_names(sections) == [prompt.LEGACY_SECTION]


def test_from_legacy_tolerates_nothing():
    assert prompt.is_empty(prompt.from_legacy(""))
