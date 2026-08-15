"""Format chips land as lines, with the hole already selected.

The single-line behaviour is pinned next door in test_prompt_insert.py; what is new here is
everything that follows from a fragment spanning lines, from a fragment carrying a
placeholder, and from a chip that knows which section it belongs to.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from harmon3 import config, snippets  # noqa: E402  (after the importorskip)


@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def editor(qapp):
    from harmon3.ui.prompt_editor import PromptEditor
    widget = PromptEditor()
    yield widget
    widget.deleteLater()


BODY = "detailed_description"
DEFINITIONS = "subject_definitions"


def _at(editor, name, text, position=None):
    """Put `text` in a section, place the caret, and return the section."""
    section = editor.sections[name]
    section.set_text(text)
    editor._on_section_focused(section)
    cursor = section.editor.textCursor()
    cursor.setPosition(len(text) if position is None else position)
    section.editor.setTextCursor(cursor)
    return section


# ------------------------------------------------------------------------ block inserts

def test_a_block_into_an_empty_section_has_no_leading_blank_line(editor):
    section = _at(editor, BODY, "")
    editor.insert_snippet("first line\nsecond line", section=section)
    assert section.text() == "first line\nsecond line"


def test_a_one_line_block_is_still_separated_by_a_newline(editor):
    """A shot line has no newline in it and is still a line. Hence the explicit flag."""
    section = _at(editor, BODY, "[Shot 1] a woman waits.")
    editor.insert_snippet("[Shot 2] At 00:00.000, ", section=section, block=True)
    assert section.text() == "[Shot 1] a woman waits.\n[Shot 2] At 00:00.000, "


def test_a_block_does_not_double_a_newline_that_is_already_there(editor):
    section = _at(editor, BODY, "[Shot 1] a woman waits.\n")
    editor.insert_snippet("[Shot 2] At 00:00.000, ", section=section, block=True)
    assert section.text() == "[Shot 1] a woman waits.\n[Shot 2] At 00:00.000, "


def test_a_block_in_the_middle_gets_a_line_of_its_own(editor):
    section = _at(editor, BODY, "before\nafter", position=len("before\n"))
    editor.insert_snippet("one\ntwo", section=section)
    assert section.text() == "before\none\ntwo\nafter"


def test_a_single_line_fragment_still_pads_with_spaces(editor):
    """The existing rule, unchanged: only a multi-line fragment becomes a line."""
    section = _at(editor, BODY, "Use as reference", position=4)
    editor.insert_snippet("<scenetrans>", section=section)
    assert section.text() == "Use <scenetrans> as reference"


# ---------------------------------------------------------------------------- the holes

def test_the_hole_is_selected_so_typing_replaces_it(editor):
    section = _at(editor, BODY, "")
    editor.insert_snippet("<d>[English] ...</d>", snippets.HOLE, section)

    cursor = section.editor.textCursor()
    assert cursor.selectedText() == snippets.HOLE
    cursor.insertText("Watch your dog!")
    assert section.text() == "<d>[English] Watch your dog!</d>"


def test_the_hole_is_found_past_the_padding(editor):
    section = _at(editor, BODY, "She says")
    editor.insert_snippet("<d>[English] ...</d>", snippets.HOLE, section)
    assert section.editor.textCursor().selectedText() == snippets.HOLE
    assert section.text() == "She says <d>[English] ...</d>"


def test_a_fragment_with_no_hole_leaves_the_caret_where_a_tag_would(editor):
    section = _at(editor, BODY, "Use as reference", position=4)
    editor.insert_snippet("<cutoff>", section=section)
    position = section.editor.textCursor().position()
    assert section.text()[:position] == "Use <cutoff>"


def test_a_hole_that_is_not_in_the_text_is_ignored(editor):
    section = _at(editor, BODY, "")
    editor.insert_snippet("no hole here", "...", section)
    assert section.text() == "no hole here"
    assert section.editor.textCursor().hasSelection() is False


# ----------------------------------------------------------------------------- aiming

def test_a_chip_inserts_into_its_own_section_not_the_focused_one(editor):
    _at(editor, "overall_soundscape", "rain on glass")
    body = editor.sections[BODY]
    editor.insert_snippet("[Shot 1] ", section=body)

    assert body.text() == "[Shot 1] "
    assert editor.sections["overall_soundscape"].text() == "rain on glass"


def test_inserting_into_a_folded_section_opens_it(editor):
    section = editor.sections["summary"]
    section.set_expanded(False)
    editor.insert_snippet("[reference generation] ", section=section)
    assert section.is_expanded() is True


def test_one_undo_takes_a_whole_skeleton_back_out(editor):
    section = _at(editor, BODY, "already written")
    editor.insert_snippet(snippets.SKELETONS[BODY], snippets.HOLE, section)
    assert "\n" in section.text()

    section.editor.undo()
    assert section.text() == "already written"


def test_a_snippet_does_not_touch_the_clipboard(editor, qapp):
    from PySide6.QtGui import QGuiApplication

    QGuiApplication.clipboard().setText("SENTINEL")
    editor.insert_snippet(snippets.SKELETONS[BODY], snippets.HOLE,
                          editor.sections[BODY])
    assert QGuiApplication.clipboard().text() == "SENTINEL"


# ------------------------------------------------------------------------- task types

def test_a_task_type_creates_the_prefix(editor):
    editor.apply_task_type("video editing")
    assert editor.sections["summary"].text().startswith("[video editing] ")


def test_a_second_task_type_joins_the_first(editor):
    editor.apply_task_type("video editing")
    editor.apply_task_type("audio reuse")
    assert editor.sections["summary"].text() == "[video editing + audio reuse] "


def test_the_same_task_type_twice_does_not_duplicate(editor):
    editor.apply_task_type("audio reuse")
    before = editor.sections["summary"].text()
    editor.apply_task_type("audio reuse")
    assert editor.sections["summary"].text() == before


def test_a_task_type_keeps_the_rest_of_the_summary(editor):
    section = _at(editor, "summary", "[video editing] The target video is an edit.")
    editor.apply_task_type("audio reuse")
    assert section.text() == "[video editing + audio reuse] The target video is an edit."


def test_a_task_type_does_not_throw_away_the_undo_history(editor):
    """set_text would call setPlainText, which clears the stack -- hence the cursor edit."""
    section = editor.sections["summary"]
    section.editor.setPlainText("")
    section.editor.textCursor().insertText("typed by hand")
    editor.apply_task_type("video editing")

    section.editor.undo()
    section.editor.undo()
    assert section.text() == ""


# ----------------------------------------------------------------------- subject chips

def _labels(row):
    return [chip.snippet.label for chip in row.chips()]


def test_a_defined_subject_becomes_a_chip_in_the_other_sections(editor):
    editor.sections[DEFINITIONS].set_text("<Subject 1> is the woman in <Picture 1>.")

    assert "<Subject 1>" in _labels(editor.sections[BODY].snippets)
    assert "<Subject 1>" in _labels(editor.sections["summary"].snippets)


def test_the_definitions_do_not_offer_their_own_subjects_back(editor):
    """Only the computed `<Subject N>` chip lives there, and it is the *next* number."""
    editor.sections[DEFINITIONS].set_text("<Subject 1> is the woman.")
    labels = _labels(editor.sections[DEFINITIONS].snippets)
    assert labels.count(snippets.NEXT_SUBJECT_LABEL) == 1
    assert "<Subject 1>" not in labels


def test_a_subject_chip_is_dropped_when_its_definition_goes(editor):
    editor.sections[DEFINITIONS].set_text("<Subject 1> and <Subject 2>.")
    assert "<Subject 2>" in _labels(editor.sections[BODY].snippets)

    editor.sections[DEFINITIONS].set_text("<Subject 1> only.")
    assert "<Subject 2>" not in _labels(editor.sections[BODY].snippets)


def test_loading_a_scene_shows_its_subjects_without_a_keystroke(editor):
    editor.set_sections({DEFINITIONS: "<Subject 3> is the dancer."})
    assert "<Subject 3>" in _labels(editor.sections[BODY].snippets)


def test_a_subject_chip_inserts_the_label_it_shows(editor):
    editor.sections[DEFINITIONS].set_text("<Subject 1> is the woman.")
    body = editor.sections[BODY]
    chip = next(c for c in body.snippets.chips() if c.snippet.label == "<Subject 1>")
    body.snippets._on_clicked(chip.snippet)
    assert body.text() == "<Subject 1>"


# ----------------------------------------------------------- the chips and the buttons

def test_the_next_shot_chip_counts_what_is_already_written(editor):
    body = _at(editor, BODY, "[Shot 1] a woman waits.")
    chip = next(c for c in body.snippets.chips()
                if c.snippet.label == snippets.NEXT_SHOT_LABEL)
    body.snippets._on_clicked(chip.snippet)
    assert "[Shot 2] At 00:00.000, " in body.text()
    assert body.editor.textCursor().selectedText() == snippets.SHOT_TIMESTAMP


def test_a_definition_chip_starts_its_own_line(editor):
    """The guide gives each defined item its own line; two on one line define neither."""
    definitions = _at(editor, DEFINITIONS, "<Subject 1> is the woman.")
    chip = next(c for c in definitions.snippets.chips() if c.snippet.label == "video source")
    definitions.snippets._on_clicked(chip.snippet)
    assert definitions.text() == (
        "<Subject 1> is the woman.\n"
        "<Video 1> is the source video for the target video edit.")


def test_a_marker_chip_stays_inside_the_line_it_is_clicked_into(editor):
    """The counterpart: a marker is a word in a line, not a line of its own."""
    retention = _at(editor, "retention_analysis", "<Subject 1> (appears in [Shot 1]): ")
    chip = next(c for c in retention.snippets.chips()
                if c.snippet.label == "fully_preserved")
    retention.snippets._on_clicked(chip.snippet)
    assert retention.text() == "<Subject 1> (appears in [Shot 1]): fully_preserved"


def test_the_next_subject_chip_counts_what_is_already_defined(editor):
    definitions = _at(editor, DEFINITIONS, "<Subject 1> is the woman.\n")
    chip = next(c for c in definitions.snippets.chips()
                if c.snippet.label == snippets.NEXT_SUBJECT_LABEL)
    definitions.snippets._on_clicked(chip.snippet)
    assert definitions.text().endswith("<Subject 2>")


def test_a_task_type_chip_merges_rather_than_inserting(editor):
    summary = editor.sections["summary"]
    chip = next(c for c in summary.snippets.chips()
                if c.snippet.task_type == "video editing")
    summary.snippets._on_clicked(chip.snippet)
    assert summary.text() == "[video editing] "


def test_no_chip_can_take_focus_from_the_editor(editor, qapp):
    """A focused chip would hide the editor's selection, so the hole would look unselected."""
    from PySide6.QtCore import Qt

    for section in editor.sections.values():
        for chip in section.snippets.chips():
            assert chip.focusPolicy() == Qt.NoFocus


def test_helpers_off_hides_every_row(editor):
    editor.set_all_expanded(True)
    editor.set_helpers_visible(False)
    assert editor.helpers_visible() is False
    assert not any(s.snippets.isVisibleTo(s) for s in editor.sections.values())

    editor.set_helpers_visible(True)
    assert all(s.snippets.isVisibleTo(s) for s in editor.sections.values())


def test_a_folded_section_hides_its_chips_even_with_helpers_on(editor):
    section = editor.sections["summary"]
    editor.set_helpers_visible(True)
    section.set_expanded(False)
    assert section.snippets.isVisibleTo(section) is False


def test_the_guide_button_is_hidden_when_the_document_is_absent(monkeypatch, tmp_path,
                                                               qapp):
    from harmon3.ui.prompt_editor import PromptEditor

    monkeypatch.setattr(config, "GUIDE_PATH", tmp_path / "nothing.md")
    widget = PromptEditor()
    try:
        assert widget.guide_button.isVisible() is False
    finally:
        widget.deleteLater()


def test_the_guide_dialog_builds_from_the_document(qapp, tmp_path):
    from harmon3.ui.snippet_bar import GuideDialog

    dialog = GuideDialog("# a heading\n\nsome text", tmp_path / "guide.md")
    try:
        assert dialog.windowTitle()
    finally:
        dialog.deleteLater()
