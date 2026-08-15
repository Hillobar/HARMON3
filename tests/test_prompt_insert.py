"""Tag chips insert into the prompt at the cursor.

The padding rules matter more than they look: clicking into the middle of existing text
is the normal case, and a bare insert would routinely produce "Use<Picture 1>as".
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QtWidgets = pytest.importorskip("PySide6.QtWidgets")


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


#: Tags go into whichever section last had the caret; these tests use the first one.
SECTION = "subject_definitions"


def _box(editor):
    return editor.sections[SECTION]


def _insert(editor, text, cursor_at, tag="<Picture 1>", select_to=None):
    """Put `text` in one section, place the caret, insert a tag, return that section."""
    section = _box(editor)
    section.set_text(text)
    editor._on_section_focused(section)

    cursor = section.editor.textCursor()
    cursor.setPosition(cursor_at)
    if select_to is not None:
        cursor.setPosition(select_to, cursor.MoveMode.KeepAnchor)
    section.editor.setTextCursor(cursor)

    editor.insert_tag(tag)
    return section.text()


def test_insert_into_empty_prompt(editor):
    assert _insert(editor, "", 0) == "<Picture 1>"


def test_insert_mid_text_pads_both_sides(editor):
    assert _insert(editor, "Use as reference", 4) == "Use <Picture 1> as reference"


def test_insert_at_end_pads_only_the_left(editor):
    assert _insert(editor, "Show me", 7) == "Show me <Picture 1>"


def test_insert_at_start_pads_only_the_right(editor):
    assert _insert(editor, "is the hero", 0) == "<Picture 1> is the hero"


def test_existing_spaces_are_not_doubled(editor):
    assert _insert(editor, "Show me  now", 8) == "Show me <Picture 1> now"


def test_insert_before_punctuation_still_pads(editor):
    # A comma is not whitespace, so it gets separated rather than glued.
    assert _insert(editor, "Use, then go", 3) == "Use <Picture 1> , then go"


def test_insert_replaces_a_selection(editor):
    assert _insert(editor, "use THIS here", 4, select_to=8) == "use <Picture 1> here"


def test_caret_lands_after_the_tag_not_after_the_padding(editor):
    text = _insert(editor, "Use as reference", 4)
    position = _box(editor).editor.textCursor().position()
    assert text[:position] == "Use <Picture 1>"


def test_caret_lands_at_the_end_when_nothing_follows(editor):
    text = _insert(editor, "Show me", 7)
    assert _box(editor).editor.textCursor().position() == len(text)


def test_insert_across_a_newline_boundary(editor):
    assert _insert(editor, "line one\nline two", 8) == "line one <Picture 1>\nline two"


def test_repeated_inserts_stay_separated(editor):
    """Clicking several chips in a row should not run the tags together."""
    _box(editor).set_text("")
    editor._on_section_focused(_box(editor))
    editor.insert_tag("<Picture 1>")
    editor.insert_tag("<Audio 1>")
    assert _box(editor).text() == "<Picture 1> <Audio 1>"


def test_a_tag_goes_into_the_section_that_last_had_the_caret(editor):
    other = editor.sections["overall_soundscape"]
    other.set_text("rain on glass")
    editor._on_section_focused(other)

    cursor = other.editor.textCursor()
    cursor.setPosition(len("rain on glass"))
    other.editor.setTextCursor(cursor)
    editor.insert_tag("<Audio 1>")

    assert other.text() == "rain on glass <Audio 1>"
    assert _box(editor).text() == ""


def test_inserting_into_a_folded_section_opens_it(editor):
    section = editor.sections["summary"]
    section.set_expanded(False)
    editor._on_section_focused(section)
    editor.insert_tag("<Picture 1>")
    assert section.is_expanded() is True


def test_insert_does_not_touch_the_clipboard(editor, qapp):
    from PySide6.QtGui import QGuiApplication

    QGuiApplication.clipboard().setText("SENTINEL")
    _insert(editor, "Use as reference", 4)
    assert QGuiApplication.clipboard().text() == "SENTINEL"


def test_the_combined_prompt_carries_every_section(editor):
    editor.sections["summary"].set_text("a wide shot")
    editor.sections["overall_soundscape"].set_text("rain")
    combined = editor.text()

    assert "summary:\na wide shot" in combined
    assert "overall_soundscape:\nrain" in combined
    assert "subject_definitions:\nN/A" in combined


def test_chip_click_emits_its_tag(qapp):
    from harmon3.ui.ref_panel import TagChip

    chip = TagChip()
    chip.setText("<Audio 2>")
    seen = []
    chip.clicked.connect(seen.append)

    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtGui import QMouseEvent

    pos = QPoint(4, 4)
    qapp.sendEvent(chip, QMouseEvent(
        QEvent.MouseButtonPress, pos, chip.mapToGlobal(pos),
        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))

    assert seen == ["<Audio 2>"]
    chip.deleteLater()


def test_empty_chip_emits_nothing(qapp):
    from harmon3.ui.ref_panel import TagChip

    chip = TagChip()          # no tag assigned yet, e.g. a hidden soundtrack chip
    seen = []
    chip.clicked.connect(seen.append)

    from PySide6.QtCore import QEvent, QPoint, Qt
    from PySide6.QtGui import QMouseEvent

    pos = QPoint(4, 4)
    qapp.sendEvent(chip, QMouseEvent(
        QEvent.MouseButtonPress, pos, chip.mapToGlobal(pos),
        Qt.LeftButton, Qt.LeftButton, Qt.NoModifier))

    assert seen == []
    chip.deleteLater()
