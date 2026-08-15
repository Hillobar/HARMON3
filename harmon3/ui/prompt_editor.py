"""The prompt, authored as named collapsible sections.

The model receives one string, but a shot is easier to write and revise in parts. Each
section is a box that folds away to a single header line, so having six of them costs
almost no height when they are closed.

Tag handling spans the whole prompt: the lint looks at every section, a rewrite touches
every section, and an inserted tag goes wherever the caret last was.

Under each header sits a row of format chips (see ``snippet_bar``). They exist because the
six sections are written to a spec nobody memorises, and because retyping a label is how
``<Subjkect 1>`` got into a saved prompt in this repo.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import (
    config,
    prompt as prompt_mod,
    refs as refs_mod,
    snippets as snippets_mod,
)
from ..refs import TagAssignment
from . import style
from .snippet_bar import GuideDialog, SnippetRow

#: Height an open box gets before it starts sharing whatever is left over.
MIN_BODY_HEIGHT = 64

PLACEHOLDERS = {
    "subject_definitions": "Who and what is in the shot. Address references by tag, "
                           "e.g. <Picture 1>.",
    "summary": "The shot in a sentence or two.",
    "retention_analysis": "What holds attention, and when.",
    "detailed_description": "The shot in full: staging, camera, action, cuts.",
    "overall_soundscape": "Diegetic sound: room, weather, movement, voices.",
    "non_diegetic_music": "Score: instrumentation, mood, how it tracks the cut.",
}


class PromptSection(QWidget):
    """One named box: a header line that folds the editor away."""

    changed = Signal()
    focused = Signal(object)
    snippet_requested = Signal(object, object)     # (section, Snippet)
    task_type_requested = Signal(object, str)      # (section, task type)

    def __init__(self, name: str, parent=None):
        super().__init__(parent)
        self.name = name
        self._helpers_wanted = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QHBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(4)

        self.toggle = QToolButton()
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setArrowType(Qt.DownArrow)
        self.toggle.setAutoRaise(True)
        self.toggle.setText(prompt_mod.title(name))
        self.toggle.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.toggle.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.toggle.setProperty("role", "section-toggle")
        self.toggle.toggled.connect(self._on_toggled)
        header.addWidget(self.toggle, 1)

        # Says whether a folded box has anything in it, so nothing hides unnoticed.
        self.badge = style.hint("")
        style.mono(self.badge, size=8)
        header.addWidget(self.badge)
        layout.addLayout(header)

        self.snippets = SnippetRow(name)
        self.snippets.insert_requested.connect(
            lambda snippet: self.snippet_requested.emit(self, snippet))
        self.snippets.task_type_requested.connect(
            lambda task_type: self.task_type_requested.emit(self, task_type))
        layout.addWidget(self.snippets)

        self.editor = QPlainTextEdit()
        self.editor.setPlaceholderText(PLACEHOLDERS.get(name, ""))
        self.editor.setMinimumHeight(MIN_BODY_HEIGHT)
        self.editor.setTabChangesFocus(True)
        self.editor.textChanged.connect(self._on_text_changed)
        self.editor.focusInEvent = self._wrap_focus(self.editor.focusInEvent)
        layout.addWidget(self.editor, 1)

        self._refresh_badge()

    def _wrap_focus(self, original):
        def handler(event):
            self.focused.emit(self)
            original(event)
        return handler

    # -- state ---------------------------------------------------------------------

    def text(self) -> str:
        return self.editor.toPlainText()

    def set_text(self, value: str) -> None:
        if value != self.editor.toPlainText():
            self.editor.setPlainText(value or "")
        self._refresh_badge()

    def set_expanded(self, expanded: bool) -> None:
        self.toggle.setChecked(expanded)

    def is_expanded(self) -> bool:
        return self.toggle.isChecked()

    def set_helpers_visible(self, visible: bool) -> None:
        """Show or hide the chip row, independently of whether the box is folded.

        Two conditions, one row: the chips are for a box you can actually type into, so a
        folded box hides them whatever this says.
        """
        self._helpers_wanted = visible
        self.snippets.setVisible(visible and self.is_expanded())

    def _on_toggled(self, expanded: bool) -> None:
        self.toggle.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.editor.setVisible(expanded)
        self.snippets.setVisible(expanded and self._helpers_wanted)
        self.setSizePolicy(
            QSizePolicy.Preferred,
            QSizePolicy.Expanding if expanded else QSizePolicy.Fixed)

    def _on_text_changed(self) -> None:
        self._refresh_badge()
        self.changed.emit()

    def _refresh_badge(self) -> None:
        length = len(self.text().strip())
        self.badge.setText(str(length) if length else prompt_mod.EMPTY)
        self.badge.setProperty("role", "hint" if length else "warn")
        style.restyle(self.badge)


def _compose(pending: dict[str, str], migration: dict[str, str]) -> dict[str, str]:
    """Fold a fresh renumbering into one the user has not acted on yet.

    ``pending`` maps the tag as the prompt still spells it to the tag it should become;
    ``migration`` maps the tags as they were a moment ago to what they are now. Composing
    the two keeps the prompt's own spelling as the key, so two quick edits do not lose the
    first one's mapping.

    Composed from a snapshot rather than folded in place. Doing it entry by entry treats
    the *entries of a single migration* as if they were successive edits, which quietly
    cancels any permutation cycle: swapping two references gives {P1: P2, P2: P1}, and
    chaining those two turns them into P1 -> P1 and drops the rewrite entirely. Reordering
    makes cycles routine, so this has to be a composition.
    """
    composed = {
        original: migration.get(current, current)
        for original, current in pending.items()
    }
    already = set(pending.values())
    composed.update({
        old: new for old, new in migration.items() if old not in already
    })
    return {old: new for old, new in composed.items() if old != new}


class PromptEditor(QGroupBox):
    changed = Signal()

    def __init__(self, parent=None):
        # No group title: the pane this fills has a title bar that already says "PROMPTS".
        super().__init__("", parent)
        self.setProperty("role", "pane")
        self._assignment = TagAssignment()
        self._pending_migration: dict[str, str] = {}
        self._vanished: list[str] = []
        self._loading = False
        self._guide_dialog = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(4)

        layout.addLayout(self._build_header())

        body = QWidget()
        column = QVBoxLayout(body)
        column.setContentsMargins(0, 0, 0, 0)
        column.setSpacing(2)      # tight: six headers should not cost real height

        self.sections: dict[str, PromptSection] = {}
        for name in prompt_mod.SECTIONS:
            section = PromptSection(name)
            section.changed.connect(self._on_text_changed)
            section.focused.connect(self._on_section_focused)
            section.snippet_requested.connect(self._on_snippet_requested)
            section.task_type_requested.connect(self._on_task_type_requested)
            self.sections[name] = section
            column.addWidget(section)
        column.addStretch(0)

        # Kept, rather than being a local: an insert that expands a folded box and adds
        # eight lines can push the caret out of view, and only the outer area can scroll
        # to it -- the editor's own ensureCursorVisible knows nothing about this one.
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._scroll.setFrameShape(QScrollArea.NoFrame)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._scroll.setWidget(body)
        layout.addWidget(self._scroll, 1)

        self._focused = self.sections[prompt_mod.SECTIONS[0]]

        self.banner = _MigrationBanner()
        self.banner.rewrite_clicked.connect(self._apply_migration)
        self.banner.dismiss_clicked.connect(self._dismiss_migration)
        self.banner.hide()
        layout.addWidget(self.banner)

        self.lint_label = QLabel("")
        self.lint_label.setWordWrap(True)
        self.lint_label.hide()
        layout.addWidget(self.lint_label)

    def _build_header(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(4)

        expand = QPushButton("Expand all")
        expand.setFlat(True)
        expand.clicked.connect(lambda: self.set_all_expanded(True))
        row.addWidget(expand)

        collapse = QPushButton("Collapse all")
        collapse.setFlat(True)
        collapse.clicked.connect(lambda: self.set_all_expanded(False))
        row.addWidget(collapse)

        row.addStretch(1)

        self.helpers_button = QPushButton("Helpers")
        self.helpers_button.setFlat(True)
        self.helpers_button.setCheckable(True)
        self.helpers_button.setChecked(True)
        self.helpers_button.setToolTip(
            "Show the format chips under each section.\n\n"
            "Turn them off to give the editors the height back once the format is in "
            "your fingers.")
        self.helpers_button.toggled.connect(self.set_helpers_visible)
        row.addWidget(self.helpers_button)

        self.guide_button = QPushButton("Guide")
        self.guide_button.setFlat(True)
        self.guide_button.setToolTip("Open the prompt writing guide beside the editor")
        self.guide_button.clicked.connect(self._show_guide)
        # Hidden rather than disabled when the document is not there: the chips carry the
        # format themselves, so its absence is not a degraded state worth pointing at.
        self.guide_button.setVisible(config.GUIDE_PATH.is_file())
        row.addWidget(self.guide_button)

        return row

    # -- content -------------------------------------------------------------------

    def sections_text(self) -> dict:
        return {name: section.text() for name, section in self.sections.items()}

    def set_sections(self, sections) -> None:
        self._loading = True
        try:
            filled = prompt_mod.normalise(sections)
            for name, section in self.sections.items():
                section.set_text(filled[name])
                # Open what has something in it, fold the rest away.
                section.set_expanded(bool(filled[name].strip()))
            if not prompt_mod.filled_names(filled):
                self.sections[prompt_mod.SECTIONS[0]].set_expanded(True)
        finally:
            self._loading = False
        # Past the `finally`, because `_on_text_changed` returns early while loading and a
        # freshly-opened scene would otherwise show no subject chips until the first key.
        self._refresh_subject_chips()
        self._refresh_lint()

    def text(self) -> str:
        """The combined prompt, exactly as the model will receive it."""
        return prompt_mod.combine(self.sections_text())

    def set_all_expanded(self, expanded: bool) -> None:
        for section in self.sections.values():
            section.set_expanded(expanded)

    def insert_tag(self, tag: str) -> None:
        """Insert a reference tag at the caret, padded so it cannot glue to a neighbour."""
        self.insert_snippet(tag)

    def insert_snippet(self, text: str, select: str | None = None, section=None,
                       block: bool | None = None) -> None:
        """Put a fragment of the output format into a section, ready to be typed over.

        Clicking into the middle of existing text is the normal case, so a bare insert
        would routinely produce "Use<Picture 1>as" - the surrounding characters decide
        whether a space is needed on each side.

        Two things beyond that. A ``block`` fragment is a *line* of the format rather than
        a phrase inside one, so it is separated by newlines instead: gluing "[Shot 2] At
        ..." onto the tail of the previous shot is precisely the error the chips exist to
        prevent. That is a property of the fragment, not of its text - a shot line is one
        line long and still must not be glued on - so the caller says which it is, and
        multi-line text is only the obvious default. And a fragment with a hole in it
        leaves that hole selected, so the next keystroke replaces it: a placeholder the
        user has to go and find is worse than none at all.
        """
        section = section or self._focused
        self._focused = section
        section.set_expanded(True)

        editor = section.editor
        cursor = editor.textCursor()
        existing = editor.toPlainText()
        start = min(cursor.selectionStart(), cursor.selectionEnd())
        end = max(cursor.selectionStart(), cursor.selectionEnd())

        before = existing[start - 1] if start > 0 else ""
        after = existing[end] if end < len(existing) else ""
        gap = "\n" if (("\n" in text) if block is None else block) else " "
        lead = gap if before and not before.isspace() else ""
        trail = gap if after and not after.isspace() else ""

        # One insertText, so one undo takes the whole skeleton back out again.
        cursor.beginEditBlock()
        cursor.insertText(f"{lead}{text}{trail}")
        cursor.endEditBlock()

        hole = text.find(select) if select else -1
        if hole >= 0:
            anchor = start + len(lead) + hole
            cursor.setPosition(anchor)
            cursor.setPosition(anchor + len(select), cursor.MoveMode.KeepAnchor)
        elif trail:
            # Leave the caret after the fragment rather than after the padding.
            cursor.movePosition(cursor.MoveOperation.Left)

        editor.setTextCursor(cursor)
        editor.ensureCursorVisible()
        self._scroll.ensureWidgetVisible(section)
        # Last: the selection above is only drawn once the editor has focus back.
        editor.setFocus()

    def apply_task_type(self, task_type: str) -> None:
        """Add a task type to the summary's bracketed prefix, or start one.

        Not an insert. The guide's rule - combine with " + " and never repeat - is about
        the prefix as a whole, so the chip rewrites it. Through a cursor, though, and never
        ``set_text``: that calls ``setPlainText``, which would throw away the user's entire
        undo history the moment they clicked a task type.
        """
        section = self.sections["summary"]
        section.set_expanded(True)
        merged = snippets_mod.merge_task_type(section.text(), task_type)
        if merged is None:
            return                       # already carries this type; nothing to say

        replacement, start, end = merged
        editor = section.editor
        cursor = editor.textCursor()
        cursor.setPosition(start)
        cursor.setPosition(end, cursor.MoveMode.KeepAnchor)
        cursor.beginEditBlock()
        cursor.insertText(replacement)
        cursor.endEditBlock()
        editor.setTextCursor(cursor)
        self._scroll.ensureWidgetVisible(section)
        editor.setFocus()

    def set_helpers_visible(self, visible: bool) -> None:
        for section in self.sections.values():
            section.set_helpers_visible(visible)
        if self.helpers_button.isChecked() != visible:
            self.helpers_button.setChecked(visible)

    def helpers_visible(self) -> bool:
        return self.helpers_button.isChecked()

    def _on_snippet_requested(self, section: PromptSection, snippet) -> None:
        """Insert a chip's fragment, computing the two that depend on what is written."""
        if snippet.label == snippets_mod.NEXT_SHOT_LABEL:
            snippet = snippets_mod.next_shot_snippet(section.text())
        elif snippet.label == snippets_mod.NEXT_SUBJECT_LABEL:
            snippet = snippets_mod.next_subject_snippet(
                self.sections[prompt_mod.SECTIONS[0]].text())
        self.insert_snippet(snippet.text, snippet.select, section, snippet.is_block)

    def _on_task_type_requested(self, _section: PromptSection, task_type: str) -> None:
        self.apply_task_type(task_type)

    def _show_guide(self) -> None:
        text = snippets_mod.load_guide()
        if text is None:
            # It can also go missing while the app is running.
            self.guide_button.hide()
            QMessageBox.information(
                self, "Guide not found",
                f"The prompt writing guide is not at {config.GUIDE_PATH}.\n\n"
                "The chips under each section carry the format either way.")
            return
        if self._guide_dialog is None:
            self._guide_dialog = GuideDialog(text, config.GUIDE_PATH, self)
        self._guide_dialog.show()
        self._guide_dialog.raise_()

    def _refresh_subject_chips(self) -> None:
        """Offer every subject the prompt defines, everywhere except where it is defined.

        A chip for a subject the definitions never introduce would hand out a label the
        model has never been told about, so this reads one box and offers it to the others.
        """
        defined = refs_mod.defined_subjects(
            self.sections[prompt_mod.SECTIONS[0]].text())
        for name, section in self.sections.items():
            section.snippets.set_subjects([] if name == prompt_mod.SECTIONS[0] else defined)

    def _on_section_focused(self, section: PromptSection) -> None:
        self._focused = section

    def _on_text_changed(self) -> None:
        if self._loading:
            return
        self._refresh_subject_chips()
        self._refresh_lint()
        self.changed.emit()

    # -- tags ----------------------------------------------------------------------

    def update_tags(self, assignment: TagAssignment) -> None:
        """Adopt a new tag assignment, offering a rewrite if anything was renumbered."""
        migration = refs_mod.tag_migration(self._assignment, assignment)
        # A tag can also vanish outright -- unticking a soundtrack removes its <Audio j>
        # while a later reference slides into that same number, so the prompt keeps
        # reading correctly while pointing somewhere new. Worth saying out loud.
        mentioned = refs_mod.tags_in_prompt(self.text())
        vanished = [
            tag for tag in self._assignment.order
            if tag not in assignment.all_tags() and tag in mentioned
        ]
        self._assignment = assignment

        if migration:
            self._pending_migration = _compose(self._pending_migration, migration)

        if vanished:
            self._vanished = sorted(set(self._vanished + vanished))

        if self._pending_migration or self._vanished:
            self.banner.show_migration(self._pending_migration, self._vanished)
        else:
            self.banner.hide()

        self._refresh_lint()

    def _apply_migration(self) -> None:
        if not self._pending_migration:
            self._dismiss_migration()
            return
        self._loading = True
        try:
            for section in self.sections.values():
                section.set_text(
                    refs_mod.remap_prompt(section.text(), self._pending_migration))
        finally:
            self._loading = False
        self._dismiss_migration()
        self.changed.emit()

    def _dismiss_migration(self) -> None:
        self._pending_migration = {}
        self._vanished = []
        self.banner.hide()
        self._refresh_lint()

    def _refresh_lint(self) -> None:
        text = self.text()
        unknown = refs_mod.unknown_tags(text, self._assignment)
        unused = refs_mod.unused_tags(text, self._assignment)

        messages = []
        if unknown:
            messages.append(
                f"{', '.join(unknown)} {'is' if len(unknown) == 1 else 'are'} referenced "
                "but not loaded."
            )
        if unused:
            messages.append(f"Loaded but never mentioned: {', '.join(unused)}.")

        self.lint_label.setText("  ".join(messages))
        self.lint_label.setProperty("role", "warn" if messages else "hint")
        self.lint_label.setVisible(bool(messages))
        style.restyle(self.lint_label)

    def unused_tags(self) -> list[str]:
        return refs_mod.unused_tags(self.text(), self._assignment)


class _MigrationBanner(QFrame):
    """Amber strip announcing that reference tags were renumbered."""

    rewrite_clicked = Signal()
    dismiss_clicked = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("role", "banner")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(8)

        self.label = QLabel()
        self.label.setWordWrap(True)
        layout.addWidget(self.label, 1)

        rewrite = QPushButton("Rewrite tags in prompt")
        rewrite.clicked.connect(self.rewrite_clicked.emit)
        layout.addWidget(rewrite)

        dismiss = QPushButton("Keep as is")
        dismiss.clicked.connect(self.dismiss_clicked.emit)
        layout.addWidget(dismiss)

    def show_migration(self, migration: dict[str, str], vanished: list[str] | None = None) -> None:
        parts = []
        if migration:
            parts.append("Reference tags changed: " + ", ".join(
                f"{old} → {new}" for old, new in sorted(migration.items())))
        if vanished:
            parts.append(
                f"{', '.join(vanished)} no longer exists, but your prompt still uses it."
            )
        self.label.setText("  ".join(parts))
        self.show()
