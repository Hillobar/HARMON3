"""The row of format chips that sits under a section header.

Each chip's caption is the fragment it inserts, so the row is simultaneously the control
and the reference card: the one under ``retention_analysis`` is how you insert a marker
*and* the list of which markers exist. That is the whole idea, and it is why the chips are
here rather than behind a menu.

Two rules everything in here obeys:

* **Nothing takes focus.** ``PromptSection`` learns which box the caret is in by watching
  ``focusInEvent``, and ``QPlainTextEdit`` hides its selection the moment it loses focus.
  A chip that stole focus would leave the freshly-selected placeholder looking unselected,
  and the next click would type over a selection the user cannot see.
* **The row wraps rather than widening.** It lives in a scrolling column with a horizontal
  scrollbar switched off, so a row that insisted on its natural width would push the
  editors out of the panel.
"""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QLayout,
    QPlainTextEdit,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QWidgetItem,
)

from .. import snippets as snippets_mod
from . import style

#: Gaps between chips, and between wrapped lines of them.
_H_GAP = 4
_V_GAP = 3


class FlowLayout(QLayout):
    """Left-to-right, wrapping to a new line when the width runs out.

    Qt ships no such layout. The height-for-width pair is the part that matters: without
    it the row is either clipped to one line inside the scroll area or claims a height it
    never uses.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[QWidgetItem] = []
        self.setContentsMargins(0, 0, 0, 0)

    def addItem(self, item) -> None:
        self._items.append(item)

    def count(self) -> int:
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientations(0)

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._layout(QRect(0, 0, width, 0), apply=False)

    def setGeometry(self, rect) -> None:
        super().setGeometry(rect)
        self._layout(rect, apply=True)

    def sizeHint(self) -> QSize:
        return self.minimumSize()

    def minimumSize(self) -> QSize:
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        margins = self.contentsMargins()
        return size + QSize(margins.left() + margins.right(),
                            margins.top() + margins.bottom())

    def _layout(self, rect: QRect, apply: bool) -> int:
        margins = self.contentsMargins()
        left = rect.x() + margins.left()
        right = rect.right() - margins.right()
        x, y, line_height = left, rect.y() + margins.top(), 0

        for item in self._items:
            hint = item.sizeHint()
            if x > left and x + hint.width() > right:
                x, y = left, y + line_height + _V_GAP
                line_height = 0
            if apply:
                item.setGeometry(QRect(QPoint(x, y), hint))
            x += hint.width() + _H_GAP
            line_height = max(line_height, hint.height())

        return y + line_height + margins.bottom() - rect.y()


class SnippetChip(QLabel):
    """A fragment of the format that inserts itself when clicked.

    The same object as ``ref_panel.TagChip`` in everything but colour and what it carries:
    the two rows end up a few pixels apart, and someone who has learnt that a bordered pill
    means "click to insert" should not have to learn it twice.
    """

    clicked = Signal(object)

    def __init__(self, snippet: snippets_mod.Snippet, role: str = "snippet", parent=None):
        super().__init__(snippet.label, parent)
        self.snippet = snippet
        self.setProperty("role", role)
        style.mono(self, size=8)
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.NoFocus)
        self.setToolTip(snippet.hint or "Click to insert this into the prompt")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit(self.snippet)
        super().mousePressEvent(event)


class SnippetRow(QWidget):
    """Everything clickable for one section: its fixed vocabulary, then its subjects.

    The subject chips are the reason this exists at all. ``<Subject 3>`` is a label the
    user invented, which the model resolves by matching the string -- and the prompts in
    this repo already contain ``<Subjkect 1>`` and ``<Subjct 1>``. Once it is defined once,
    it should never be typed again.
    """

    insert_requested = Signal(object)          # a Snippet
    task_type_requested = Signal(str)

    def __init__(self, section: str, parent=None):
        super().__init__(parent)
        self.section = section
        self._subjects: list[str] = []

        self.setFocusPolicy(Qt.NoFocus)
        # setHeightForWidth on the policy, not just on the layout: the column above only
        # asks a child how tall it is at a given width when its policy says to.
        policy = QSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        policy.setHeightForWidth(True)
        self.setSizePolicy(policy)

        self._layout = FlowLayout(self)
        self._fixed_count = 0
        for entry in snippets_mod.rows_for(section):
            if isinstance(entry, str):
                self._layout.addWidget(self._caption(entry))
            else:
                self._layout.addWidget(self._chip(entry))
            self._fixed_count += 1

    # -- construction ------------------------------------------------------------------

    def _caption(self, text: str) -> QLabel:
        label = style.hint(text)
        style.mono(label, size=8)
        label.setFocusPolicy(Qt.NoFocus)
        return label

    def _chip(self, snippet: snippets_mod.Snippet, role: str = "snippet") -> SnippetChip:
        chip = SnippetChip(snippet, role)
        chip.clicked.connect(self._on_clicked)
        return chip

    # -- state -------------------------------------------------------------------------

    def set_subjects(self, subjects: list[str]) -> None:
        """Show a chip for each subject the prompt currently defines.

        Rebuilds only when the list actually changed: this is driven by ``textChanged``,
        and tearing down five rows of labels on every keystroke would both waste work and
        flicker.
        """
        if subjects == self._subjects:
            return
        self._subjects = list(subjects)

        while self._layout.count() > self._fixed_count:
            item = self._layout.takeAt(self._layout.count() - 1)
            if item.widget() is not None:
                item.widget().deleteLater()

        for tag in self._subjects:
            # Magenta, like the reference chips: these name something that exists, rather
            # than being vocabulary out of the guide.
            self._layout.addWidget(self._chip(
                snippets_mod.Snippet(tag, tag, None,
                                     "A subject you have defined. Click to use it again "
                                     "without retyping it."),
                role="tag"))
        self._layout.invalidate()
        self.updateGeometry()

    def chips(self) -> list[SnippetChip]:
        widgets = (self._layout.itemAt(i).widget() for i in range(self._layout.count()))
        return [w for w in widgets if isinstance(w, SnippetChip)]

    def _on_clicked(self, snippet: snippets_mod.Snippet) -> None:
        if snippet.task_type:
            self.task_type_requested.emit(snippet.task_type)
        else:
            self.insert_requested.emit(snippet)

    # -- geometry ----------------------------------------------------------------------

    def hasHeightForWidth(self) -> bool:
        return True

    def heightForWidth(self, width: int) -> int:
        return self._layout.heightForWidth(width)


class GuideDialog(QDialog):
    """The format spec, read-only, beside the editor rather than over it.

    Modelled on the log window, with one difference: shown rather than executed, because
    the point of having the guide open is to write while looking at it.
    """

    def __init__(self, text: str, path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Prompt writing guide")
        self.resize(820, 640)
        self._path = path

        layout = QVBoxLayout(self)

        view = QPlainTextEdit(text)
        view.setReadOnly(True)
        view.setLineWrapMode(QPlainTextEdit.WidgetWidth)
        style.mono(view, size=9)
        layout.addWidget(view, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        external = buttons.addButton("Open in editor", QDialogButtonBox.ActionRole)
        external.clicked.connect(self._open_externally)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _open_externally(self) -> None:
        from PySide6.QtCore import QUrl

        QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._path)))
