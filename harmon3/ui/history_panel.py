"""Run history: every submitted job, with replay and re-queue."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QMenu,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
)

from ..history import (
    STATUS_CANCELLED,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    STATUS_SUCCESS,
    RunRecord,
)
from . import style

STATUS_GLYPH = {
    STATUS_SUCCESS: "OK",
    STATUS_FAILED: "FAIL",
    STATUS_CANCELLED: "STOP",
    STATUS_RUNNING: "...",
    # Waiting rather than working: the run bar carries no list of what is queued, so this
    # column is where the depth of the queue is actually read.
    STATUS_QUEUED: "WAIT",
}

STATUS_COLOUR = {
    STATUS_SUCCESS: style.OK,
    STATUS_FAILED: style.ERROR,
    STATUS_CANCELLED: style.WARN,
    STATUS_RUNNING: style.ACCENT,
    STATUS_QUEUED: style.TEXT_DIM,
}


class HistoryPanel(QGroupBox):
    replay_requested = Signal(object)     # RunRecord
    requeue_requested = Signal(object)    # RunRecord
    restore_requested = Signal(object)    # RunRecord -- put its settings back in the editor

    def __init__(self, parent=None):
        # No group title: this fills a tab that is already labelled "History".
        super().__init__("", parent)
        self.setProperty("role", "pane")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)

        self.tree = QTreeWidget()
        self.tree.setColumnCount(4)
        self.tree.setHeaderLabels(["", "Time", "Output", "Prompt"])
        self.tree.setRootIsDecorated(False)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_menu)
        self.tree.itemDoubleClicked.connect(self._on_double_clicked)
        self.tree.itemSelectionChanged.connect(self._update_buttons)

        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.Stretch)
        layout.addWidget(self.tree, 1)

        buttons = QHBoxLayout()
        buttons.setContentsMargins(0, 0, 0, 0)

        self.replay_button = QPushButton("Play")
        self.replay_button.setToolTip("Play the saved copy of this run's video")
        self.replay_button.clicked.connect(
            lambda: self._emit(self.replay_requested))
        buttons.addWidget(self.replay_button)

        self.restore_button = QPushButton("Load settings")
        self.restore_button.setToolTip("Put this run's prompt and settings back in the editor")
        self.restore_button.clicked.connect(
            lambda: self._emit(self.restore_requested))
        buttons.addWidget(self.restore_button)

        self.requeue_button = QPushButton("Re-queue")
        self.requeue_button.setToolTip("Submit this run's exact graph again")
        self.requeue_button.clicked.connect(
            lambda: self._emit(self.requeue_requested))
        buttons.addWidget(self.requeue_button)

        buttons.addStretch(1)
        layout.addLayout(buttons)

        self._update_buttons()

    # -- population ----------------------------------------------------------------

    def set_records(self, records: list[RunRecord]) -> None:
        self.tree.clear()
        for record in reversed(records):        # newest first
            self.tree.addTopLevelItem(self._make_item(record))
        self._update_buttons()

    def add_record(self, record: RunRecord) -> None:
        self.tree.insertTopLevelItem(0, self._make_item(record))
        self._update_buttons()

    def update_record(self, record: RunRecord) -> None:
        for index in range(self.tree.topLevelItemCount()):
            item = self.tree.topLevelItem(index)
            if item.data(0, Qt.UserRole).prompt_id == record.prompt_id:
                self._fill_item(item, record)
                return
        self.add_record(record)

    def selected_record(self) -> RunRecord | None:
        items = self.tree.selectedItems()
        return items[0].data(0, Qt.UserRole) if items else None

    # -- internals -----------------------------------------------------------------

    def _make_item(self, record: RunRecord) -> QTreeWidgetItem:
        item = QTreeWidgetItem()
        self._fill_item(item, record)
        return item

    def _fill_item(self, item: QTreeWidgetItem, record: RunRecord) -> None:
        from PySide6.QtGui import QBrush, QColor

        item.setData(0, Qt.UserRole, record)
        item.setText(0, STATUS_GLYPH.get(record.status, "?"))
        item.setForeground(0, QBrush(QColor(STATUS_COLOUR.get(record.status, style.MUTED))))

        moment = record.submitted_dt
        item.setText(1, moment.strftime("%H:%M") if moment else "")

        if record.width and record.height:
            item.setText(2, f"{record.width}x{record.height}  {record.frames}f")
        item.setText(3, record.summary())

        tooltip = [f"Seed {record.seed}", f"{record.duration_seconds:g} s requested"]
        if record.elapsed_s:
            tooltip.append(f"took {record.elapsed_s:.0f} s")
        if record.error:
            tooltip.append(record.error)
        if record.refs:
            tooltip.append("References: " + ", ".join(
                f"{r.get('tag', '?')} {r.get('name', '')}" for r in record.refs))
        item.setToolTip(3, "\n".join(tooltip))

    def _emit(self, signal) -> None:
        record = self.selected_record()
        if record:
            signal.emit(record)

    def _on_double_clicked(self, item, _column) -> None:
        record = item.data(0, Qt.UserRole)
        if record and record.video_path():
            self.replay_requested.emit(record)

    def _update_buttons(self) -> None:
        record = self.selected_record()
        self.replay_button.setEnabled(bool(record and record.video_path()))
        self.restore_button.setEnabled(record is not None)
        self.requeue_button.setEnabled(bool(record and record.graph))

    def _show_menu(self, position) -> None:
        record = self.selected_record()
        if not record:
            return

        menu = QMenu(self)
        if record.video_path():
            menu.addAction("Play", lambda: self.replay_requested.emit(record))
        menu.addAction("Load settings into editor", lambda: self.restore_requested.emit(record))
        if record.graph:
            menu.addAction("Re-queue this exact graph", lambda: self.requeue_requested.emit(record))
        menu.addSeparator()
        menu.addAction("Copy prompt", lambda: _copy(record.prompt_text))
        menu.addAction("Copy prompt id", lambda: _copy(record.prompt_id))
        if record.error:
            menu.addAction("Copy error", lambda: _copy(record.error))
        menu.exec(self.tree.viewport().mapToGlobal(position))


def _copy(text: str) -> None:
    from PySide6.QtGui import QGuiApplication
    QGuiApplication.clipboard().setText(text or "")
