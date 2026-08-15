"""The project catalogue: scenes grouped into the videos they are pieces of.

Two levels, one tree. A project is a top-level row carrying the running total of the
scenes under it; a scene is a child row, numbered by its place in the sequence. Scenes
belonging to no project sit under a standing "Ungrouped" row, which is a display device
rather than a project -- it cannot be renamed, deleted, or reordered.

Dragging is how a scene changes hands. The tree does not move anything itself: a drop
works out where it landed, says so, and waits to be handed a rebuilt catalogue, so the
row order on screen can never disagree with what is on disk.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QBrush, QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QHeaderView,
    QInputDialog,
    QLayout,
    QLineEdit,
    QMenu,
    QMessageBox,
    QPushButton,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from .. import mathmirror
from ..scenes import (
    MAX_DESCRIPTION_LENGTH,
    MAX_NAME_LENGTH,
    MAX_PROJECT_LENGTH,
    UNGROUPED,
    Scene,
)
from . import style
from .snippet_bar import FlowLayout

#: What the standing row for project-less scenes is called. Not a project name: no scene
#: ever carries it, and the store never sees it.
UNGROUPED_LABEL = "Ungrouped"

#: Roles on a row. A project row carries its name and no scene; a scene row the reverse.
SCENE_ROLE = Qt.UserRole
PROJECT_ROLE = Qt.UserRole + 1


class SceneDetailsDialog(QDialog):
    """Name and description, used both when saving a new scene and when editing one."""

    def __init__(self, parent, title: str, name: str = "", description: str = "",
                 accept_label: str = "Save"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(420)

        layout = QVBoxLayout(self)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignRight)

        self.name_edit = QLineEdit(name)
        self.name_edit.setMaxLength(MAX_NAME_LENGTH)
        self.name_edit.setPlaceholderText("Rooftop hero")
        form.addRow("Name", self.name_edit)

        self.description_edit = QLineEdit(description)
        self.description_edit.setMaxLength(MAX_DESCRIPTION_LENGTH)
        self.description_edit.setPlaceholderText("What this scene is for (optional)")
        form.addRow("Description", self.description_edit)

        layout.addLayout(form)

        hint = style.hint(
            "A scene stores its references, prompt, length and seed settings."
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.button(QDialogButtonBox.Ok).setText(accept_label)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        layout.addWidget(self.buttons)

        self.name_edit.textChanged.connect(self._update_ok)
        self._update_ok()
        self.name_edit.setFocus()
        self.name_edit.selectAll()

    def _update_ok(self) -> None:
        self.buttons.button(QDialogButtonBox.Ok).setEnabled(bool(self.name_edit.text().strip()))

    def values(self) -> tuple[str, str]:
        return self.name_edit.text().strip(), self.description_edit.text().strip()


class _ProjectTree(QTreeWidget):
    """A two-level tree whose drops are reported rather than performed.

    Qt's own InternalMove would reparent the row it was given and leave the store to catch
    up, which puts the display one step ahead of the disk for as long as the write takes --
    and permanently, if it fails. So the drop is turned into (scene, project, position),
    handed over, and nothing moves until a rebuilt catalogue comes back.
    """

    scene_dropped = Signal(object, str, int)     # scene, project name (or ""), position

    def __init__(self, parent=None):
        super().__init__(parent)
        self._dragged: Scene | None = None
        self.setDragEnabled(True)
        self.setAcceptDrops(True)
        self.setDropIndicatorShown(True)
        self.setDragDropMode(QAbstractItemView.InternalMove)
        self.setDefaultDropAction(Qt.MoveAction)

    def startDrag(self, actions) -> None:
        """Remember what is being dragged, rather than asking afterwards.

        ``currentItem`` is close enough most of the time and wrong exactly when it
        matters: anything that changes the current row mid-drag would send the wrong
        scene to the drop.
        """
        self._dragged = _scene_of(self.currentItem())
        try:
            super().startDrag(actions)
        finally:
            self._dragged = None

    def dragEnterEvent(self, event) -> None:
        """Let our own drags in. Anything from elsewhere is not ours to interpret."""
        if self._scene_being_dragged() is not None and event.source() is self:
            event.acceptProposedAction()
            return
        super().dragEnterEvent(event)

    def dragMoveEvent(self, event) -> None:
        # super() first, always. It is what computes dropIndicatorPosition(), which
        # _target then reads -- asking before it runs returns the *previous* move's
        # answer, and refusing on that answer means super() never runs again and the
        # cursor stays forbidden for the rest of the drag.
        super().dragMoveEvent(event)
        # Then our own verdict, not the base class's. It decides on the model's flags,
        # which describe a move it is not going to be allowed to make anyway -- so
        # deferring to it can only forbid a drop this widget would have accepted.
        if self._scene_being_dragged() is not None and self._target(event) is not None:
            event.acceptProposedAction()
        else:
            event.ignore()

    def _scene_being_dragged(self) -> Scene | None:
        return self._dragged or _scene_of(self.currentItem())

    def dropEvent(self, event) -> None:
        scene = self._scene_being_dragged()
        target = self._target(event)
        if scene is None or target is None:
            event.ignore()
            return

        project, position = target
        # Accepted with IgnoreAction: the drop is a request, and letting Qt also move the
        # row would double the change the moment the rebuild arrives. It also keeps the
        # view's own startDrag from deleting the row it thinks it just moved away.
        event.setDropAction(Qt.IgnoreAction)
        event.accept()
        self.scene_dropped.emit(scene, project, position)

    def _target(self, event) -> tuple[str, int] | None:
        """Which project a drop lands in, and where in its order. None if nowhere.

        A drop anywhere on a project row means "the end of this project", which is what
        dragging onto a folder means everywhere else. A drop between two scenes means
        that slot.
        """
        item = self.itemAt(event.position().toPoint())
        if item is None:
            return None

        position = self.dropIndicatorPosition()
        project = _project_of(item)
        if project is not None:
            # Above and below included on purpose. Qt reports those within a few pixels
            # of a row's edge, and a band of "forbidden" along every project row reads as
            # the feature being broken -- while the intent, on a row you are pointing at,
            # is not actually in doubt.
            return project, -1                      # -1: onto the end

        scene = _scene_of(item)
        if scene is None:
            return None
        parent = item.parent()
        project = _project_of(parent) if parent is not None else UNGROUPED
        index = parent.indexOfChild(item) if parent is not None else 0
        if position in (QAbstractItemView.BelowItem, QAbstractItemView.OnItem):
            index += 1                              # onto a scene: just after it
        return project, index


def _scene_of(item) -> Scene | None:
    return item.data(0, SCENE_ROLE) if item is not None else None


def _project_of(item) -> str | None:
    """The project a *project row* stands for. None if this is not one."""
    return item.data(0, PROJECT_ROLE) if item is not None else None


class ScenePanel(QWidget):
    """Lists the saved scenes and the actions that operate on them.

    The panel owns no state of its own: the main window holds the store and the notion of
    which scene is loaded, and calls refresh() whenever either changes.
    """

    save_as_requested = Signal(str, str)       # name, description for a new scene
    update_requested = Signal(object)          # overwrite this Scene from the editor
    load_requested = Signal(object)
    run_requested = Signal(object)
    details_requested = Signal(object, str, str)   # scene, new name, new description
    duplicate_requested = Signal(object)
    delete_requested = Signal(object)
    revert_requested = Signal(object)
    project_create_requested = Signal(str)
    project_rename_requested = Signal(str, str)     # old, new
    project_delete_requested = Signal(str)
    scene_moved = Signal(object, str, int)          # scene, project (or ""), position

    def __init__(self, parent=None):
        super().__init__(parent)
        self._current: Scene | None = None
        self._dirty = False
        #: Projects the user has folded away. Tracked as collapses rather than as
        #: expansions so that a project appearing for the first time is open, while
        #: folding the only project does not read as "nothing remembered, open it".
        self._collapsed: set[str] = set()

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)

        layout.addWidget(self.current_label_row())
        layout.addLayout(self._build_header())

        self.tree = _ProjectTree()
        self.tree.setColumnCount(5)
        self.tree.setHeaderLabels(
            ["Project / scene", "Description", "Length", "References", "Prompt"])
        self.tree.setRootIsDecorated(True)
        self.tree.setAlternatingRowColors(True)
        self.tree.setSelectionMode(QAbstractItemView.SingleSelection)
        self.tree.setContextMenuPolicy(Qt.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_menu)
        self.tree.itemDoubleClicked.connect(self._on_double_clicked)
        self.tree.itemSelectionChanged.connect(self._update_buttons)
        self.tree.itemExpanded.connect(self._remember_expanded)
        self.tree.itemCollapsed.connect(self._remember_expanded)
        self.tree.scene_dropped.connect(self.scene_moved.emit)

        header = self.tree.header()
        header.setSectionResizeMode(0, QHeaderView.Interactive)   # Scene
        header.setSectionResizeMode(1, QHeaderView.Interactive)   # Description
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.Stretch)       # Prompt
        self.tree.setColumnWidth(0, 160)
        self.tree.setColumnWidth(1, 200)
        layout.addWidget(self.tree, 1)

        layout.addLayout(self._build_actions())

        self.hint_label = style.hint(
            "A project is the finished video its scenes are pieces of. Drag a scene onto "
            "a project to file it there, or between scenes to set its place in the order. "
            "A scene stores its references, prompt, length and seed; resolution is a "
            "render setting and stays as you have it."
        )
        self.hint_label.setWordWrap(True)
        layout.addWidget(self.hint_label)

        self._update_buttons()

    def current_label_row(self) -> QWidget:
        """Which scene is loaded, on a line of its own.

        Beside the buttons it set a floor under the width of the whole panel; the panel is
        a column now, and columns are narrow.
        """
        self.current_label = style.ElidedLabel("No scene loaded")
        style.mono(self.current_label, size=9)
        self.current_label.setProperty("role", "hint")
        return self.current_label

    def _build_header(self) -> QLayout:
        # Wrapping rather than one row: four buttons side by side demanded more width than
        # the column this panel now lives in has to give.
        row = FlowLayout()

        # The only way a project comes into existence. Dragging is how scenes get *into*
        # one, which needs a row to drop onto first.
        self.new_project_button = QPushButton("New project...")
        self.new_project_button.setToolTip(
            "Start an empty project, then drag scenes onto it")
        self.new_project_button.clicked.connect(self._on_new_project)
        row.addWidget(self.new_project_button)

        self.save_button = QPushButton("Save as scene...")
        self.save_button.setToolTip("Save the current references, prompt, length and seed")
        self.save_button.clicked.connect(self._on_save_as)
        row.addWidget(self.save_button)

        self.update_button = QPushButton("Update")
        self.update_button.setToolTip("Overwrite the loaded scene with the current editor")
        self.update_button.clicked.connect(
            lambda: self._current and self.update_requested.emit(self._current))
        row.addWidget(self.update_button)

        self.revert_button = QPushButton("Revert")
        self.revert_button.setToolTip("Discard the edits and reload the saved scene")
        self.revert_button.clicked.connect(
            lambda: self._current and self.revert_requested.emit(self._current))
        row.addWidget(self.revert_button)

        return row

    def _build_actions(self) -> QLayout:
        row = FlowLayout()

        self.load_button = QPushButton("Load")
        self.load_button.setToolTip("Put this scene into the editor")
        self.load_button.clicked.connect(lambda: self._emit(self.load_requested))
        row.addWidget(self.load_button)

        self.run_button = QPushButton("Load and run")
        self.run_button.setToolTip("Load this scene and queue it immediately")
        self.run_button.clicked.connect(lambda: self._emit(self.run_requested))
        row.addWidget(self.run_button)

        self.details_button = QPushButton("Details...")
        self.details_button.setToolTip("Edit this scene's name and description")
        self.details_button.clicked.connect(
            lambda: self.selected_scene() and self._on_edit_details(self.selected_scene()))
        row.addWidget(self.details_button)

        self.duplicate_button = QPushButton("Duplicate")
        self.duplicate_button.clicked.connect(lambda: self._emit(self.duplicate_requested))
        row.addWidget(self.duplicate_button)

        self.delete_button = QPushButton("Delete")
        self.delete_button.clicked.connect(self._on_delete)
        row.addWidget(self.delete_button)

        return row

    # -- population ----------------------------------------------------------------

    def refresh(self, scenes: list[Scene], projects: list[str], current: Scene | None,
                dirty: bool) -> None:
        """Rebuild the whole tree. Only for when the catalogue itself changes.

        ``projects`` comes from the store rather than being derived from the scenes,
        because a project with nothing in it yet exists in exactly one place -- the store's
        registry -- and it is the row everything else has to be dragged onto.
        """
        previous = self.selected_scene()
        self._current = current
        self._dirty = dirty

        self.tree.clear()
        to_reselect = None

        grouped: dict[str, list[Scene]] = {name: [] for name in projects}
        loose: list[Scene] = []
        for scene in scenes:
            grouped.setdefault(scene.project, []).append(scene) if scene.project \
                else loose.append(scene)
        for members in grouped.values():
            members.sort(key=lambda s: (s.project_index, s.name.casefold()))
        loose.sort(key=lambda s: s.name.casefold())

        # Ungrouped last: a project is the point of this panel, and a long tail of loose
        # scenes above them would bury the thing being assembled.
        sections = [(name, grouped.get(name, []), True) for name in projects]
        if loose or not projects:
            sections.append((UNGROUPED_LABEL, loose, False))

        for label, members, is_project in sections:
            parent = QTreeWidgetItem()
            self._fill_project_item(parent, label, members, is_project)
            self.tree.addTopLevelItem(parent)

            for order, scene in enumerate(members):
                item = QTreeWidgetItem(parent)
                self._fill_item(item, scene, is_current=scene is current, dirty=dirty,
                                position=order + 1 if is_project else None)
                if previous is not None and scene is previous:
                    to_reselect = item
                elif to_reselect is None and current is not None and scene is current:
                    to_reselect = item

            key = label if is_project else UNGROUPED
            parent.setExpanded(key not in self._collapsed)

        if to_reselect is not None:
            self.tree.setCurrentItem(to_reselect)

        self._update_current_label()
        self._update_buttons()

    def expand_project(self, name: str) -> None:
        """Make sure a project is open before the next rebuild, so a drop is visible."""
        self._collapsed.discard(name or UNGROUPED)

    def _remember_expanded(self, item: QTreeWidgetItem) -> None:
        """Keep folds across the rebuilds that follow every save, drag and rename."""
        project = _project_of(item)
        if project is None:
            return
        key = project or UNGROUPED
        if item.isExpanded():
            self._collapsed.discard(key)
        else:
            self._collapsed.add(key)

    def _fill_project_item(self, item: QTreeWidgetItem, label: str, members: list[Scene],
                           is_project: bool) -> None:
        """A project row: its name, how many scenes, and how long the whole thing runs."""
        item.setData(0, PROJECT_ROLE, label if is_project else UNGROUPED)
        item.setFirstColumnSpanned(False)
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDropEnabled)

        font = item.font(0)
        font.setBold(True)
        item.setFont(0, font)
        item.setText(0, label)
        item.setForeground(0, QBrush(QColor(style.ACCENT if is_project else style.MUTED)))

        if not members:
            item.setText(1, "empty - drag a scene here" if is_project else "")
            item.setForeground(1, QBrush(QColor(style.MUTED)))
            return

        frames = sum(mathmirror.frames_from_seconds(s.duration_seconds) for s in members)
        item.setText(1, f"{len(members)} scene(s)")
        item.setForeground(1, QBrush(QColor(style.MUTED)))
        if is_project:
            # The running total is the whole reason for grouping: it answers "how long is
            # the video so far" without adding up rows by hand.
            item.setText(2, f"{mathmirror.true_seconds(frames):.1f}s")
            item.setToolTip(0, f"{label}\n{len(members)} scene(s), "
                               f"{mathmirror.true_seconds(frames):.2f}s in total")

    def update_current(self, current: Scene | None, dirty: bool) -> None:
        """Update only the loaded-scene indicator.

        This runs from the editor's refresh, which fires on every keystroke, so it must
        not rebuild the tree -- and it does nothing at all unless something actually
        changed, which after the first edited character is the common case.
        """
        if current is self._current and dirty == self._dirty:
            return
        self._current = current
        self._dirty = dirty

        for item in self._scene_items():
            scene = _scene_of(item)
            self._mark_item(item, scene, is_current=scene is current, dirty=dirty)

        self._update_current_label()
        self._update_buttons()

    def _scene_items(self):
        """Every scene row, whichever project it is filed under."""
        for index in range(self.tree.topLevelItemCount()):
            parent = self.tree.topLevelItem(index)
            for child in range(parent.childCount()):
                yield parent.child(child)

    def _update_current_label(self) -> None:
        current, dirty = self._current, self._dirty
        if current is None:
            self.current_label.setText("No scene loaded")
            self.current_label.setProperty("role", "hint")
        elif dirty:
            self.current_label.setText(f"Editing: {current.name} (modified)")
            self.current_label.setProperty("role", "warn")
        else:
            self.current_label.setText(f"Editing: {current.name}")
            self.current_label.setProperty("role", "ok")
        style.restyle(self.current_label)

    def _mark_item(self, item: QTreeWidgetItem, scene: Scene, is_current: bool,
                   dirty: bool) -> None:
        """Apply (or clear) the "this is the loaded scene" styling on one row."""
        prefix = item.data(0, Qt.UserRole + 2) or ""
        if is_current:
            label = f"{scene.name} (modified)" if dirty else f"{scene.name}  <"
            item.setForeground(0, QBrush(QColor(style.WARN if dirty else style.ACCENT)))
        else:
            label = scene.name
            item.setForeground(0, QBrush())
        item.setText(0, f"{prefix}{label}")

        font = item.font(0)
        font.setBold(is_current)
        item.setFont(0, font)

    def _fill_item(self, item: QTreeWidgetItem, scene: Scene, is_current: bool,
                   dirty: bool, position: int | None = None) -> None:
        item.setData(0, SCENE_ROLE, scene)
        # Numbered only inside a project, where the order is the point. Numbering the
        # ungrouped pile would imply a sequence that is not being kept.
        item.setData(0, Qt.UserRole + 2, f"{position}. " if position else "")
        # Dragged, but never dropped onto: a scene cannot contain another scene, and
        # allowing it would let a drop land somewhere with no meaning.
        item.setFlags(Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsDragEnabled)
        self._mark_item(item, scene, is_current, dirty)

        frames = mathmirror.frames_from_seconds(scene.duration_seconds)
        item.setText(1, scene.description)
        item.setText(2, f"{mathmirror.true_seconds(frames):.1f}s")
        item.setText(3, scene.ref_summary())
        item.setText(4, scene.prompt_summary())

        if not scene.description:
            item.setForeground(1, QBrush(QColor(style.MUTED)))

        tooltip = [scene.name]
        if scene.description:
            tooltip.append(scene.description)
        tooltip += [
            f"{frames} frames ({mathmirror.true_seconds(frames):.2f}s)",
            f"Seed {scene.seed}" + (" (randomized each run)" if scene.randomize_seed else ""),
            scene.ref_summary(),
        ]
        if scene.updated_at:
            tooltip.append(f"Updated {scene.updated_at.replace('T', ' ')}")

        missing = scene.missing_local_files()
        if missing:
            item.setForeground(3, QBrush(QColor(style.ERROR)))
            item.setText(3, f"{scene.ref_summary()}  (!)")
            tooltip.append("Missing files:\n  " + "\n  ".join(missing))

        for column in range(5):
            item.setToolTip(column, "\n".join(tooltip))

    # -- selection -----------------------------------------------------------------

    def selected_scene(self) -> Scene | None:
        items = self.tree.selectedItems()
        return _scene_of(items[0]) if items else None

    def selected_project(self) -> str | None:
        """The project row that is selected, if the selection is one."""
        items = self.tree.selectedItems()
        return _project_of(items[0]) if items else None

    def select(self, scene: Scene | None) -> None:
        if scene is None:
            return
        for item in self._scene_items():
            if _scene_of(item) is scene:
                self.tree.setCurrentItem(item)
                return

    def _emit(self, signal) -> None:
        scene = self.selected_scene()
        if scene:
            signal.emit(scene)

    def _on_double_clicked(self, item, _column) -> None:
        scene = _scene_of(item)
        if scene:
            self.load_requested.emit(scene)
        elif _project_of(item):
            item.setExpanded(not item.isExpanded())

    def _update_buttons(self) -> None:
        selected = self.selected_scene()
        for button in (self.load_button, self.run_button, self.details_button,
                       self.duplicate_button, self.delete_button):
            button.setEnabled(selected is not None)

        self.update_button.setEnabled(self._current is not None)
        self.revert_button.setEnabled(self._current is not None and self._dirty)
        self.update_button.setText(
            "Update" if not self._dirty else "Update *")

    # -- actions -------------------------------------------------------------------

    def _on_save_as(self) -> None:
        dialog = SceneDetailsDialog(
            self, "Save scene",
            name=f"{self._current.name} copy" if self._current else "",
            description=self._current.description if self._current else "",
        )
        if dialog.exec() == QDialog.Accepted:
            name, description = dialog.values()
            if name:
                self.save_as_requested.emit(name, description)

    def _on_delete(self) -> None:
        scene = self.selected_scene()
        if not scene:
            return
        confirm = QMessageBox.question(
            self, "Delete scene",
            f"Delete the scene “{scene.name}”?\n\n"
            "The reference files themselves are not touched.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.delete_requested.emit(scene)

    def _on_edit_details(self, scene: Scene) -> None:
        dialog = SceneDetailsDialog(
            self, "Scene details", name=scene.name, description=scene.description,
            accept_label="Apply")
        if dialog.exec() != QDialog.Accepted:
            return
        name, description = dialog.values()
        if name and (name != scene.name or description != scene.description):
            self.details_requested.emit(scene, name, description)

    def _on_new_project(self) -> None:
        name, ok = QInputDialog.getText(
            self, "New project", "What is this video called?",
            QLineEdit.Normal, "")
        if ok and name.strip():
            self.project_create_requested.emit(name.strip()[:MAX_PROJECT_LENGTH])

    def _on_rename_project(self, old: str) -> None:
        name, ok = QInputDialog.getText(
            self, "Rename project", "New name", QLineEdit.Normal, old)
        if ok and name.strip() and name.strip() != old:
            self.project_rename_requested.emit(old, name.strip()[:MAX_PROJECT_LENGTH])

    def _on_delete_project(self, name: str, count: int) -> None:
        confirm = QMessageBox.question(
            self, "Delete project",
            f"Delete the project “{name}”?\n\n"
            + (f"Its {count} scene(s) are kept and move back to Ungrouped."
               if count else "It has no scenes in it."),
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self.project_delete_requested.emit(name)

    def _show_menu(self, position) -> None:
        item = self.tree.itemAt(position)
        project = _project_of(item)
        if project is not None:
            menu = QMenu(self)
            if project:                    # the Ungrouped row is not a project
                menu.addAction("Rename project...",
                               lambda: self._on_rename_project(project))
                menu.addAction("Delete project...",
                               lambda: self._on_delete_project(project, item.childCount()))
                menu.addSeparator()
            menu.addAction("New project...", self._on_new_project)
            menu.exec(self.tree.viewport().mapToGlobal(position))
            return

        scene = self.selected_scene()
        menu = QMenu(self)

        if scene:
            menu.addAction("Load", lambda: self.load_requested.emit(scene))
            menu.addAction("Load and run", lambda: self.run_requested.emit(scene))
            menu.addSeparator()
            menu.addAction("Edit name and description...",
                           lambda: self._on_edit_details(scene))
            menu.addAction("Duplicate", lambda: self.duplicate_requested.emit(scene))
            if scene is self._current:
                menu.addAction(
                    "Update from editor", lambda: self.update_requested.emit(scene))
            menu.addSeparator()
            menu.addAction("Copy prompt", lambda: _copy(scene.prompt_text))
            if scene.path:
                menu.addAction("Show file", lambda: _reveal(scene.path))
            menu.addSeparator()
            menu.addAction("Delete...", self._on_delete)
            if scene.project:
                menu.addSeparator()
                menu.addAction("Move up", lambda: self.scene_moved.emit(
                    scene, scene.project, scene.project_index - 1))
                menu.addAction("Move down", lambda: self.scene_moved.emit(
                    scene, scene.project, scene.project_index + 1))
                menu.addAction("Remove from project", lambda: self.scene_moved.emit(
                    scene, UNGROUPED, -1))
        else:
            menu.addAction("New project...", self._on_new_project)
            menu.addAction("Save the current editor as a scene...", self._on_save_as)

        menu.exec(self.tree.viewport().mapToGlobal(position))


def _copy(text: str) -> None:
    from PySide6.QtGui import QGuiApplication
    QGuiApplication.clipboard().setText(text or "")


def _reveal(path) -> None:
    from PySide6.QtCore import QUrl
    from PySide6.QtGui import QDesktopServices
    QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent)))
