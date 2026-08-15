"""The workspace: six panes the user arranges, and the default arrangement they start from.

Every major surface of the app is a ``QDockWidget`` rather than a cell in a fixed splitter,
so a pane can be dragged to another column, tabbed behind a neighbour, or floated onto a
second monitor. Qt's own docking does all of that and saves the result through
``QMainWindow.saveState``, which is why there is no docking library here.

The panes cannot be *closed*. Six of them is the whole app, and a close button on each one
buys a way to lose Prompts and no way to get it back short of a menu bar this window does
not have. They move and they float; Settings carries a Reset layout button for when an
arrangement has gone wrong.

The default is three columns of two: Projects over References, the Viewer over Prompts, the
run controls over everything else. It is built here, in one function, because it is needed
in three places -- first run, a saved layout that no longer matches the panes, and Reset --
and three copies of a layout drift apart.
"""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QDockWidget, QMainWindow, QTabWidget

#: Names the *set* of panes. ``restoreState`` happily accepts a state naming a dock that no
#: longer exists -- it just leaves a hole -- so the pane set is versioned separately from
#: the state format below, and a mismatch falls back to the default layout.
LAYOUT_SCHEMA = "3x2-v1"

#: Passed to ``saveState``/``restoreState``. Bump when the toolbars or the state format
#: change; ``restoreState`` returns False on a mismatch, which is already handled.
LAYOUT_VERSION = 1

DOCK_PROJECTS = "dock.projects"
DOCK_REFERENCES = "dock.references"
DOCK_VIEWER = "dock.viewer"
DOCK_PROMPTS = "dock.prompts"
DOCK_RUN = "dock.run"
DOCK_OTHER = "dock.other"

#: The order the default layout builds them in, and the order Reset restores.
DOCK_NAMES = (
    DOCK_PROJECTS, DOCK_REFERENCES, DOCK_VIEWER, DOCK_PROMPTS, DOCK_RUN, DOCK_OTHER,
)

#: Column widths and row heights of the default, at the 1360x900 the window opens on.
#: Carried over from the splitter sizes this replaced.
COLUMN_WIDTHS = (340, 690, 330)
LEFT_HEIGHTS = (300, 520)
MIDDLE_HEIGHTS = (520, 300)
RIGHT_HEIGHTS = (110, 710)


def make_dock(object_name: str, title: str, widget, parent=None) -> QDockWidget:
    """Wrap a panel in a pane the user can move.

    The object name is the first argument and has no default because ``saveState`` silently
    drops any dock without one -- the layout would appear to save and come back empty.
    """
    dock = QDockWidget(title.upper(), parent)
    dock.setObjectName(object_name)
    # Movable and floatable, never closable: see the module docstring.
    dock.setFeatures(QDockWidget.DockWidgetMovable | QDockWidget.DockWidgetFloatable)
    dock.setWidget(widget)
    return dock


def configure(window: QMainWindow) -> None:
    """Put the window into docking mode. Must run before any layout is restored.

    ``setDockNestingEnabled`` in particular: a nested layout saved by a window that does not
    have it set restores flattened, with the columns collapsed into one row.
    """
    window.setDockNestingEnabled(True)
    window.setDockOptions(
        QMainWindow.AnimatedDocks
        | QMainWindow.AllowNestedDocks
        | QMainWindow.AllowTabbedDocks
        | QMainWindow.GroupedDragging
    )
    # Docked tab bars sit at the bottom by default, where the accent underline the theme
    # draws under a selected tab reads as belonging to whatever is below it. Not recorded
    # by saveState, so this is applied on every start rather than only with the default.
    window.setTabPosition(Qt.AllDockWidgetAreas, QTabWidget.North)
    # With no central widget every pane is a dock and the columns can be split freely.
    window.setCentralWidget(None)


def apply_default_layout(window: QMainWindow) -> None:
    """Three columns of two, at the sizes the app opens on.

    Also the Reset: it un-floats and re-shows every pane first, so an arrangement dragged
    into a corner or left floating on a monitor that is no longer there comes back.
    """
    docks = [getattr(window, attr) for attr in _DOCK_ATTRS]
    projects, references, viewer, prompts, run, other = docks

    for dock in docks:
        dock.setFloating(False)
        dock.show()

    # Columns first, left to right, then each column split into its two rows.
    window.addDockWidget(Qt.LeftDockWidgetArea, projects)
    window.splitDockWidget(projects, viewer, Qt.Horizontal)
    window.splitDockWidget(viewer, run, Qt.Horizontal)
    window.splitDockWidget(projects, references, Qt.Vertical)
    window.splitDockWidget(viewer, prompts, Qt.Vertical)
    window.splitDockWidget(run, other, Qt.Vertical)

    # Deferred: before the window has been shown, resizeDocks clamps silently to the
    # minimum size hints, which are not yet what they will be, and the proportions land
    # nowhere near these numbers.
    QTimer.singleShot(0, lambda: apply_default_sizes(window))


def apply_default_sizes(window: QMainWindow) -> None:
    """The proportions of the default layout, applied once the window has a size."""
    projects, references, viewer, prompts, run, other = (
        getattr(window, attr) for attr in _DOCK_ATTRS)
    window.resizeDocks([projects, viewer, run], list(COLUMN_WIDTHS), Qt.Horizontal)
    window.resizeDocks([projects, references], list(LEFT_HEIGHTS), Qt.Vertical)
    window.resizeDocks([viewer, prompts], list(MIDDLE_HEIGHTS), Qt.Vertical)
    window.resizeDocks([run, other], list(RIGHT_HEIGHTS), Qt.Vertical)


#: The attribute each pane is held under on the main window, in DOCK_NAMES order.
_DOCK_ATTRS = (
    "dock_projects", "dock_references", "dock_viewer",
    "dock_prompts", "dock_run", "dock_other",
)
