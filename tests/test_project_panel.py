"""The project tree: two levels, a running total, and drops that are only reported.

The drop logic is the part worth pinning. It turns a position in a tree into a
(project, index) pair, and getting it wrong moves a scene somewhere the user did not
point at -- which is silent, because the rebuild afterwards looks perfectly tidy.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from harmon3.scenes import UNGROUPED, Scene, SceneStore   # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def store(tmp_path):
    return SceneStore(tmp_path / "scenes")


@pytest.fixture
def panel(qapp):
    from harmon3.ui.scene_panel import ScenePanel
    widget = ScenePanel()
    yield widget
    widget.deleteLater()


def _show(panel, store, current=None, dirty=False):
    panel.refresh(store.scenes, store.project_names(), current, dirty)


def _rows(panel):
    """The tree as {top-level label: [child labels]}."""
    tree = panel.tree
    out = {}
    for i in range(tree.topLevelItemCount()):
        item = tree.topLevelItem(i)
        out[item.text(0)] = [item.child(c).text(0) for c in range(item.childCount())]
    return out


def _filled(store, project, *names):
    for name in names:
        store.set_project(store.save(Scene(name=name)), project)


# ------------------------------------------------------------------------- grouping

def test_scenes_appear_under_their_project(panel, store):
    _filled(store, store.create_project("Hero film"), "Opening", "Rooftop")
    store.save(Scene(name="Loose"))
    _show(panel, store)

    assert _rows(panel) == {"Hero film": ["1. Opening", "2. Rooftop"],
                            "Ungrouped": ["Loose"]}


def test_a_project_row_carries_the_running_total(panel, store):
    project = store.create_project("Hero film")
    for name, duration in (("A", 5.0), ("B", 10.0)):
        store.set_project(store.save(Scene(name=name, duration_seconds=duration)),
                          project)
    _show(panel, store)

    row = panel.tree.topLevelItem(0)
    assert row.text(1) == "2 scene(s)"
    assert row.text(2).endswith("s")
    assert float(row.text(2).rstrip("s")) == pytest.approx(15.3, abs=0.4)


def test_only_scenes_in_a_project_are_numbered(panel, store):
    """A number implies a sequence, and the ungrouped pile is not one."""
    store.save(Scene(name="Loose"))
    _show(panel, store)
    assert _rows(panel) == {"Ungrouped": ["Loose"]}


def test_the_ungrouped_row_disappears_once_everything_is_filed(panel, store):
    _filled(store, store.create_project("Hero film"), "Only")
    _show(panel, store)
    assert list(_rows(panel)) == ["Hero film"]


def test_an_empty_project_still_gets_a_row_to_drag_onto(panel, store):
    store.create_project("Nothing yet")
    _show(panel, store)

    row = panel.tree.topLevelItem(0)
    assert row.text(0) == "Nothing yet"
    assert "drag" in row.text(1)


def test_the_loaded_scene_is_marked_wherever_it_is_filed(panel, store):
    project = store.create_project("Hero film")
    _filled(store, project, "Opening")
    current = store.find("Opening")
    _show(panel, store, current=current)

    item = panel.tree.topLevelItem(0).child(0)
    assert item.text(0).startswith("1. Opening")

    panel.update_current(current, dirty=True)
    assert "(modified)" in panel.tree.topLevelItem(0).child(0).text(0)


def test_selecting_and_finding_a_scene_reaches_into_the_groups(panel, store):
    project = store.create_project("Hero film")
    _filled(store, project, "Opening", "Rooftop")
    _show(panel, store)

    target = store.find("Rooftop")
    panel.select(target)
    assert panel.selected_scene() is target


def test_folds_survive_the_rebuild_after_every_save(panel, store):
    _filled(store, store.create_project("Hero film"), "Opening")
    _show(panel, store)

    panel.tree.topLevelItem(0).setExpanded(False)
    _show(panel, store)
    assert panel.tree.topLevelItem(0).isExpanded() is False

    panel.expand_project("Hero film")
    _show(panel, store)
    assert panel.tree.topLevelItem(0).isExpanded() is True


# ----------------------------------------------------------------------------- drops

class _Drop:
    """The two things a drop event is asked for."""

    def __init__(self, point):
        self._point = point

    def position(self):
        return self

    def toPoint(self):
        return self._point


def _drop_target(panel, item, indicator):
    """Ask the tree where a drop on `item` with `indicator` would land."""
    from PySide6.QtWidgets import QAbstractItemView

    tree = panel.tree
    rect = tree.visualItemRect(item)
    tree.dropIndicatorPosition = lambda _pos=indicator: _pos
    return tree._target(_Drop(rect.center()))


def test_a_drop_onto_a_project_row_means_the_end_of_it(panel, store):
    from PySide6.QtWidgets import QAbstractItemView

    _filled(store, store.create_project("Hero film"), "Opening")
    _show(panel, store)

    row = panel.tree.topLevelItem(0)
    assert _drop_target(panel, row, QAbstractItemView.OnItem) == ("Hero film", -1)


def test_a_project_row_accepts_whatever_the_drop_indicator_says(panel, store):
    """Two things at once. Qt reports Above/Below within a few pixels of a row, so those
    must not be dead bands -- and OnViewport is what the indicator still holds if it is
    read before Qt updates it, which is exactly how the forbidden cursor happened."""
    from PySide6.QtWidgets import QAbstractItemView

    _filled(store, store.create_project("Hero film"), "Opening")
    _show(panel, store)

    row = panel.tree.topLevelItem(0)
    for indicator in (QAbstractItemView.AboveItem, QAbstractItemView.BelowItem,
                      QAbstractItemView.OnItem, QAbstractItemView.OnViewport):
        assert _drop_target(panel, row, indicator) == ("Hero film", -1)


def test_a_drop_between_two_scenes_takes_that_slot(panel, store):
    from PySide6.QtWidgets import QAbstractItemView

    _filled(store, store.create_project("Hero film"), "Opening", "Rooftop", "Chase")
    _show(panel, store)

    second = panel.tree.topLevelItem(0).child(1)
    assert _drop_target(panel, second, QAbstractItemView.AboveItem) == ("Hero film", 1)
    assert _drop_target(panel, second, QAbstractItemView.BelowItem) == ("Hero film", 2)


def test_a_drop_onto_the_ungrouped_row_files_a_scene_nowhere(panel, store):
    from PySide6.QtWidgets import QAbstractItemView

    _filled(store, store.create_project("Hero film"), "Opening")
    store.save(Scene(name="Loose"))
    _show(panel, store)

    ungrouped = panel.tree.topLevelItem(1)
    assert _drop_target(panel, ungrouped, QAbstractItemView.OnItem) == (UNGROUPED, -1)


def test_a_drop_on_empty_space_lands_nowhere(panel, store):
    _show(panel, store)
    assert panel.tree._target(_Drop(panel.tree.viewport().rect().bottomRight())) is None


def test_a_scene_cannot_be_dropped_into_another_scene(panel, store):
    """Only project rows accept drops; a scene inside a scene has no meaning."""
    from PySide6.QtCore import Qt

    _filled(store, store.create_project("Hero film"), "Opening")
    _show(panel, store)

    child = panel.tree.topLevelItem(0).child(0)
    assert not (child.flags() & Qt.ItemIsDropEnabled)
    assert child.flags() & Qt.ItemIsDragEnabled


def _real_drag(panel, source_item, target_item):
    """Drive Qt's own drag events, rather than stubbing the drop indicator.

    The bug this guards against lived entirely in the ordering inside dragMoveEvent, so a
    test that supplies the indicator itself would have passed throughout.
    """
    from PySide6.QtCore import QPoint, Qt
    from PySide6.QtGui import QDragMoveEvent, QDropEvent

    tree = panel.tree
    tree.resize(600, 400)
    tree.setCurrentItem(source_item)
    mime = tree.mimeData([source_item])
    point = tree.visualItemRect(target_item).center()

    move = QDragMoveEvent(point, Qt.MoveAction, mime, Qt.LeftButton, Qt.NoModifier)
    tree.dragMoveEvent(move)
    accepted = move.isAccepted()

    drop = QDropEvent(point, Qt.MoveAction, mime, Qt.LeftButton, Qt.NoModifier)
    tree.dropEvent(drop)
    return accepted


def test_dragging_an_ungrouped_scene_onto_a_project_is_allowed(panel, store):
    """The forbidden cursor came from reading the drop indicator before Qt set it."""
    _filled(store, store.create_project("Hero film"), "Opening")
    store.save(Scene(name="Loose"))
    _show(panel, store)

    seen = []
    panel.scene_moved.connect(lambda *args: seen.append(args))

    project_row = panel.tree.topLevelItem(0)
    loose_row = panel.tree.topLevelItem(1).child(0)
    accepted = _real_drag(panel, loose_row, project_row)

    assert accepted is True                      # not the forbidden cursor
    assert seen == [(store.find("Loose"), "Hero film", -1)]


def test_a_drag_over_nothing_is_still_refused(panel, store):
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QDragMoveEvent

    _filled(store, store.create_project("Hero film"), "Opening")
    _show(panel, store)
    tree = panel.tree
    tree.resize(600, 400)

    source = tree.topLevelItem(0).child(0)
    tree.setCurrentItem(source)
    event = QDragMoveEvent(tree.viewport().rect().bottomRight(), Qt.MoveAction,
                           tree.mimeData([source]), Qt.LeftButton, Qt.NoModifier)
    tree.dragMoveEvent(event)
    assert event.isAccepted() is False


def test_the_move_signal_carries_the_scene_and_where_it_landed(panel, store):
    _filled(store, store.create_project("Hero film"), "Opening")
    _show(panel, store)

    seen = []
    panel.scene_moved.connect(lambda *args: seen.append(args))
    panel.tree.scene_dropped.emit(store.find("Opening"), "Hero film", 2)
    assert seen == [(store.find("Opening"), "Hero film", 2)]
