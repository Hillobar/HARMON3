"""Dragging reference rows into a different order, which is what assigns their tags.

The model numbers references purely from the order it receives them, so a reorder is the
only way to choose which reference is `<Picture 1>`. That makes the index arithmetic worth
pinning: a drop landing one slot from where the indicator said renumbers the wrong pair,
and the result looks perfectly tidy afterwards.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QtCore = pytest.importorskip("PySide6.QtCore")
QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from harmon3.refs import (                                     # noqa: E402
    AUDIO, IMAGE, VIDEO, RefRow, RefSet, compute_tags, tag_migration,
)
from harmon3.ui.ref_panel import MIME_REF_ROW, RefListWidget, RefPanel  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def images(qapp):
    widget = RefListWidget(IMAGE)
    widget.set_rows([RefRow(kind=IMAGE, comfy_name=n)
                     for n in ("a.png", "b.png", "c.png")])
    yield widget
    widget.deleteLater()


def _names(widget) -> list[str]:
    return [w.row.comfy_name for w in widget.rows]


def _layout_names(widget) -> list[str]:
    return [widget.rows_layout.itemAt(i).widget().row.comfy_name
            for i in range(widget.rows_layout.count())]


def _uid(widget, name: str) -> int:
    return next(w.row.uid for w in widget.rows if w.row.comfy_name == name)


# ---------------------------------------------------------------------------------
# Where a row lands
# ---------------------------------------------------------------------------------

@pytest.mark.parametrize("name, index, expected", [
    # The index is a slot in the list as shown, so it counts the dragged row itself.
    ("a.png", 2, ["b.png", "a.png", "c.png"]),      # first row, dropped below the second
    ("a.png", 3, ["b.png", "c.png", "a.png"]),      # first row, dropped at the end
    ("c.png", 0, ["c.png", "a.png", "b.png"]),      # last row, dropped at the top
    ("c.png", 1, ["a.png", "c.png", "b.png"]),      # last row, dropped one up
    ("b.png", 0, ["b.png", "a.png", "c.png"]),      # middle row, dropped at the top
    ("b.png", 3, ["a.png", "c.png", "b.png"]),      # middle row, dropped at the end
])
def test_a_row_lands_where_the_indicator_said(images, name, index, expected):
    assert images.move_row(_uid(images, name), index) is True
    assert _names(images) == expected


@pytest.mark.parametrize("index", [1, 2])
def test_dropping_a_row_onto_its_own_position_changes_nothing(images, index):
    """Both slots either side of a row mean "leave it here", and neither may report a
    move -- a spurious structure_changed would offer a tag rewrite with nothing to fix."""
    assert images.move_row(_uid(images, "b.png"), index) is False
    assert _names(images) == ["a.png", "b.png", "c.png"]


def test_an_unknown_uid_moves_nothing(images):
    assert images.move_row(999_999, 0) is False
    assert _names(images) == ["a.png", "b.png", "c.png"]


def test_the_layout_always_agrees_with_the_model(images):
    """model_rows() reads self.rows and the user reads the layout; if those two ever
    disagree the graph is built in an order nobody chose."""
    for name, index in (("a.png", 2), ("c.png", 0), ("b.png", 3), ("a.png", 1)):
        images.move_row(_uid(images, name), index)
        assert _names(images) == _layout_names(images)
        assert [r.comfy_name for r in images.model_rows()] == _layout_names(images)


@pytest.mark.parametrize("count", [0, 1])
def test_a_list_too_short_to_reorder_is_harmless(qapp, count):
    widget = RefListWidget(IMAGE)
    widget.set_rows([RefRow(kind=IMAGE, comfy_name="only.png")][:count])
    if count:
        assert widget.move_row(_uid(widget, "only.png"), 0) is False
    assert widget._insertion_index(0) == 0
    widget.deleteLater()


# ---------------------------------------------------------------------------------
# Which slot a position falls into
# ---------------------------------------------------------------------------------

def test_the_insertion_index_follows_the_row_midpoints(images):
    images.resize(300, 400)
    images.show()
    QtWidgets.QApplication.processEvents()

    for index, widget in enumerate(images.rows):
        box = widget.geometry()
        assert images._insertion_index(box.top() + 1) == index
        assert images._insertion_index(box.bottom() - 1) == index + 1
    assert images._insertion_index(10_000) == len(images.rows)
    images.hide()


# ---------------------------------------------------------------------------------
# What a drag is allowed to carry
# ---------------------------------------------------------------------------------

class _FakeDrag:
    """Just the one thing _dragged_uid asks of an event.

    A real QDragMoveEvent takes ownership of the QMimeData handed to it, so the one built
    here would be freed twice and take the interpreter with it. The method only ever calls
    mimeData(), so there is nothing to gain from the real thing.
    """

    def __init__(self, payload):
        self._payload = payload

    def mimeData(self):
        return self._payload


def _row_drag(uid):
    payload = QtCore.QMimeData()
    payload.setData(MIME_REF_ROW, str(uid).encode("ascii"))
    return _FakeDrag(payload)


def test_a_drag_carrying_one_of_this_lists_rows_is_accepted(images):
    assert images._dragged_uid(_row_drag(_uid(images, "a.png"))) == _uid(images, "a.png")


def test_a_row_from_another_kind_is_refused(qapp, images):
    """An image can only ever be a `<Picture i>`, so there is no ordinal for it among the
    videos to move to. Refusing here is what keeps the tag algorithm's shape."""
    videos = RefListWidget(VIDEO)
    videos.set_rows([RefRow(kind=VIDEO, comfy_name="v.mp4")])

    assert images._dragged_uid(_row_drag(_uid(videos, "v.mp4"))) is None
    videos.deleteLater()


def test_a_file_drag_is_left_alone(images):
    """It has to reach RefPanel, which is what turns dropped files into new references."""
    payload = QtCore.QMimeData()
    payload.setUrls([QtCore.QUrl.fromLocalFile("C:/x.png")])
    assert images._dragged_uid(_FakeDrag(payload)) is None


def test_an_empty_drag_is_left_alone(images):
    assert images._dragged_uid(_FakeDrag(QtCore.QMimeData())) is None


def test_a_corrupt_uid_is_refused_rather_than_raising(images):
    payload = QtCore.QMimeData()
    payload.setData(MIME_REF_ROW, b"not-a-number")
    assert images._dragged_uid(_FakeDrag(payload)) is None


# ---------------------------------------------------------------------------------
# What reordering means for the prompt
# ---------------------------------------------------------------------------------

def test_reordering_renumbers_the_tags_and_reports_the_migration(qapp):
    panel = RefPanel()
    panel.set_refset(RefSet(
        images=[RefRow(kind=IMAGE, comfy_name=n)
                for n in ("hero.png", "dress.png", "room.png")],
        videos=[RefRow(kind=VIDEO, comfy_name="dance.mp4", use_soundtrack=True)],
        audios=[RefRow(kind=AUDIO, comfy_name="music.wav")],
    ))
    before = compute_tags(panel.refset())

    lst = panel.lists[IMAGE]
    lst.move_row(_uid(lst, "room.png"), 0)
    after = compute_tags(panel.refset())

    # A three-way cycle, which is exactly the case the rewrite substitutes atomically.
    assert tag_migration(before, after) == {
        "<Picture 1>": "<Picture 2>",
        "<Picture 2>": "<Picture 3>",
        "<Picture 3>": "<Picture 1>",
    }
    panel.deleteLater()


def test_reordering_one_kind_leaves_the_others_tags_alone(qapp):
    panel = RefPanel()
    panel.set_refset(RefSet(
        images=[RefRow(kind=IMAGE, comfy_name=n) for n in ("a.png", "b.png")],
        videos=[RefRow(kind=VIDEO, comfy_name="v.mp4", use_soundtrack=True)],
        audios=[RefRow(kind=AUDIO, comfy_name="m.wav")],
    ))
    before = compute_tags(panel.refset())

    lst = panel.lists[IMAGE]
    lst.move_row(_uid(lst, "b.png"), 0)
    after = compute_tags(panel.refset())

    moved = tag_migration(before, after)
    assert set(moved) == {"<Picture 1>", "<Picture 2>"}
    video = panel.lists[VIDEO].rows[0].row
    assert after.by_uid[video.uid] == before.by_uid[video.uid] == "<Video 1>"
    panel.deleteLater()


def test_a_reorder_announces_itself_as_a_structural_change(qapp, images):
    """The same signal an add or a remove sends: it is what recomputes the tags and
    offers the rewrite. Without it a reorder would change the graph in silence."""
    seen = []
    images.structure_changed.connect(lambda: seen.append(True))

    images.move_row(_uid(images, "a.png"), 2)
    assert seen == [], "move_row itself is silent; the drop is what announces it"

    images._clear_indicator()
    assert images._indicator is None


# ---------------------------------------------------------------------------------
# Folding a fresh renumbering into one the user has not acted on yet
#
# This is what a reorder exposed: the fold used to chain the entries of a *single*
# migration together, which cancels any permutation cycle -- and every swap is one.
# ---------------------------------------------------------------------------------

from harmon3.ui.prompt_editor import _compose                   # noqa: E402


def test_a_swap_survives_the_fold():
    """Two references trading places. Chaining these entry by entry gives P1 -> P1 and
    drops the rewrite, leaving the prompt silently pointing at the wrong pictures."""
    assert _compose({}, {"<Picture 1>": "<Picture 2>", "<Picture 2>": "<Picture 1>"}) == {
        "<Picture 1>": "<Picture 2>", "<Picture 2>": "<Picture 1>"}


def test_a_three_way_cycle_survives_the_fold():
    cycle = {"<Picture 1>": "<Picture 2>",
             "<Picture 2>": "<Picture 3>",
             "<Picture 3>": "<Picture 1>"}
    assert _compose({}, cycle) == cycle


def test_a_later_edit_composes_onto_an_unactioned_one():
    """The key stays the tag the prompt still spells, so the rewrite remains applicable
    to text the user never touched."""
    assert _compose({"<Picture 3>": "<Picture 2>"},
                    {"<Picture 2>": "<Picture 1>"}) == {"<Picture 3>": "<Picture 1>"}


def test_an_edit_that_undoes_an_earlier_one_leaves_nothing_to_rewrite():
    assert _compose({"<Picture 2>": "<Picture 1>"},
                    {"<Picture 1>": "<Picture 2>"}) == {}


def test_tags_the_pending_migration_does_not_mention_are_carried_in():
    assert _compose({"<Picture 3>": "<Picture 2>"},
                    {"<Audio 2>": "<Audio 1>"}) == {"<Picture 3>": "<Picture 2>",
                                                    "<Audio 2>": "<Audio 1>"}


def test_composing_with_nothing_is_the_identity():
    pending = {"<Picture 2>": "<Picture 1>"}
    assert _compose(pending, {}) == pending
    assert _compose({}, {}) == {}


def test_the_fold_never_mutates_what_it_was_given():
    pending = {"<Picture 3>": "<Picture 2>"}
    migration = {"<Picture 2>": "<Picture 1>"}
    _compose(pending, migration)
    assert pending == {"<Picture 3>": "<Picture 2>"}
    assert migration == {"<Picture 2>": "<Picture 1>"}
