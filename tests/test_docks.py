"""The workspace: six panes the user arranges, and the arrangement surviving a restart.

The panes are the whole app, so the invariants worth pinning are the ones whose failure is
silent. A dock with no object name is dropped from the saved state without a word, and the
layout appears to save and comes back empty. A saved state naming a pane that no longer
exists restores as a hole and reports success. Both are checked here rather than found in
the wild a version later.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("PySide6.QtWidgets")

from harmon3.ui import docks                                    # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    import os

    from PySide6.QtWidgets import QApplication

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QApplication.instance() or QApplication([])
    yield app


@pytest.fixture(scope="module")
def window(qapp, tmp_path_factory):
    """A real main window, with its layout file pointed away from the user's own."""
    from harmon3 import config
    from harmon3.ui.main_window import MainWindow

    home = tmp_path_factory.mktemp("ui-state")
    original = config.UI_STATE_PATH
    config.UI_STATE_PATH = home / "ui_state.ini"

    made = MainWindow("http://127.0.0.1:8188", "docks-test")
    made._save_settings = lambda: None          # never write the user's settings file
    yield made

    made._save_timer.stop()
    made.ws_client.stop()
    made.ws_thread.quit()
    made.ws_thread.wait(2000)
    made.job_thread.quit()
    made.job_thread.wait(2000)
    config.UI_STATE_PATH = original


def _settle(window) -> None:
    """Let Qt actually lay the panes out, so their geometry means something.

    The default sizes are applied from a zero timer -- before the window has been shown,
    resizeDocks clamps silently to minimum size hints that are not yet what they will be.
    """
    from PySide6.QtCore import QEventLoop, QTimer
    from PySide6.QtWidgets import QApplication

    window.show()
    loop = QEventLoop()
    QTimer.singleShot(50, loop.quit)
    loop.exec()
    QApplication.processEvents()


@pytest.fixture
def panes(window):
    """The six panes, put back where they belong before each test."""
    docks.apply_default_layout(window)
    _settle(window)
    return [getattr(window, attr) for attr in docks._DOCK_ATTRS]


# ---------------------------------------------------------------------------------
# Every pane is a pane, and Qt can tell them apart
# ---------------------------------------------------------------------------------

def test_there_are_six_panes(window, panes):
    from PySide6.QtWidgets import QDockWidget

    assert len(window.findChildren(QDockWidget)) == 6


def test_every_pane_has_a_unique_object_name(window, panes):
    """Without one, saveState drops the pane and the layout silently does not persist."""
    names = [dock.objectName() for dock in panes]
    assert all(names)
    assert sorted(names) == sorted(docks.DOCK_NAMES)


def test_the_toolbar_carrying_the_banner_is_named_too(window):
    assert window.banner_bar.objectName() == "toolbar.banner"


def test_panes_cannot_be_closed(window, panes):
    """Six panes are the whole app; a close button buys a way to lose one and no way back."""
    from PySide6.QtWidgets import QDockWidget

    for dock in panes:
        assert not (dock.features() & QDockWidget.DockWidgetClosable)
        assert dock.features() & QDockWidget.DockWidgetMovable
        assert dock.features() & QDockWidget.DockWidgetFloatable


def test_the_default_is_three_columns_of_two(window, panes):
    from PySide6.QtCore import Qt

    for dock in panes:
        assert dock.isFloating() is False
        assert not dock.isHidden()
        assert window.dockWidgetArea(dock) == Qt.LeftDockWidgetArea

    # Left to right and top to bottom: geometry rather than areas, because nesting puts
    # every pane in the same dock area.
    projects, references, viewer, prompts, run, other = panes
    assert projects.x() < viewer.x() < run.x()
    assert projects.y() < references.y()
    assert viewer.y() < prompts.y()
    assert run.y() < other.y()


def test_the_viewer_gets_the_middle_column(window, panes):
    """It is the widest pane: it is where the picture is."""
    projects, _references, viewer, _prompts, run, _other = panes
    assert viewer.width() > projects.width()
    assert viewer.width() > run.width()


# ---------------------------------------------------------------------------------
# The arrangement survives a restart, and a stale one is not restored
# ---------------------------------------------------------------------------------

def test_an_arrangement_round_trips(window, panes):
    _projects, _references, _viewer, prompts, _run, _other = panes
    state = window.saveState(docks.LAYOUT_VERSION)

    prompts.setFloating(True)
    assert window.restoreState(state, docks.LAYOUT_VERSION)
    assert prompts.isFloating() is False


def test_a_state_from_another_version_is_refused(window, panes):
    """The format changed, so the bytes mean something else now. Qt says so; we listen."""
    state = window.saveState(docks.LAYOUT_VERSION)
    assert window.restoreState(state, docks.LAYOUT_VERSION + 1) is False


def test_a_state_naming_panes_that_no_longer_exist_is_caught_by_the_schema(window, panes):
    """restoreState would accept it and leave a hole, which is why the schema is separate."""
    from PySide6.QtCore import QSettings

    from harmon3 import config

    store = QSettings(str(config.UI_STATE_PATH), QSettings.IniFormat)
    store.setValue("window/state", window.saveState(docks.LAYOUT_VERSION))
    store.setValue("layout/schema", "some-older-set-of-panes")
    store.sync()

    _projects, _references, _viewer, prompts, _run, _other = panes
    prompts.setFloating(True)
    window._restore_geometry()
    assert prompts.isFloating() is False


def test_a_layout_file_from_before_the_panes_falls_back_to_the_default(window, panes):
    """Older versions saved three splitter states and nothing else."""
    from PySide6.QtCore import QSettings

    from harmon3 import config

    store = QSettings(str(config.UI_STATE_PATH), QSettings.IniFormat)
    store.clear()
    store.setValue("splitter/editor", b"nonsense")
    store.setValue("splitter/main", b"nonsense")
    store.sync()

    window._restore_geometry()

    assert all(not dock.isFloating() for dock in panes)
    # The stale keys are cleared out on the way past rather than left to rot.
    assert QSettings(str(config.UI_STATE_PATH), QSettings.IniFormat).value(
        "splitter/editor") is None


def test_the_selected_tab_of_the_other_pane_is_remembered(window, panes):
    """saveState knows about tabbed panes, not about a tab widget inside one."""
    from PySide6.QtCore import QSettings

    from harmon3 import config

    window.other_tabs.setCurrentIndex(2)
    window._save_geometry()

    window.other_tabs.setCurrentIndex(0)
    window._restore_geometry()
    assert window.other_tabs.currentIndex() == 2

    QSettings(str(config.UI_STATE_PATH), QSettings.IniFormat).clear()


def test_a_notice_showing_at_shutdown_does_not_come_back_stale(window, panes):
    """restoreState brings toolbar visibility with it; the notice it described is gone."""
    window._show_banner("the server went away", error=True)
    window._save_geometry()

    window._restore_geometry()
    assert window.banner_bar.isVisible() is False


# ---------------------------------------------------------------------------------
# Getting back to the default
# ---------------------------------------------------------------------------------

def test_reset_recovers_a_pane_left_floating(window, panes):
    _projects, _references, viewer, _prompts, _run, _other = panes
    viewer.setFloating(True)

    window._on_reset_layout()

    assert viewer.isFloating() is False
    assert not viewer.isHidden()


def test_settings_offers_the_reset(window):
    """The panes cannot be closed, so there is no menu; this is the way back."""
    assert window.settings_panel.reset_layout_button.text() == "Reset layout"
