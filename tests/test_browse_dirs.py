"""The + Add dialogs remember where you were, separately for each kind of reference."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from harmon3 import settings as settings_mod                    # noqa: E402
from harmon3.refs import AUDIO, IMAGE, VIDEO                    # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def panel(qapp):
    from harmon3.ui.ref_panel import RefPanel
    widget = RefPanel()
    yield widget
    widget.deleteLater()


@pytest.fixture
def folders(tmp_path):
    made = {}
    for kind, name in ((IMAGE, "stills"), (VIDEO, "clips"), (AUDIO, "sound")):
        folder = tmp_path / name
        folder.mkdir()
        suffix = {IMAGE: ".png", VIDEO: ".mp4", AUDIO: ".wav"}[kind]
        target = folder / f"ref{suffix}"
        target.write_bytes(b"x")
        made[kind] = target
    return made


def test_each_kind_remembers_its_own_folder(panel, folders):
    for kind, file in folders.items():
        panel.lists[kind].add_paths([str(file)])

    remembered = panel.last_dirs()
    assert remembered[IMAGE] == str(folders[IMAGE].parent)
    assert remembered[VIDEO] == str(folders[VIDEO].parent)
    assert remembered[AUDIO] == str(folders[AUDIO].parent)
    assert len({remembered[IMAGE], remembered[VIDEO], remembered[AUDIO]}) == 3


def test_adding_one_kind_leaves_the_others_alone(panel, folders):
    panel.lists[VIDEO].add_paths([str(folders[VIDEO])])
    remembered = panel.last_dirs()
    assert remembered == {VIDEO: str(folders[VIDEO].parent)}


def test_the_dialog_opens_where_that_kind_left_off(panel, folders):
    panel.lists[IMAGE].add_paths([str(folders[IMAGE])])
    assert panel.lists[IMAGE]._start_dir() == str(folders[IMAGE].parent)
    # Untouched kinds have nothing remembered and let Qt choose.
    assert panel.lists[VIDEO]._start_dir() == ""


def test_a_folder_that_has_since_been_removed_is_not_used(panel, folders, tmp_path):
    gone = tmp_path / "deleted"
    gone.mkdir()
    panel.lists[IMAGE].set_last_dir(str(gone))
    assert panel.lists[IMAGE]._start_dir() == str(gone)

    gone.rmdir()
    assert panel.lists[IMAGE]._start_dir() == ""


def test_only_the_first_of_a_multi_select_sets_the_folder(panel, tmp_path):
    first = tmp_path / "a"
    second = tmp_path / "b"
    for folder in (first, second):
        folder.mkdir()
        (folder / "x.png").write_bytes(b"x")

    panel.lists[IMAGE].add_paths([str(first / "x.png"), str(second / "x.png")])
    assert panel.last_dirs()[IMAGE] == str(first)


def test_dropped_files_also_set_the_folder(panel, folders):
    """Drag and drop routes through add_paths, so it feeds the same memory."""
    panel.lists[AUDIO].add_paths([str(folders[AUDIO])])
    assert panel.last_dirs()[AUDIO] == str(folders[AUDIO].parent)


def test_paths_over_the_limit_do_not_change_the_folder(panel, tmp_path, folders):
    """add_paths stops at the model maximum; a rejected file must not be remembered."""
    from harmon3.refs import KIND_LIMITS

    full = tmp_path / "full"
    full.mkdir()
    accepted = [full / f"v{i}.mp4" for i in range(KIND_LIMITS[VIDEO])]
    for path in accepted:
        path.write_bytes(b"x")
    panel.lists[VIDEO].add_paths([str(p) for p in accepted])
    assert panel.last_dirs()[VIDEO] == str(full)

    later = tmp_path / "later"
    later.mkdir()
    overflow = later / "extra.mp4"
    overflow.write_bytes(b"x")
    panel.lists[VIDEO].add_paths([str(overflow)])

    assert len(panel.lists[VIDEO].rows) == KIND_LIMITS[VIDEO]
    assert panel.last_dirs()[VIDEO] == str(full)


def test_round_trip_through_settings(panel, folders):
    for kind, file in folders.items():
        panel.lists[kind].add_paths([str(file)])
    saved = panel.last_dirs()

    from harmon3.ui.ref_panel import RefPanel
    restored = RefPanel()
    try:
        restored.set_last_dirs(saved)
        assert restored.last_dirs() == saved
        assert restored.lists[VIDEO]._start_dir() == str(folders[VIDEO].parent)
    finally:
        restored.deleteLater()


def test_set_last_dirs_tolerates_missing_and_none(panel):
    panel.set_last_dirs(None)
    assert panel.last_dirs() == {}
    panel.set_last_dirs({IMAGE: "/somewhere"})
    assert panel.last_dirs() == {IMAGE: "/somewhere"}


def test_replace_button_prefers_the_rows_own_folder(panel, folders, tmp_path):
    """The "..." dialog should open next to the file being replaced, not the last add."""
    other = tmp_path / "elsewhere"
    other.mkdir()
    panel.lists[IMAGE].set_last_dir(str(other))
    panel.lists[IMAGE].add_paths([str(folders[IMAGE])])

    row = panel.lists[IMAGE].row_widgets()[0]
    assert row.current_dir() == str(folders[IMAGE].parent)


def test_replace_falls_back_to_the_kinds_folder_for_a_server_row(panel, folders):
    from harmon3.refs import RefRow

    panel.lists[IMAGE].set_last_dir(str(folders[IMAGE].parent))
    panel.lists[IMAGE].set_rows([RefRow(kind=IMAGE, comfy_name="on_server.png")])
    row = panel.lists[IMAGE].row_widgets()[0]

    assert row.current_dir() == ""
    assert panel.lists[IMAGE]._start_dir() == str(folders[IMAGE].parent)


# ---------------------------------------------------------------------------------
# Settings persistence and migration
# ---------------------------------------------------------------------------------

def test_settings_round_trip(tmp_path):
    target = tmp_path / "settings.json"
    data = dict(settings_mod.DEFAULTS)
    data["last_browse_dirs"] = {IMAGE: "C:/stills", VIDEO: "C:/clips"}
    settings_mod.save_settings(data, target)

    assert settings_mod.load_settings(target)["last_browse_dirs"] == {
        IMAGE: "C:/stills", VIDEO: "C:/clips",
    }


def test_the_old_single_folder_setting_is_migrated(tmp_path):
    """An existing install stored one folder for everything; keep it rather than lose it."""
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({"last_browse_dir": "C:/refs"}), encoding="utf-8")

    loaded = settings_mod.load_settings(target)
    assert loaded["last_browse_dirs"] == {
        "image": "C:/refs", "video": "C:/refs", "audio": "C:/refs",
    }


def test_migration_does_not_override_per_kind_values(tmp_path):
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({
        "last_browse_dir": "C:/old",
        "last_browse_dirs": {"image": "C:/new"},
    }), encoding="utf-8")

    assert settings_mod.load_settings(target)["last_browse_dirs"] == {"image": "C:/new"}


def test_a_corrupt_value_falls_back_to_empty(tmp_path):
    target = tmp_path / "settings.json"
    target.write_text(json.dumps({"last_browse_dirs": "not a dict"}), encoding="utf-8")
    assert settings_mod.load_settings(target)["last_browse_dirs"] == {}


def test_defaults_have_no_remembered_folders():
    assert settings_mod.load_settings(Path("does-not-exist.json"))["last_browse_dirs"] == {}
