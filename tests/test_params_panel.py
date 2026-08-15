"""The Parameters panel: one group, nothing hidden, and a fixed dimension grid."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QtWidgets = pytest.importorskip("PySide6.QtWidgets")

from harmon3 import config, mathmirror, settings as settings_mod   # noqa: E402
from harmon3.graph_builder import BuildState                        # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    import os

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


@pytest.fixture
def panel(qapp):
    from harmon3.ui.params_panel import ParamsPanel

    widget = ParamsPanel()
    yield widget
    widget.deleteLater()


# ---------------------------------------------------------------------------------
# What the panel offers
# ---------------------------------------------------------------------------------

def test_everything_lives_under_one_heading(panel):
    groups = panel.findChildren(QtWidgets.QGroupBox)
    assert [group.title() for group in groups] == ["Parameters"]


def test_nothing_is_folded_away(panel):
    """The seed used to sit inside a collapsed group; a checkable group is that shape."""
    for group in panel.findChildren(QtWidgets.QGroupBox):
        assert group.isCheckable() is False


def test_the_seed_controls_are_present_without_expanding_anything(panel):
    assert panel.seed_spin.isHidden() is False
    assert panel.dice_button.isHidden() is False
    assert panel.randomize_box.isHidden() is False


def test_the_dimension_grid_is_no_longer_a_control(panel):
    assert not hasattr(panel, "multiple_spin")


def test_steps_and_reference_sizing_are_controls(panel):
    assert panel.steps_spin.minimum() == config.MIN_STEPS
    assert panel.steps_spin.maximum() == config.MAX_STEPS
    assert [panel.ref_size_combo.itemText(i)
            for i in range(panel.ref_size_combo.count())] == list(config.REF_IMAGE_SIZES)


def test_the_sampler_and_scheduler_start_from_stock_comfyui(panel):
    """What they offer before there is a server to ask."""
    assert [panel.sampler_combo.itemText(i)
            for i in range(panel.sampler_combo.count())] == list(config.SAMPLERS)
    assert [panel.scheduler_combo.itemText(i)
            for i in range(panel.scheduler_combo.count())] == list(config.SCHEDULERS)


def test_the_sampler_sits_above_the_scheduler(panel):
    """The order they act in: the solver first, then how its sigmas are spaced."""
    form = panel.findChild(QtWidgets.QFormLayout)
    rows = [form.itemAt(i, QtWidgets.QFormLayout.FieldRole) for i in range(form.rowCount())]
    widgets = [item.widget() for item in rows if item is not None]
    assert widgets.index(panel.sampler_combo) < widgets.index(panel.scheduler_combo)


def test_the_server_replaces_the_built_in_lists(panel):
    """A machine with sampler packs installed offers more than this build knew about."""
    panel.set_server_options({
        "KSamplerSelect": {"input": {"required": {
            "sampler_name": ["COMBO", {"options": ["euler", "res_multistep", "res_2m_ode"]}]}}},
        "BasicScheduler": {"input": {"required": {
            "scheduler": [["simple", "karras", "beta_57"]]}}},
    })
    assert [panel.sampler_combo.itemText(i)
            for i in range(panel.sampler_combo.count())] == [
        "euler", "res_multistep", "res_2m_ode"]
    assert "beta_57" in [panel.scheduler_combo.itemText(i)
                         for i in range(panel.scheduler_combo.count())]


def test_the_choice_survives_the_server_replacing_the_list(panel):
    panel.load_state(BuildState(sampler_name="euler"), randomize=False)
    panel.set_server_options({
        "KSamplerSelect": {"input": {"required": {
            "sampler_name": ["COMBO", {"options": ["euler", "res_multistep"]}]}}},
    })
    assert panel.sampler_combo.currentText() == "euler"


def test_a_sampler_the_list_has_never_heard_of_is_still_shown(panel):
    """From a settings file or a restored run: it must not silently revert."""
    panel.load_state(BuildState(sampler_name="res_2m_ode"), randomize=False)
    assert panel.sampler_combo.currentText() == "res_2m_ode"

    written = BuildState()
    panel.apply_to_state(written)
    assert written.sampler_name == "res_2m_ode"


def test_an_unreachable_server_leaves_the_lists_alone(panel):
    """No schemas yet is not the same as "this server has no samplers"."""
    before = [panel.sampler_combo.itemText(i) for i in range(panel.sampler_combo.count())]
    panel.set_server_options({})
    panel.set_server_options({"KSamplerSelect": None})
    assert [panel.sampler_combo.itemText(i)
            for i in range(panel.sampler_combo.count())] == before


def test_the_schedule_is_free_text(panel):
    """The node parses it, and the useful schedules are a continuum rather than a menu."""
    assert isinstance(panel.schedule_edit, QtWidgets.QLineEdit)
    assert panel.schedule_edit.isReadOnly() is False


def test_a_schedule_the_sampler_would_refuse_is_flagged_before_queueing(panel):
    panel.load_state(BuildState(schedule="0.5:0.55"), randomize=False)
    assert panel.schedule_note.isHidden() is False
    assert "last stage" in panel.schedule_note.text()


def test_a_workable_schedule_costs_no_line(panel):
    panel.load_state(BuildState(schedule=config.DEFAULT_SCHEDULE), randomize=False)
    assert panel.schedule_note.isHidden() is True


def test_the_schedule_field_is_hidden_when_the_workflow_has_no_sampler_for_it(panel, qapp):
    """A SamplerCustomAdvanced workflow declares no 'schedule' input, so offering the
    field would mean a control that silently does nothing."""
    from harmon3 import roles as roles_mod

    without = roles_mod.NodeRoles(single={}, multi={}, keep=(), injected={})
    panel.show()
    panel.set_workflow_features(without)
    assert panel.schedule_edit.isVisible() is False
    assert panel.schedule_label.isVisible() is False
    # And a schedule that would otherwise warn raises no note, because it is not in play.
    panel.load_state(BuildState(schedule="0.5:0.55"), randomize=False)
    assert panel.schedule_note.isHidden() is True


def test_the_schedule_field_is_shown_for_a_workflow_that_has_one(panel, qapp):
    workflow = config.load_workflow()
    panel.show()
    panel.set_workflow_features(workflow.roles)
    assert panel.schedule_edit.isVisible() is True


def test_the_stage_upscale_and_sigma_shift_are_shown_for_this_workflow(panel, qapp):
    workflow = config.load_workflow()
    panel.show()
    panel.set_workflow_features(workflow.roles)
    assert panel.upscale_combo.isVisible() is True
    assert panel.shift_spin.isVisible() is True


def test_the_stage_upscale_and_sigma_shift_hide_without_a_node_for_them(panel, qapp):
    """Both write inputs that only some nodes declare, and an undeclared input makes
    ComfyUI reject the whole prompt rather than ignore the key."""
    from harmon3 import roles as roles_mod

    without = roles_mod.NodeRoles(single={}, multi={}, keep=(), injected={})
    panel.show()
    panel.set_workflow_features(without)
    assert panel.upscale_combo.isVisible() is False
    assert panel.upscale_label.isVisible() is False
    assert panel.shift_spin.isVisible() is False
    assert panel.shift_label.isVisible() is False


def test_the_upscale_list_is_the_nodes_own_fixed_set(panel):
    """Unlike sampler and scheduler, this combo is not extended by plugin packs, so it is
    never repopulated from the server."""
    assert [panel.upscale_combo.itemText(i)
            for i in range(panel.upscale_combo.count())] == list(config.UPSCALE_METHODS)


# ---------------------------------------------------------------------------------
# Values in and out
# ---------------------------------------------------------------------------------

def test_the_panel_round_trips_a_state(panel):
    state = BuildState(
        aspect_ratio="1:1 (Square)", megapixels=1.2, duration_seconds=3.0,
        seed=99, steps=35, sampler_name="dpmpp_2m", scheduler="karras",
        upscale_method="bislerp", shift_video=6.5, ref_image_size="max",
    )
    panel.load_state(state, randomize=False)

    written = BuildState()
    panel.apply_to_state(written)
    assert (written.steps, written.ref_image_size) == (35, "max")
    assert (written.sampler_name, written.scheduler) == ("dpmpp_2m", "karras")
    assert (written.upscale_method, written.shift_video) == ("bislerp", 6.5)
    assert (written.aspect_ratio, written.megapixels, written.seed) == (
        "1:1 (Square)", 1.2, 99)


def test_a_state_from_outside_its_range_is_brought_back_in(panel):
    """Loading must not throw, and must not leave the widget disagreeing with the state."""
    panel.load_state(
        BuildState(steps=10_000_000, ref_image_size="nonsense", scheduler="",
                   sampler_name="", upscale_method="lanczos", shift_video=-3),
        randomize=True)

    written = BuildState()
    panel.apply_to_state(written)
    assert written.steps == config.MAX_STEPS
    assert written.ref_image_size == config.DEFAULT_REF_IMAGE_SIZE
    assert written.sampler_name == config.DEFAULT_SAMPLER
    assert written.scheduler == config.DEFAULT_SCHEDULER
    assert written.upscale_method == config.DEFAULT_UPSCALE_METHOD
    assert written.shift_video == config.MIN_SHIFT


def test_the_resolution_readout_uses_the_fixed_grid(panel):
    panel.load_state(BuildState(aspect_ratio="16:9 (Widescreen)", megapixels=0.4),
                     randomize=True)
    width, height = panel.resolution()

    assert (width, height) == mathmirror.resolution("16:9 (Widescreen)", 0.4, config.MULTIPLE)
    assert width % config.MULTIPLE == 0 and height % config.MULTIPLE == 0


def test_editing_steps_announces_a_change(panel):
    seen = []
    panel.changed.connect(lambda: seen.append(panel.steps_spin.value()))
    panel.steps_spin.setValue(31)
    assert seen == [31]


def test_editing_the_reference_size_announces_a_change(panel):
    seen = []
    panel.changed.connect(lambda: seen.append(panel.ref_size_combo.currentText()))
    panel.ref_size_combo.setCurrentText("max")
    assert seen == ["max"]


# ---------------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------------

def test_both_new_settings_survive_a_save_and_reload(tmp_path):
    path = tmp_path / "settings.json"
    state = BuildState(steps=33, ref_image_size="max")

    settings_mod.save_settings(
        settings_mod.capture_from_state(settings_mod.load_settings(path), state), path)

    restored = BuildState()
    settings_mod.apply_to_state(restored, settings_mod.load_settings(path))
    assert (restored.steps, restored.ref_image_size) == (33, "max")


def test_the_sampler_settings_survive_a_save_and_reload(tmp_path):
    path = tmp_path / "settings.json"
    state = BuildState(sampler_name="dpmpp_2m", scheduler="beta",
                       upscale_method="area", shift_video=8.25)

    settings_mod.save_settings(
        settings_mod.capture_from_state(settings_mod.load_settings(path), state), path)

    restored = BuildState()
    settings_mod.apply_to_state(restored, settings_mod.load_settings(path))
    assert restored.sampler_name == "dpmpp_2m"
    assert restored.scheduler == "beta"
    assert restored.upscale_method == "area"
    assert restored.shift_video == 8.25


def test_a_settings_file_from_before_the_sampler_was_exposed_still_loads(tmp_path):
    """It names neither, so both fall back to what the shipped workflow uses."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"steps": 24, "megapixels": 0.8}), encoding="utf-8")

    state = BuildState()
    settings_mod.apply_to_state(state, settings_mod.load_settings(path))
    assert state.steps == 24
    assert state.sampler_name == config.DEFAULT_SAMPLER
    assert state.scheduler == config.DEFAULT_SCHEDULER


def test_a_settings_file_from_before_the_grid_was_fixed_still_loads(tmp_path):
    """It carries a 'multiple' the app no longer has anywhere to put."""
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"multiple": 64, "megapixels": 0.8}), encoding="utf-8")

    data = settings_mod.load_settings(path)
    assert "multiple" not in data
    assert data["megapixels"] == 0.8

    state = BuildState()
    settings_mod.apply_to_state(state, data)
    assert not hasattr(state, "multiple")
