"""The generation parameters: everything about a run that is not the prompt or a reference.

Every field mirrors a widget in the workflow. The readouts show what the server will
actually compute, so the two numbers that really matter -- the pixel dimensions and the
frame count -- are visible before anything is queued.

They are all in one panel and none of them are hidden behind a disclosure. Seed in
particular is not an advanced setting: it is the difference between two runs of the same
scene, and it needs to be readable at a glance rather than a click away.
"""

from __future__ import annotations

import secrets

from PySide6.QtCore import QRegularExpression, Qt, Signal
from PySide6.QtGui import QRegularExpressionValidator
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import config, graph_builder, mathmirror
from ..mathmirror import ASPECT_RATIOS
from . import style


def _select(combo: "QComboBox", value: str) -> None:
    """Show ``value`` in ``combo``, adding it if the list has never heard of it.

    Without the add, a sampler this build does not list -- from a settings file, a restored
    run, or a server with packs installed -- would silently leave the combo on whatever was
    already selected, and the next edit would write *that* back over the saved choice.
    """
    index = combo.findText(value)
    if index < 0:
        combo.addItem(value)
        index = combo.count() - 1
    combo.setCurrentIndex(index)


def _repopulate(combo: "QComboBox", options: list) -> None:
    """Replace a combo's list with the server's, keeping what is selected selected."""
    current = combo.currentText()
    if not options or [combo.itemText(i) for i in range(combo.count())] == list(options):
        return
    blocked = combo.blockSignals(True)
    try:
        combo.clear()
        combo.addItems([str(option) for option in options])
        _select(combo, current)
    finally:
        combo.blockSignals(blocked)


class SeedEdit(QLineEdit):
    """A 64-bit-capable seed field.

    QSpinBox is limited to 32-bit signed values, which cannot hold the workflow's own
    seed (157368968253448), so the field is a validated line edit instead.
    """

    valueChanged = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setValidator(QRegularExpressionValidator(QRegularExpression(r"\d{0,17}"), self))
        self.setPlaceholderText("0")
        self.textEdited.connect(lambda _t: self.valueChanged.emit())
        self.setValue(0)

    def value(self) -> int:
        try:
            return min(int(self.text() or 0), config.MAX_SEED)
        except ValueError:
            return 0

    def setValue(self, value: int) -> None:
        clamped = max(0, min(int(value), config.MAX_SEED))
        if clamped != self.value() or not self.text():
            self.setText(str(clamped))


class ParamsPanel(QWidget):
    changed = Signal()
    check_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._suppress = False
        #: True until told otherwise, so the field is visible for the workflow that has
        #: one and a caller that never calls set_workflow_features loses nothing.
        self._has_schedule = True

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        layout.addWidget(self._build_parameters_group())

    # -- construction --------------------------------------------------------------

    def _build_parameters_group(self) -> QGroupBox:
        group = QGroupBox("Parameters")
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignRight)

        self.aspect_combo = QComboBox()
        self.aspect_combo.addItems(list(ASPECT_RATIOS))
        self.aspect_combo.currentTextChanged.connect(self._on_changed)
        form.addRow("Aspect ratio", self.aspect_combo)

        self.megapixels_spin = QDoubleSpinBox()
        self.megapixels_spin.setRange(config.MIN_MEGAPIXELS, config.MAX_MEGAPIXELS)
        self.megapixels_spin.setSingleStep(config.STEP_MEGAPIXELS)
        self.megapixels_spin.setDecimals(1)
        self.megapixels_spin.setSuffix(" MP")
        self.megapixels_spin.setToolTip(
            "Target pixel count. The exact width and height are derived from this and the\n"
            f"aspect ratio, then snapped to multiples of {config.MULTIPLE}."
        )
        self.megapixels_spin.valueChanged.connect(self._on_changed)
        form.addRow("Megapixels", self.megapixels_spin)

        self.resolution_label = style.metric()
        form.addRow("Output", self.resolution_label)

        self.duration_spin = QDoubleSpinBox()
        self.duration_spin.setRange(config.MIN_DURATION, config.MAX_DURATION)
        self.duration_spin.setSingleStep(0.5)
        self.duration_spin.setDecimals(2)
        self.duration_spin.setSuffix(" s")
        self.duration_spin.setToolTip(
            "Requested length. The model only accepts frame counts of the form 17k+5 at\n"
            "24 fps, so the actual clip is rounded up to the next valid length."
        )
        self.duration_spin.valueChanged.connect(self._on_changed)
        form.addRow("Length", self.duration_spin)

        self.frames_label = style.metric()
        form.addRow("Actual", self.frames_label)

        form.addRow(style.separator())

        self.steps_spin = QSpinBox()
        self.steps_spin.setRange(config.MIN_STEPS, config.MAX_STEPS)
        self.steps_spin.setToolTip(
            "Sampling steps. Run time is very nearly proportional to this, and the\n"
            "workflow's 20 is what the model was tuned around."
        )
        self.steps_spin.valueChanged.connect(self._on_changed)
        form.addRow("Steps", self.steps_spin)

        self.sampler_combo = QComboBox()
        self.sampler_combo.addItems(list(config.SAMPLERS))
        self.sampler_combo.setToolTip(
            "Which solver walks the sigmas. The workflow's 'res_multistep' is what MiniMax\n"
            "H3 was tuned around.\n\n"
            "This list is stock ComfyUI's until the server has been read; once connected it\n"
            "becomes whatever that server actually offers."
        )
        self.sampler_combo.currentTextChanged.connect(self._on_changed)
        form.addRow("Sampler", self.sampler_combo)

        self.scheduler_combo = QComboBox()
        self.scheduler_combo.addItems(list(config.SCHEDULERS))
        self.scheduler_combo.setToolTip(
            "How the sigmas are spaced across those steps. The workflow's 'simple' is what\n"
            "MiniMax H3 was tuned around; the rest are ComfyUI's standard set."
        )
        self.scheduler_combo.currentTextChanged.connect(self._on_changed)
        form.addRow("Scheduler", self.scheduler_combo)

        self.schedule_edit = QLineEdit()
        self.schedule_edit.setPlaceholderText(config.DEFAULT_SCHEDULE)
        style.mono(self.schedule_edit, size=9)
        self.schedule_edit.setToolTip(
            "Progressive staging, as 'scale:end_percent' per stage.\n\n"
            "scale is the spatial latent grid for that stage, end_percent is where it ends\n"
            "as a fraction of the steps. Early steps run on a smaller grid and the estimate\n"
            "is upscaled between stages, so most of the run is cheap and only the last stage\n"
            "pays full resolution. The last stage must be 1.0:1.0.\n\n"
            f"{config.BASELINE_SCHEDULE} - one full-resolution stage, the A/B baseline.\n"
            f"{config.DEFAULT_SCHEDULE} - half grid for the first 55% of the steps."
        )
        self.schedule_edit.textChanged.connect(self._on_changed)
        #: Kept so the whole row can be hidden: a workflow whose sampler takes no schedule
        #: must not show a field that would write an input no node declares.
        self.schedule_label = QLabel("Schedule")
        form.addRow(self.schedule_label, self.schedule_edit)

        #: Says why the sampler would refuse the schedule, before Queue is pressed. Takes
        #: up no room while the schedule is fine, which is nearly always.
        self.schedule_note = style.hint("")
        self.schedule_note.setWordWrap(True)
        self.schedule_note.hide()
        form.addRow("", self.schedule_note)

        self.upscale_combo = QComboBox()
        self.upscale_combo.addItems(list(config.UPSCALE_METHODS))
        self.upscale_combo.setToolTip(
            "How the estimate is resampled when one staging stage hands over to the next\n"
            "at a larger latent grid.\n\n"
            "Only does anything with a multi-stage schedule: at 1.0:1.0 nothing is ever\n"
            "upscaled. bicubic is the node's own default."
        )
        self.upscale_combo.currentTextChanged.connect(self._on_changed)
        self.upscale_label = QLabel("Stage upscale")
        form.addRow(self.upscale_label, self.upscale_combo)

        self.shift_spin = QDoubleSpinBox()
        self.shift_spin.setRange(config.MIN_SHIFT, config.MAX_SHIFT)
        self.shift_spin.setSingleStep(config.STEP_SHIFT)
        self.shift_spin.setDecimals(2)
        self.shift_spin.setToolTip(
            "MiniMaxH3SigmaShift.shift_video: where the weight of the sigma schedule sits.\n\n"
            "Higher spends more of the run at high noise, which moves the picture further\n"
            "from the references; lower holds closer to them. The workflow's value is what\n"
            "the model was tuned around.\n\n"
            "shift_audio is not exposed and stays as the workflow sets it."
        )
        self.shift_spin.valueChanged.connect(self._on_changed)
        self.shift_label = QLabel("Sigma shift")
        form.addRow(self.shift_label, self.shift_spin)

        self.ref_size_combo = QComboBox()
        self.ref_size_combo.addItems(list(config.REF_IMAGE_SIZES))
        self.ref_size_combo.setToolTip(
            "ref_image_size: how reference images are sized before they are encoded.\n\n"
            "match - scale each one down to the generation's own pixel area.\n"
            "max - use the reference pipeline's 2048px short edge, for the best identity\n"
            "fidelity. Reference tokens ride through every sampling step, so this can be\n"
            "several times slower."
        )
        self.ref_size_combo.currentTextChanged.connect(self._on_changed)
        form.addRow("Reference size", self.ref_size_combo)

        form.addRow(style.separator())

        seed_row = QHBoxLayout()
        seed_row.setContentsMargins(0, 0, 0, 0)

        self.seed_spin = SeedEdit()
        style.mono(self.seed_spin, size=10)
        self.seed_spin.setToolTip(f"Noise seed, 0 to {config.MAX_SEED}")
        self.seed_spin.valueChanged.connect(self._on_changed)
        seed_row.addWidget(self.seed_spin, 1)

        self.dice_button = QToolButton()
        self.dice_button.setText("\U0001F3B2")
        self.dice_button.setToolTip("Roll a new seed now")
        self.dice_button.clicked.connect(self.roll_seed)
        seed_row.addWidget(self.dice_button)

        seed_container = QWidget()
        seed_container.setLayout(seed_row)
        form.addRow("Seed", seed_container)

        self.randomize_box = QCheckBox("New seed for every run")
        self.randomize_box.setChecked(True)
        self.randomize_box.toggled.connect(self._on_randomize_toggled)
        form.addRow("", self.randomize_box)

        form.addRow(style.separator())

        self.check_button = QPushButton("Check the workflow")
        self.check_button.setToolTip(
            "Confirm the workflow still satisfies the role contract, that every node it\n"
            "uses exists on this server, and that nothing needed would be pruned.\n"
            "Loads no models, so it takes about a second."
        )
        self.check_button.clicked.connect(self.check_requested.emit)
        form.addRow("", self.check_button)

        self.fixed_note = style.hint(
            f"Width and height are snapped to {config.MULTIPLE}px and the length to the "
            f"17k+5 frame grid, here rather than in the workflow, at {config.FPS} fps."
        )
        self.fixed_note.setWordWrap(True)
        form.addRow("", self.fixed_note)

        return group

    # -- values --------------------------------------------------------------------

    def load_state(self, state, randomize: bool) -> None:
        self._suppress = True
        try:
            index = self.aspect_combo.findText(state.aspect_ratio)
            self.aspect_combo.setCurrentIndex(index if index >= 0 else 0)
            self.megapixels_spin.setValue(state.megapixels)
            self.duration_spin.setValue(mathmirror.clamp_duration(state.duration_seconds))
            self.seed_spin.setValue(min(int(state.seed), config.MAX_SEED))
            self.randomize_box.setChecked(randomize)
            self.steps_spin.setValue(graph_builder.clamp_steps(state.steps))
            _select(self.sampler_combo, graph_builder.clean_sampler(state.sampler_name))
            _select(self.scheduler_combo, graph_builder.clean_scheduler(state.scheduler))
            self.schedule_edit.setText(graph_builder.clean_schedule(state.schedule))
            _select(self.upscale_combo,
                    graph_builder.clean_upscale_method(state.upscale_method))
            self.shift_spin.setValue(graph_builder.clamp_shift(state.shift_video))
            self.ref_size_combo.setCurrentText(
                graph_builder.clean_ref_image_size(state.ref_image_size))
        finally:
            self._suppress = False
        self.refresh_readouts()

    def apply_to_state(self, state) -> None:
        state.aspect_ratio = self.aspect_combo.currentText()
        state.megapixels = self.megapixels_spin.value()
        state.duration_seconds = self.duration_spin.value()
        state.seed = self.seed_spin.value()
        state.steps = self.steps_spin.value()
        state.sampler_name = self.sampler_combo.currentText()
        state.scheduler = self.scheduler_combo.currentText()
        state.schedule = self.schedule_edit.text()
        state.upscale_method = self.upscale_combo.currentText()
        state.shift_video = self.shift_spin.value()
        state.ref_image_size = self.ref_size_combo.currentText()

    def set_workflow_features(self, roles) -> None:
        """Show only the parameters the loaded workflow actually has a node for.

        Three of them vary. MiniMaxH3ProgressiveSampler declares the staging schedule and
        the stage upscale method; SamplerCustomAdvanced declares neither. The sigma shift
        needs an h3-shift node. Offering any of them regardless would mean a control that
        silently does nothing -- and worse, writes an input no node declares, which makes
        ComfyUI reject the whole prompt rather than ignore the key.
        """
        self._has_schedule = graph_builder.schedule_node(roles) is not None
        for widgets, present in (
            ((self.schedule_label, self.schedule_edit), self._has_schedule),
            ((self.upscale_label, self.upscale_combo),
             graph_builder.upscale_node(roles) is not None),
            ((self.shift_label, self.shift_spin),
             graph_builder.shift_node(roles) is not None),
        ):
            for widget in widgets:
                widget.setVisible(present)
        self.refresh_readouts()

    def set_server_options(self, object_info: dict) -> None:
        """Replace the sampler and scheduler lists with the ones this server offers.

        The built-in lists are stock ComfyUI's, which is right until there is a server to
        ask; a machine with sampler packs installed has more, and those are the names its
        own graphs will need.
        """
        from ..validator import combo_options

        _repopulate(self.sampler_combo,
                    combo_options(object_info, "KSamplerSelect", "sampler_name"))
        _repopulate(self.scheduler_combo,
                    combo_options(object_info, "BasicScheduler", "scheduler"))

    @property
    def randomize(self) -> bool:
        return self.randomize_box.isChecked()

    def roll_seed(self) -> None:
        self.seed_spin.setValue(secrets.randbelow(config.MAX_SEED + 1))

    def resolution(self) -> tuple[int, int]:
        return mathmirror.resolution(
            self.aspect_combo.currentText(), self.megapixels_spin.value())

    def frames(self) -> int:
        return mathmirror.frames_from_seconds(self.duration_spin.value())

    def duration_seconds(self) -> float:
        return float(self.duration_spin.value())

    def problems(self) -> list[str]:
        found = []
        width, height = self.resolution()
        error = mathmirror.resolution_error(width, height)
        if error:
            found.append(error)
        error = mathmirror.duration_error(self.duration_spin.value())
        if error:
            found.append(error)
        return found

    # -- readouts ------------------------------------------------------------------

    def refresh_readouts(self) -> None:
        width, height = self.resolution()
        resolution_error = mathmirror.resolution_error(width, height)
        self.resolution_label.setText(f"{width} × {height}")
        # role2, not role: these labels keep their "metric" role so the monospaced
        # treatment survives an error state.
        self.resolution_label.setProperty("role2", "error" if resolution_error else "")
        self.resolution_label.setToolTip(resolution_error or "")
        style.restyle(self.resolution_label)

        frames = self.frames()
        seconds = mathmirror.true_seconds(frames)
        duration_error = mathmirror.duration_error(self.duration_spin.value())
        self.frames_label.setText(f"{frames}f · {seconds:.2f}s @ {config.FPS}fps")
        self.frames_label.setProperty("role2", "error" if duration_error else "")
        self.frames_label.setToolTip(duration_error or "")
        style.restyle(self.frames_label)

        # A warning rather than a block: the node owns the grammar and its own message is
        # the authority, but there is no reason to find out several minutes into a run.
        schedule_error = (
            graph_builder.schedule_error(self.schedule_edit.text())
            if self._has_schedule else "")
        self.schedule_note.setText(
            f"{schedule_error} The sampler will refuse this." if schedule_error else "")
        self.schedule_note.setProperty("role", "warn" if schedule_error else "hint")
        self.schedule_note.setVisible(bool(schedule_error))
        style.restyle(self.schedule_note)

    def _on_changed(self, *_args) -> None:
        if self._suppress:
            return
        self.refresh_readouts()
        self.changed.emit()

    def _on_randomize_toggled(self, checked: bool) -> None:
        # The field stays editable either way: with randomize on it shows the seed the
        # next run will replace, which is still worth being able to read and copy.
        self.seed_spin.setToolTip(
            "The next run will replace this with a fresh seed"
            if checked else f"Noise seed, 0 to {config.MAX_SEED}"
        )
        if not self._suppress:
            self.changed.emit()
