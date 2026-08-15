"""The Settings tab: where things are stored and how to reach ComfyUI.

Edits are staged and committed with Apply rather than taking effect as you type, because
every field here has a side effect -- reconnecting, or moving files on disk.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPlainTextEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from .. import config, graph_builder
from . import style


def _contract_report() -> str:
    """Which node plays which role, for the Diagnostics readout.

    Read fresh rather than taken from the running window, so that editing the workflow
    and reopening Settings shows the edit -- including one that no longer satisfies the
    contract, which is exactly when this is worth looking at.
    """
    try:
        workflow = config.load_workflow()
    except (FileNotFoundError, ValueError) as exc:
        return str(exc)

    graph, roles = workflow.graph, workflow.roles
    lines = [f"{config.WORKFLOW_PATH.name}: {len(graph)} nodes"]
    lines += [f"  {role:<16}{node_id:<6}{class_type}"
              for role, node_id, class_type in roles.describe(graph)]
    if roles.keep:
        lines.append("  h3-keep (kept from the orphan sweep): " + ", ".join(roles.keep))
    lines += [f"  WARNING: {w}" for w in graph_builder.geometry_warnings(graph, roles)]
    return "\n".join(lines)

#: model id -> (what to call it in the list, what choosing it costs or buys). Kept here
#: rather than in config because it is wording, not behaviour; config.POSE_MODELS stays the
#: list of what exists, and anything it gains without an entry here still appears, by id.
POSE_MODEL_LABELS = {
    "vitpose-l": ("ViTPose-L (recommended)",
                  "The strongest of these on occluded and partial bodies."),
    "vitpose-b": ("ViTPose-B (faster)",
                  "About a third of the size. Quicker per frame, looser on hard poses."),
    "vitpose-l-wholebody": ("ViTPose-L wholebody",
                            "Adds face, hands and feet keypoints to the same body model."),
}

#: style id -> (list caption, what it draws differently). Independent of the model above:
#: the models differ in which points they find, these in what is joined to what.
POSE_STYLE_LABELS = {
    "openpose": ("OpenPose (standard)",
                 "The convention pose-conditioned models are trained on: both hips hang "
                 "off the neck."),
    "openpose-torso": ("OpenPose, torso from the shoulders",
                       "Each hip joins its own shoulder instead of the neck, and the two "
                       "hips join each other, so the trunk is a closed shape with width. "
                       "Anatomical, but no longer the trained convention."),
    "figure": ("Solid figure (easiest to read)",
               "Not the OpenPose convention at all: a filled trunk, a real head, black "
               "outlines so crossing limbs separate, and warm on the right against cool "
               "on the left. It works out which way the subject is facing and draws a "
               "face only when there is one to see. Wholebody adds the feet, which no "
               "other style draws. For looking at rather than conditioning on."),
}


class PathRow(QWidget):
    """A path field with Browse, and optionally a reset to the built-in default."""

    changed = Signal()

    def __init__(self, placeholder: str = "", caption: str = "Choose a folder",
                 default_label: str | None = None, parent=None):
        super().__init__(parent)
        self.caption = caption

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(5)

        self.edit = QLineEdit()
        self.edit.setPlaceholderText(placeholder)
        # Wrapped, not connected straight to changed.emit: textChanged carries a str, and
        # a zero-argument Signal's emit rejects it (unlike a plain Python callable, which
        # PySide truncates the extra argument for).
        self.edit.textChanged.connect(lambda _text: self.changed.emit())
        layout.addWidget(self.edit, 1)

        browse = QPushButton("Browse...")
        browse.clicked.connect(self._browse)
        layout.addWidget(browse)

        if default_label:
            reset = QPushButton(default_label)
            reset.setToolTip("Go back to the built-in location")
            reset.clicked.connect(lambda: self.edit.setText(""))
            layout.addWidget(reset)

    def _browse(self) -> None:
        start = self.edit.text().strip() or str(config.HOME)
        chosen = QFileDialog.getExistingDirectory(self, self.caption, start)
        if chosen:
            self.edit.setText(str(Path(chosen)))

    def text(self) -> str:
        return self.edit.text().strip()

    def setText(self, value: str) -> None:
        self.edit.setText(value or "")


class SettingsPanel(QWidget):
    """Staged settings. The main window applies them and reports the outcome."""

    apply_requested = Signal(dict)
    previews_toggled = Signal(bool)
    sage_toggled = Signal(bool)
    pose_model_changed = Signal(str)
    pose_style_changed = Signal(str)
    pose_cache_clear_requested = Signal()
    bundle_requested = Signal()
    reset_layout_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._loaded: dict = {}

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(8)

        # The groups keep their natural height and the page scrolls, rather than the
        # rows being squeezed into overlap when the panel is short.
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)
        layout.addWidget(self._build_storage())
        layout.addWidget(self._build_connection())
        layout.addWidget(self._build_acceleration())
        layout.addWidget(self._build_pose())
        layout.addWidget(self._build_diagnostics())
        layout.addWidget(self._build_interface())
        layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(body)
        outer.addWidget(scroll, 1)

        actions = self._build_actions()
        actions.setContentsMargins(10, 0, 10, 10)
        outer.addLayout(actions)

    # -- construction --------------------------------------------------------------

    def _build_storage(self) -> QGroupBox:
        group = QGroupBox("Storage")
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setVerticalSpacing(8)

        self.scenes_row = PathRow(
            placeholder=str(config.SCENES_DIR),
            caption="Choose where scenes are saved",
            default_label="Default")
        self.scenes_row.changed.connect(self._refresh_state)
        form.addRow("Scenes folder", self.scenes_row)

        self.scenes_note = style.hint("")
        self.scenes_note.setWordWrap(True)
        self.scenes_note.setMinimumHeight(32)
        form.addRow("", self.scenes_note)

        self.runs_label = _fixed_path_label()
        form.addRow("Results", self.runs_label)

        self.settings_label = _fixed_path_label()
        form.addRow("Settings file", self.settings_label)

        open_row = QHBoxLayout()
        open_row.setContentsMargins(0, 0, 0, 0)
        for text, getter in (
            ("Open scenes folder", lambda: config.resolve_scenes_dir(self.scenes_row.text())),
            ("Open results folder", lambda: config.RUNS_DIR),
            ("Open app folder", lambda: config.HOME),
        ):
            button = QPushButton(text)
            button.clicked.connect(lambda _=False, g=getter: _open(g()))
            open_row.addWidget(button)
        open_row.addStretch(1)
        form.addRow("", open_row)

        return group

    def _build_connection(self) -> QGroupBox:
        group = QGroupBox("ComfyUI")
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setVerticalSpacing(8)

        self.server_edit = QLineEdit()
        self.server_edit.setPlaceholderText(config.DEFAULT_SERVER_URL)
        self.server_edit.textChanged.connect(self._refresh_state)
        form.addRow("Address", self.server_edit)

        self.output_row = PathRow(
            placeholder="detected from the server",
            caption="Choose ComfyUI's output folder")
        self.output_row.changed.connect(self._refresh_state)
        form.addRow("Output folder", self.output_row)

        self.output_note = style.hint("")
        self.output_note.setWordWrap(True)
        form.addRow("", self.output_note)
        self.set_detected_output_dir("")

        self.prefix_edit = QLineEdit()
        self.prefix_edit.setPlaceholderText(config.DEFAULT_FILENAME_PREFIX)
        self.prefix_edit.setToolTip(
            "Where the output node files the finished video, inside ComfyUI's own output\n"
            "folder. A subfolder and a name stem, so 'videos/h3' writes\n"
            "output/videos/h3_00001.mp4. ComfyUI's date tokens work here:\n"
            "%date:yyyy-MM-dd%/shot writes a folder per day.")
        self.prefix_edit.textChanged.connect(self._refresh_state)
        form.addRow("Filename prefix", self.prefix_edit)

        note = style.hint(
            "ComfyUI adds the counter and the extension. Leave blank to keep whatever the "
            "workflow's own output node was saved with.")
        note.setWordWrap(True)
        form.addRow("", note)

        return group

    def set_detected_output_dir(self, path: str) -> None:
        """Say what was found, so the empty field reads as working rather than unset."""
        self.output_note.setText(
            f"Results are read from {path}, where ComfyUI wrote them - nothing is copied "
            "or downloaded. Fill the field in to override it."
            if path else
            "Optional. The server is asked where it writes; set this only when that "
            "answer is wrong here, as it is for a container or a network share. Without "
            "either, finished videos are downloaded into runs/videos.")

    def _build_acceleration(self) -> QGroupBox:
        group = QGroupBox("Acceleration")
        layout = QVBoxLayout(group)

        self.sage_box = QCheckBox("Sage Attention")
        self.sage_box.setToolTip(
            "Patch the diffusion model's attention with SageAttention, which is faster\n"
            "than the default at some cost in numerical exactness.\n\n"
            "This drives the switch in the workflow: off sends the loader's own model and\n"
            "the patch node is left out of the submitted graph entirely, rather than being\n"
            "included and bypassed.\n\n"
            "Takes effect on the next run. Needs the sageattention library on the server.")
        self.sage_box.toggled.connect(self.sage_toggled.emit)
        layout.addWidget(self.sage_box)

        note = style.hint(
            "Requires the sageattention library where ComfyUI runs. If a run fails "
            "complaining about it, turn this off.")
        note.setWordWrap(True)
        layout.addWidget(note)

        return group

    def _build_pose(self) -> QGroupBox:
        """Which weights the Pose toggle renders skeletons with.

        Immediate rather than staged, like the checkboxes above: there is nothing to
        reconnect or move on disk. Existing pose clips are not thrown away, because the
        model is part of ``pose.cache_key`` -- a clip rendered with the old one simply
        stops being a hit, and switching back finds it again.
        """
        group = QGroupBox("Pose")
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignRight)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        form.setVerticalSpacing(8)

        self.pose_model_box = QComboBox()
        for model in config.POSE_MODELS:
            label, _description = POSE_MODEL_LABELS.get(model, (model, ""))
            self.pose_model_box.addItem(label, model)
        self.pose_model_box.setToolTip(
            "The estimator run over a reference with Pose ticked.\n\n"
            "Weights are downloaded the first time a model is used, and the download is\n"
            "part of the pose pass rather than something to wait for here.")
        self.pose_model_box.currentIndexChanged.connect(self._on_pose_model_changed)
        form.addRow("Model", self.pose_model_box)

        self.pose_model_note = style.hint("")
        self.pose_model_note.setWordWrap(True)
        self.pose_model_note.setMinimumHeight(32)
        form.addRow("", self.pose_model_note)

        self.pose_style_box = QComboBox()
        for name in config.POSE_STYLES:
            label, _description = POSE_STYLE_LABELS.get(name, (name, ""))
            self.pose_style_box.addItem(label, name)
        self.pose_style_box.setToolTip(
            "How the keypoints are joined up.\n\n"
            "A property of the drawing, not of the estimator, so this applies to whichever\n"
            "model is selected above.")
        self.pose_style_box.currentIndexChanged.connect(self._on_pose_style_changed)
        form.addRow("Style", self.pose_style_box)

        self.pose_style_note = style.hint("")
        self.pose_style_note.setWordWrap(True)
        self.pose_style_note.setMinimumHeight(32)
        form.addRow("", self.pose_style_note)

        cache_row = QHBoxLayout()
        cache_row.setContentsMargins(0, 0, 0, 0)

        self.pose_clear_button = QPushButton("Clear pose clips")
        self.pose_clear_button.setToolTip(
            "Delete every skeleton clip rendered so far.\n\n"
            "They are only a cache: anything still needed is rendered again on the next\n"
            "run, which costs time rather than work. Your references are untouched.")
        self.pose_clear_button.clicked.connect(self.pose_cache_clear_requested.emit)
        cache_row.addWidget(self.pose_clear_button)

        open_cache = QPushButton("Open folder")
        open_cache.clicked.connect(lambda: _open(config.POSE_CACHE_DIR))
        cache_row.addWidget(open_cache)

        cache_row.addStretch(1)
        form.addRow("Rendered clips", cache_row)

        self.pose_cache_note = style.hint("")
        self.pose_cache_note.setWordWrap(True)
        form.addRow("", self.pose_cache_note)

        return group

    def refresh_pose_cache(self) -> None:
        """Say what is on disk, and disable the button when there is nothing to clear."""
        from .. import pose as pose_mod

        count, size = pose_mod.cache_usage()
        self.pose_clear_button.setEnabled(count > 0)
        if not count:
            self.pose_cache_note.setText("No skeleton clips rendered yet.")
            return
        self.pose_cache_note.setText(
            f"{count} clip(s), {size / (1 << 20):.1f} MB in {config.POSE_CACHE_DIR}")

    def _build_diagnostics(self) -> QGroupBox:
        """One button, for the question that comes up whenever a result looks wrong."""
        group = QGroupBox("Diagnostics")
        layout = QVBoxLayout(group)

        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)

        self.bundle_button = QPushButton("Export references")
        self.bundle_button.setToolTip(
            "Write out exactly what the model is about to be given: the reference\n"
            "files as they will be uploaded, the prompt as the node receives it, and\n"
            "a manifest of what the node then resizes each one to.\n\n"
            "Uses the editor's current contents. Nothing is uploaded.")
        self.bundle_button.clicked.connect(self.bundle_requested.emit)
        row.addWidget(self.bundle_button)

        open_bundle = QPushButton("Open folder")
        open_bundle.clicked.connect(lambda: _open(config.BUNDLE_DIR))
        row.addWidget(open_bundle)
        row.addStretch(1)
        layout.addLayout(row)

        self.bundle_note = style.hint(
            "A reference is cut, posed, rescaled or rotated before it is sent, and the "
            "node resizes it again on arrival. This writes down what survives all of "
            f"that, in {config.BUNDLE_DIR}.")
        self.bundle_note.setWordWrap(True)
        layout.addWidget(self.bundle_note)

        layout.addWidget(style.separator())
        layout.addWidget(style.hint("Workflow contract"))

        self.contract_view = QPlainTextEdit()
        self.contract_view.setReadOnly(True)
        style.mono(self.contract_view, size=9)
        self.contract_view.setPlainText(_contract_report())
        self.contract_view.setToolTip(
            "Which node in the workflow plays which part, found by the 'h3-' tag at the\n"
            "start of its ComfyUI title. Renumber or extend the workflow freely; keep the\n"
            "tags, and tag any branch of your own 'h3-keep' so it is not pruned.")
        self.contract_view.setMaximumHeight(180)
        layout.addWidget(self.contract_view)

        return group

    def _build_interface(self) -> QGroupBox:
        group = QGroupBox("Interface")
        layout = QVBoxLayout(group)

        self.previews_box = QCheckBox("Show sampler previews during a run")
        self.previews_box.setToolTip(
            "MiniMax H3 samples video and audio together, so the preview frames are "
            "often not meaningful. Off by default.")
        self.previews_box.toggled.connect(self.previews_toggled.emit)
        layout.addWidget(self.previews_box)

        # The panes cannot be closed, so nothing can be lost -- but an arrangement can
        # still end up somewhere you would rather it did not, and this is the way back.
        reset_row = QHBoxLayout()
        reset_row.setContentsMargins(0, 0, 0, 0)
        self.reset_layout_button = QPushButton("Reset layout")
        self.reset_layout_button.setToolTip(
            "Put the panes back to three columns of two")
        self.reset_layout_button.clicked.connect(self.reset_layout_requested.emit)
        reset_row.addWidget(self.reset_layout_button)
        reset_row.addStretch(1)
        layout.addLayout(reset_row)

        return group

    def _build_actions(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)

        self.status_label = style.hint("")
        self.status_label.setWordWrap(True)
        row.addWidget(self.status_label, 1)

        self.revert_button = QPushButton("Discard")
        self.revert_button.clicked.connect(lambda: self.load(self._loaded))
        row.addWidget(self.revert_button)

        self.apply_button = QPushButton("Apply")
        self.apply_button.setObjectName("primary")
        self.apply_button.clicked.connect(self._on_apply)
        row.addWidget(self.apply_button)

        return row

    # -- values --------------------------------------------------------------------

    def load(self, settings: dict) -> None:
        self._loaded = dict(settings)
        self.scenes_row.setText(settings.get("scenes_dir", ""))
        self.server_edit.setText(settings.get("server_url", ""))
        self.output_row.setText(settings.get("server_output_dir", ""))
        self.prefix_edit.setText(settings.get("filename_prefix", ""))

        # blockSignals: these take effect immediately rather than on Apply, so setting
        # them from stored settings must not read as the user toggling them.
        for box, key in ((self.previews_box, "show_previews"),
                         (self.sage_box, "sage_attention")):
            box.blockSignals(True)
            box.setChecked(bool(settings.get(key)))
            box.blockSignals(False)

        model = settings.get("pose_model") or config.DEFAULT_POSE_MODEL
        index = self.pose_model_box.findData(model)
        self.pose_model_box.blockSignals(True)
        self.pose_model_box.setCurrentIndex(
            index if index >= 0 else self.pose_model_box.findData(config.DEFAULT_POSE_MODEL))
        self.pose_model_box.blockSignals(False)
        self._refresh_pose_note()

        pose_style = settings.get("pose_style") or config.DEFAULT_POSE_STYLE
        style_index = self.pose_style_box.findData(pose_style)
        self.pose_style_box.blockSignals(True)
        self.pose_style_box.setCurrentIndex(
            style_index if style_index >= 0
            else self.pose_style_box.findData(config.DEFAULT_POSE_STYLE))
        self.pose_style_box.blockSignals(False)
        self._refresh_pose_style_note()
        self.refresh_pose_cache()

        self.runs_label.setText(str(config.RUNS_DIR))
        self.settings_label.setText(str(config.SETTINGS_PATH))
        self._refresh_state()

    def staged(self) -> dict:
        return {
            "scenes_dir": self.scenes_row.text(),
            "server_url": self.server_edit.text().strip(),
            "server_output_dir": self.output_row.text(),
            "filename_prefix": config.clean_filename_prefix(self.prefix_edit.text()),
        }

    def is_dirty(self) -> bool:
        return any(
            value != str(self._loaded.get(key, "") or "")
            for key, value in self.staged().items()
        )

    def set_status(self, message: str, role: str = "hint") -> None:
        self.status_label.setText(message)
        self.status_label.setProperty("role", role)
        style.restyle(self.status_label)

    def _refresh_state(self) -> None:
        from ..scenes import PROJECTS_FILENAME

        resolved = config.resolve_scenes_dir(self.scenes_row.text())
        count = len([p for p in resolved.glob("*.json") if p.name != PROJECTS_FILENAME]) \
            if resolved.is_dir() else 0

        if not self.scenes_row.text():
            where = f"{resolved}  (default)"
        else:
            where = str(resolved)
        exists = "" if resolved.is_dir() else "  -  will be created"
        self.scenes_note.setText(
            f"{where}{exists}\n{count} scene(s) here, one .json file each.")

        dirty = self.is_dirty()
        self.apply_button.setEnabled(dirty)
        self.revert_button.setEnabled(dirty)
        if dirty:
            self.set_status("Unapplied changes.", "warn")
        else:
            self.set_status("")

    def _on_pose_model_changed(self) -> None:
        self._refresh_pose_note()
        self.pose_model_changed.emit(self.pose_model_box.currentData())

    def _on_pose_style_changed(self) -> None:
        self._refresh_pose_style_note()
        self.pose_style_changed.emit(self.pose_style_box.currentData())

    def _refresh_pose_style_note(self) -> None:
        name = self.pose_style_box.currentData() or config.DEFAULT_POSE_STYLE
        _label, description = POSE_STYLE_LABELS.get(name, (name, ""))
        self.pose_style_note.setText(description)

    def _refresh_pose_note(self) -> None:
        """Say what the choice means and whether its weights are already on disk.

        The size is read from the file rather than declared alongside the URL: a number
        written down here would be one more thing that can quietly stop being true.
        """
        from .. import pose as pose_mod

        model = self.pose_model_box.currentData() or config.DEFAULT_POSE_MODEL
        _label, description = POSE_MODEL_LABELS.get(model, (model, ""))
        try:
            path = pose_mod.model_path(model)
        except pose_mod.PoseError:                 # a model id with no entry in config
            self.pose_model_note.setText(description)
            return

        if path.is_file():
            where = f"Downloaded  -  {path.stat().st_size / (1 << 20):.0f} MB in {path.parent}"
        else:
            where = "Not downloaded yet; the first pose pass with it will fetch it."
        self.pose_model_note.setText(f"{description}\n{where}" if description else where)

    def _on_apply(self) -> None:
        self.apply_requested.emit(self.staged())


def _fixed_path_label() -> QLabel:
    label = style.hint("")
    label.setTextInteractionFlags(Qt.TextSelectableByMouse)
    return label


def _open(path) -> None:
    from PySide6.QtCore import QUrl

    target = Path(path)
    if not target.is_dir():
        target = target.parent
    if target.is_dir():
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
