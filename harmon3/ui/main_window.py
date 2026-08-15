"""The main window: owns the worker threads and all cross-panel coordination."""

from __future__ import annotations

import copy
import logging
import time
from datetime import datetime
from pathlib import Path

from PySide6.QtCore import QMetaObject, QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStatusBar,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from .. import (
    bundle as bundle_mod, comfy_http, config, graph_builder, history,
    pose as pose_mod,
    progress as progress_mod, prompt as prompt_mod, refs as refs_mod,
    runqueue, scaling as scaling_mod, settings as settings_mod,
)
from ..comfy_ws import ComfyWsClient
from ..graph_builder import BuildState
from ..history import HistoryStore, RunRecord
from ..jobs import JobFailure, JobRequest, JobRunner
from ..refs import VIDEO, RefRow, RefSet
from ..scenes import Scene, SceneStore, SceneWriteError, clean_description
from . import docks, style
from .history_panel import HistoryPanel
from .media_view import MediaItem
from .params_panel import ParamsPanel
from .player import VideoPlayer
from .pose_worker import PoseJob, PoseRunner
from .probe import MediaProbe
from .prompt_editor import PromptEditor
from .ref_panel import RefPanel
from .scene_panel import ScenePanel
from .settings_panel import SettingsPanel

log = logging.getLogger(__name__)

REACHABILITY_INTERVAL_MS = 5000
SETTINGS_DEBOUNCE_MS = 500

#: While one of these is executing, the live preview keeps animating.
SAMPLER_CLASSES = {
    "MiniMaxH3ProgressiveSampler", "MiniMaxH3ScheduleProfiler",
    "SamplerCustomAdvanced", "SamplerCustom", "KSampler", "KSamplerAdvanced",
}


class MainWindow(QMainWindow):
    #: Emitted to the job thread; queued automatically because the worker lives there.
    _submit_requested = Signal(object)
    _fetch_requested = Signal(str, object, str)
    _fetch_from_history_requested = Signal(str, str)
    _requeue_requested = Signal(object)
    _interrupt_requested = Signal(str, bool)          # prompt_id, is it executing
    _check_requested = Signal(object)
    _server_changed = Signal(str)
    _verify_refs_requested = Signal(object)
    _pose_requested = Signal(object, object)          # [PoseJob], PoseSettings

    def __init__(self, server_url: str, client_id: str):
        super().__init__()
        self.setWindowTitle("HARMON3 - MiniMax H3 reference to video")
        self.resize(1360, 900)

        self.client_id = client_id
        self.settings = settings_mod.load_settings()
        if server_url:
            self.settings["server_url"] = server_url

        workflow = config.load_workflow()
        self.base_workflow = workflow.graph
        self.roles = workflow.roles
        self.state = graph_builder.state_from_workflow(self.base_workflow, self.roles)
        settings_mod.apply_to_state(self.state, self.settings)

        self.history_store = HistoryStore()
        self.scene_store = SceneStore(config.resolve_scenes_dir(self.settings.get("scenes_dir")))
        #: The scene the editor is currently working on, if any. Purely a label plus a
        #: target for Update/Revert -- the editor stays fully usable without one.
        self.current_scene: Scene | None = None
        #: Set when a scene is loaded with the intent of queueing it straight away.
        self._run_scene_when_ready = False
        self.reachable = False
        #: Where the connected server said it writes results, if this machine can read it.
        #: Not persisted: it describes a server, not a preference.
        self.detected_output_dir = ""
        #: Everything submitted and not yet finished, oldest first. ComfyUI's queue is a
        #: FIFO, so the head is the run it is working on -- see harmon3/runqueue.py.
        self.runs = runqueue.RunQueue()
        self.total_steps = 0
        self.eta = progress_mod.EtaEstimator()
        #: Turns the staged sampler's per-stage step counts into one run-wide position.
        #: One per source: ComfyUI's progress messages and the preview node's own count
        #: restart at different moments, and a shared tracker would read one's rewind as
        #: the other's.
        self.stages = progress_mod.StageTracker()
        self.preview_stages = progress_mod.StageTracker()
        self.stage = ""
        #: Prompt id whose result download has already been requested. `executed` and
        #: `execution_success` both arrive before the file lands, so without this the
        #: history fallback would start a second, redundant download.
        self.fetch_requested_for: str | None = None
        #: True from the moment Queue is pressed until the server accepts or rejects it.
        #: active_prompt_id is still None during that window, so without this an edit
        #: that refreshes the UI would re-enable Queue and allow a second submission.
        self.submitting = False
        #: True while skeletons are being rendered locally. Its own flag rather than a
        #: sub-state of `submitting`, because a pose pass also runs on its own, from the
        #: Pose thumbnail, with no run behind it.
        self.posing = False
        #: Whether the pass currently running is the first half of a queue.
        self._pose_then_submit = False
        #: Guards the editor signal handlers while widgets are being populated, so a
        #: half-built panel cannot write itself back over the state being loaded.
        self._loading = True

        self._save_timer = QTimer(self)
        self._save_timer.setSingleShot(True)
        self._save_timer.setInterval(SETTINGS_DEBOUNCE_MS)
        self._save_timer.timeout.connect(self._save_settings)

        self._build_ui()
        self._start_workers()
        self._load_state_into_ui()
        self._restore_geometry()

        self.history_panel.set_records(self.history_store.load())

        self.scene_store.load_all()
        restored = self.settings.get("current_scene")
        if restored:
            self.current_scene = self.scene_store.find(restored)
        self._refresh_scenes()
        self.settings_panel.load(self.settings)

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(1000)
        self._elapsed_timer.timeout.connect(self._tick_elapsed)

        self._reachability_timer = QTimer(self)
        self._reachability_timer.setInterval(REACHABILITY_INTERVAL_MS)
        self._reachability_timer.timeout.connect(self._poll_reachability)
        self._reachability_timer.start()

        QTimer.singleShot(0, self._poll_reachability)
        QTimer.singleShot(0, self._refresh_derived)

    # -------------------------------------------------------------------- active run
    #
    # Everything about "the run" is a view onto the head of the queue rather than its own
    # field, so there is one place a run can be alive and no way for the two to disagree.

    @property
    def active_prompt_id(self) -> str | None:
        return self.runs.active_id

    @property
    def active_record(self) -> RunRecord | None:
        run = self.runs.active
        return run.record if run is not None else None

    @property
    def active_built(self):
        run = self.runs.active
        return run.built if run is not None else None

    @property
    def run_started_at(self) -> float | None:
        run = self.runs.active
        return run.started_at if run is not None else None

    # ------------------------------------------------------------------ construction

    def _build_ui(self) -> None:
        docks.configure(self)
        self._build_banner_bar()

        self.ref_panel = RefPanel()
        self.ref_panel.structure_changed.connect(self._on_refs_changed)
        self.ref_panel.tag_clicked.connect(self._on_tag_clicked)
        self.ref_panel.preview_requested.connect(self._on_ref_preview)
        self.ref_panel.scale_changed.connect(self._on_ref_scale_changed)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.NoFrame)
        left_scroll.setWidget(self.ref_panel)
        # Low enough that three columns still fit the window this opens on; the rows
        # elide and wrap rather than demanding room.
        left_scroll.setMinimumWidth(240)
        # Reference rows shrink to the viewport rather than scrolling sideways; the
        # filename elides and the warnings wrap.
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        self.prompt_editor = PromptEditor()
        self.prompt_editor.changed.connect(self._on_edited)

        self.params_panel = ParamsPanel()
        # Which parameters the loaded workflow actually has a node for: the staging
        # schedule only exists on a progressive sampler.
        self.params_panel.set_workflow_features(self.roles)
        self.params_panel.changed.connect(self._on_params_changed)
        self.params_panel.check_requested.connect(self._on_check_clicked)
        params_page = QWidget()
        params_layout = QVBoxLayout(params_page)
        params_layout.setContentsMargins(0, 0, 0, 0)
        params_layout.addWidget(self.params_panel)
        # The parameters keep their natural height rather than stretching to fill the tab.
        params_layout.addStretch(1)
        params_page.setMinimumWidth(240)

        self.player = VideoPlayer()
        self.player.trim_changed.connect(self._on_trim_changed)

        self.scene_panel = ScenePanel()
        self.scene_panel.save_as_requested.connect(self._on_scene_save_as)
        self.scene_panel.update_requested.connect(self._on_scene_update)
        self.scene_panel.load_requested.connect(self._on_scene_load)
        self.scene_panel.run_requested.connect(self._on_scene_run)
        self.scene_panel.revert_requested.connect(self._on_scene_revert)
        self.scene_panel.details_requested.connect(self._on_scene_details)
        self.scene_panel.duplicate_requested.connect(self._on_scene_duplicate)
        self.scene_panel.delete_requested.connect(self._on_scene_delete)
        self.scene_panel.project_create_requested.connect(self._on_project_create)
        self.scene_panel.project_rename_requested.connect(self._on_project_rename)
        self.scene_panel.project_delete_requested.connect(self._on_project_delete)
        self.scene_panel.scene_moved.connect(self._on_scene_moved)

        self.history_panel = HistoryPanel()
        self.history_panel.replay_requested.connect(self._on_replay)
        self.history_panel.requeue_requested.connect(self._on_requeue)
        self.history_panel.restore_requested.connect(self._on_restore)

        self.settings_panel = SettingsPanel()
        self.settings_panel.apply_requested.connect(self._on_settings_applied)
        self.settings_panel.previews_toggled.connect(self._on_previews_toggled)
        self.settings_panel.sage_toggled.connect(self._on_sage_toggled)
        self.settings_panel.pose_model_changed.connect(self._on_pose_model_changed)
        self.settings_panel.pose_style_changed.connect(self._on_pose_style_changed)
        self.settings_panel.pose_cache_clear_requested.connect(self._on_pose_cache_clear)
        self.settings_panel.bundle_requested.connect(self._on_export_bundle)
        self.settings_panel.reset_layout_requested.connect(self._on_reset_layout)

        self.other_tabs = QTabWidget()
        self.other_tabs.addTab(params_page, "Parameters")
        self.other_tabs.addTab(self.settings_panel, "Settings")
        self.other_tabs.addTab(self.history_panel, "History")
        self.other_tabs.setTabToolTip(1, "Storage locations and the ComfyUI address")

        self.dock_projects = docks.make_dock(docks.DOCK_PROJECTS, "Projects", self.scene_panel)
        self.dock_references = docks.make_dock(docks.DOCK_REFERENCES, "References", left_scroll)
        self.dock_viewer = docks.make_dock(docks.DOCK_VIEWER, "Viewer", self.player)
        self.dock_prompts = docks.make_dock(docks.DOCK_PROMPTS, "Prompts", self.prompt_editor)
        self.dock_run = docks.make_dock(docks.DOCK_RUN, "Run", self._build_run_pane())
        self.dock_other = docks.make_dock(docks.DOCK_OTHER, "Other", self.other_tabs)

        # Floating a pane reparents its contents into a new native window, which throws
        # away the video surfaces the viewer is presenting through.
        self.dock_viewer.topLevelChanged.connect(
            lambda _floating: self.player.refresh_surfaces())

        docks.apply_default_layout(self)

        self.setStatusBar(self._build_status_bar())
        style.stylise(self)

    def _build_banner_bar(self) -> None:
        """The notice strip, above every pane and outside the arrangement.

        A toolbar area rather than a dock: it spans the full width with no corner rules to
        reason about, has no separator the user can drag it taller by, and collapses to
        nothing when there is no notice. The bar carries the visibility; the widget inside
        it stays shown.
        """
        self.banner = QWidget()
        layout = QHBoxLayout(self.banner)
        layout.setContentsMargins(0, 0, 0, 0)

        self.banner_label = QLabel()
        self.banner_label.setWordWrap(True)
        self.banner_label.setProperty("role", "error")
        layout.addWidget(self.banner_label, 1)

        retry = QPushButton("Retry")
        retry.clicked.connect(self._poll_reachability)
        layout.addWidget(retry)

        self.banner_bar = QToolBar("Notices")
        self.banner_bar.setObjectName("toolbar.banner")
        self.banner_bar.setMovable(False)
        self.banner_bar.setFloatable(False)
        self.banner_bar.addWidget(self.banner)
        self.addToolBar(Qt.TopToolBarArea, self.banner_bar)
        self.banner_bar.hide()

    def _build_run_pane(self) -> QWidget:
        """Queue, Cancel and the progress bar on one row; what a run is doing on the next."""
        pane = QWidget()
        outer = QVBoxLayout(pane)
        outer.setContentsMargins(8, 6, 8, 8)
        outer.setSpacing(6)
        # The pane is a strip, not a panel: it keeps its two rows and gives the rest of
        # the column to whatever is under it.
        pane.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)

        top = QHBoxLayout()
        top.setContentsMargins(0, 0, 0, 0)
        top.setSpacing(8)

        self.queue_button = QPushButton("Queue")
        self.queue_button.setObjectName("primary")
        self.queue_button.clicked.connect(self._on_queue_clicked)
        top.addWidget(self.queue_button)

        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.setEnabled(False)
        self.cancel_button.clicked.connect(self._on_cancel_clicked)
        top.addWidget(self.cancel_button)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("idle")
        # Free to be narrow: the bar is the part of this row that can give way, and the
        # pane is a column rather than the full width of the window now.
        self.progress_bar.setMinimumWidth(70)
        style.mono(self.progress_bar, size=8)
        top.addWidget(self.progress_bar, 1)

        outer.addLayout(top)

        bottom = QHBoxLayout()
        bottom.setContentsMargins(0, 0, 0, 0)
        bottom.setSpacing(8)

        # The only at-a-glance marker that the editor has drifted from the loaded scene.
        self.scene_button = QPushButton()
        self.scene_button.setFlat(True)
        self.scene_button.setToolTip("Show the projects and their scenes")
        self.scene_button.clicked.connect(lambda: self._reveal_dock(self.dock_projects))
        bottom.addWidget(self.scene_button)

        # Free to be narrower than its text rather than given a floor: the run pane is a
        # column now, and the stage name is the first thing that should give way when the
        # column is narrow. A plain label, so text() is still what was set.
        self.stage_label = QLabel("")
        self.stage_label.setProperty("role", "hint")
        self.stage_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.stage_label.setMinimumWidth(0)
        bottom.addWidget(self.stage_label, 1)

        self.elapsed_label = QLabel("")
        self.elapsed_label.setMinimumWidth(48)
        style.mono(self.elapsed_label, size=9)
        bottom.addWidget(self.elapsed_label)

        self.preview_label = QLabel()
        self.preview_label.setFixedSize(64, 40)
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.hide()
        bottom.addWidget(self.preview_label)

        outer.addLayout(bottom)
        return pane

    def _build_status_bar(self) -> QStatusBar:
        status = QStatusBar()

        self.connection_label = QLabel("connecting...")
        status.addWidget(self.connection_label)

        self.queue_label = QLabel("")
        status.addPermanentWidget(self.queue_label)

        self.summary_label = style.metric()
        status.addPermanentWidget(self.summary_label)

        self.log_button = QPushButton("Log")
        self.log_button.setToolTip("Show the diagnostic log")
        self.log_button.clicked.connect(self._show_log)
        status.addPermanentWidget(self.log_button)

        self.log_lines: list[str] = []
        return status

    def _reveal_dock(self, dock, page=None) -> None:
        """Bring a pane forward, whether it is tabbed behind another or floating."""
        dock.show()
        dock.raise_()
        if page is not None:
            self.other_tabs.setCurrentWidget(page)

    def _on_reset_layout(self) -> None:
        docks.apply_default_layout(self)

    # ---------------------------------------------------------------------- workers

    def _start_workers(self) -> None:
        server = self.settings["server_url"]

        self.job_thread = QThread(self)
        self.job_thread.setObjectName("harmon3-jobs")
        self.job_runner = JobRunner(server, self.client_id)
        self.job_runner.moveToThread(self.job_thread)

        self.job_runner.reachability_changed.connect(self._on_reachability)
        self.job_runner.output_dir_found.connect(self._on_output_dir_found)
        self.job_runner.object_info_ready.connect(self._on_object_info)
        self.job_runner.model_problems.connect(self._on_model_problems)
        self.job_runner.upload_started.connect(self._on_upload_started)
        self.job_runner.upload_cache_updated.connect(self._on_upload_cached)
        self.job_runner.submitted.connect(self._on_submitted)
        self.job_runner.requeued.connect(self._on_requeued)
        self.job_runner.submit_failed.connect(self._on_submit_failed)
        self.job_runner.download_progress.connect(self._on_download_progress)
        self.job_runner.downloaded.connect(self._on_downloaded)
        self.job_runner.download_failed.connect(self._on_download_failed)
        self.job_runner.check_finished.connect(self._on_check_finished)
        self.job_runner.server_refs_verified.connect(self._on_server_refs_verified)

        self._verify_refs_requested.connect(self.job_runner.verify_server_refs)
        self._submit_requested.connect(self.job_runner.submit_job)
        self._fetch_requested.connect(self.job_runner.fetch_result)
        self._fetch_from_history_requested.connect(self.job_runner.fetch_result_from_history)
        self._requeue_requested.connect(self.job_runner.requeue)
        self._interrupt_requested.connect(self.job_runner.interrupt)
        self._check_requested.connect(self.job_runner.check_workflow)
        self._server_changed.connect(self.job_runner.set_server)

        self.job_thread.start()

        self.ws_thread = QThread(self)
        self.ws_thread.setObjectName("harmon3-ws")
        self.ws_client = ComfyWsClient(server, self.client_id)
        self.ws_client.moveToThread(self.ws_thread)
        self.ws_thread.started.connect(self.ws_client.run)

        self.ws_client.queue_changed.connect(self._on_queue_changed)
        self.ws_client.execution_start.connect(self._on_execution_start)
        self.ws_client.execution_cached.connect(self._on_execution_cached)
        self.ws_client.executing.connect(self._on_executing)
        self.ws_client.progress.connect(self._on_progress)
        self.ws_client.executed.connect(self._on_executed)
        self.ws_client.execution_success.connect(self._on_execution_success)
        self.ws_client.execution_error.connect(self._on_execution_error)
        self.ws_client.execution_interrupted.connect(self._on_execution_interrupted)
        self.ws_client.preview_frame.connect(self._on_preview_frame)
        self.ws_client.preview_clip.connect(self._on_preview_clip)
        self.ws_client.previews_enabled = bool(self.settings.get("show_previews"))

        self.ws_thread.start()

        # Its own thread rather than the job thread's: a pose pass takes tens of seconds,
        # and while `submit_job` is blocked that thread's event loop cannot deliver a
        # cancel. This one stays free the whole time it is working.
        self.pose_thread = QThread(self)
        self.pose_thread.setObjectName("harmon3-pose")
        self.pose_runner = PoseRunner()
        self.pose_runner.moveToThread(self.pose_thread)

        self.pose_runner.progress.connect(self._on_pose_progress)
        self.pose_runner.rendered.connect(self._on_pose_rendered)
        self.pose_runner.failed.connect(self._on_pose_failed)
        self.pose_runner.finished.connect(self._on_pose_finished)
        self.pose_runner.provider_known.connect(self._on_pose_provider)
        self._pose_requested.connect(self.pose_runner.render_all)

        self.pose_thread.start()

        self.media_probe = MediaProbe(self)
        self.media_probe.probed.connect(self._on_media_probed)
        self.media_probe.unreadable.connect(self._on_media_unreadable)

    # ------------------------------------------------------------------- state sync

    def _load_state_into_ui(self) -> None:
        # Populating any one panel emits its change signal, which would otherwise call
        # _collect_state() and overwrite the state from the panels not yet filled in.
        self._loading = True
        try:
            self.ref_panel.set_refset(self.state.refs)
            self.ref_panel.set_last_dirs(self.settings.get("last_browse_dirs"))
            self.prompt_editor.set_sections(self.state.prompt_sections)
            self.params_panel.load_state(
                self.state, bool(self.settings.get("randomize_seed", True)))
        finally:
            self._loading = False
        # Loading a scene builds fresh rows, so anything the trim editor was holding is
        # now a row that belongs to nothing.
        self.player.forget_rows_not_in(self.state.refs.all_rows())
        self._probe_all_media()
        self._refresh_derived()

    def _collect_state(self) -> BuildState:
        self.state.prompt_sections = self.prompt_editor.sections_text()
        self.state.refs = self.ref_panel.refset()
        self.params_panel.apply_to_state(self.state)
        return self.state

    def _on_edited(self) -> None:
        if self._loading:
            return
        self._collect_state()
        self._refresh_derived()
        self._save_timer.start()

    def _on_params_changed(self) -> None:
        if self._loading:
            return
        self._collect_state()
        self._refresh_derived()
        self._save_timer.start()

    def _on_ref_scale_changed(self, row) -> None:
        """A size ceiling moved. Re-render what the frame is showing, and record it.

        Runs on every tick of the slider, so it does the cheap half only: the picture and
        the row's own readout. The rescaled copy itself is not made until the run is
        queued -- resampling a 12-megapixel reference per pixel of slider travel would
        make the control unusable, and the file is not needed until then anyway.
        """
        self.player.refresh_scale(row)
        # A clip's ceiling changes the frames themselves, so whatever was rendered for the
        # old one no longer describes what would be sent. The cache keeps it -- the canvas
        # is part of the key -- so putting the slider back finds it again.
        if row.kind == VIDEO:
            row.pose_path = row.pose_section = None
        self._on_edited()

    def _on_tag_clicked(self, tag: str) -> None:
        self.prompt_editor.insert_tag(tag)

    def _on_ref_preview(self, row, which: str, additive: bool) -> None:
        """Show a reference in the result frame; shift-click puts it beside what is there."""
        if which == "pose":
            # The pose thumbnail shows what will be sent in this reference's place. If it
            # has not been rendered yet, clicking is how you ask for it.
            if not row.pose_path or not Path(row.pose_path).is_file():
                self._on_pose_preview(row)
                return
            self.player.show_media([MediaItem(
                path=row.pose_path, caption=f"{row.display_name} (pose)", row=None)])
            return

        path = row.local_path
        if not path or not Path(path).is_file():
            # A row restored from history names a server file and nothing local, so there
            # is no clip to mark against -- but the in point is still settable, and this
            # is the only way to reach it.
            if row.supports_trim:
                self.player.edit_trim(row)
            return

        item = MediaItem(path=path, caption=row.display_name, row=row)

        # Marking needs the clip on its own; a comparison has no single subject.
        items = self.player.shown_media() if additive else []
        # Clicking the same thing twice should not double it up.
        items = [existing for existing in items if existing.path != item.path]
        items.append(item)
        self.player.show_media(items[-2:])

    # ------------------------------------------------------------ marking a reference

    def _shown_row(self) -> RefRow | None:
        """The single reference the result frame is showing, if it is showing just one."""
        items = self.player.shown_media()
        return items[0].row if len(items) == 1 else None

    def _on_refs_changed(self) -> None:
        if self._loading:
            return
        state = self._collect_state()
        # The trim editor holds a row; if that row has just been removed or pointed at a
        # different file, what it is editing no longer exists.
        self.player.forget_rows_not_in(state.refs.all_rows())
        self._refresh_derived()
        self._probe_all_media()
        self._save_timer.start()

    def _on_trim_changed(self) -> None:
        """A section was marked in the result frame; the rows and queue state follow it."""
        if self._loading:
            return
        self._refresh_derived()
        self._save_timer.start()

    def _refresh_derived(self) -> None:
        """Recompute tags, per-row diagnostics and the queue-readiness state."""
        if self._loading:
            return
        state = self._collect_state()
        tags = refs_mod.compute_tags(state.refs)
        self.prompt_editor.update_tags(tags)

        frames = self.params_panel.frames()
        # A reference runs from its mark for the generated length, so the trim editor's
        # window moves with the duration parameter rather than with anything on the row.
        self.player.set_target_frames(frames)

        # Both of those also move which frames a skeleton would cover, so a render made
        # for the old section stops describing this one.
        duration = self.params_panel.duration_seconds()
        for row in state.refs.videos:
            pose_mod.forget_stale(row, duration)

        unused = set(self.prompt_editor.unused_tags())
        blocking: list[str] = []

        for widget in self.ref_panel.all_row_widgets():
            widget.set_tags(tags.tag_for(widget.row), tags.soundtrack_tag_for(widget.row))
            widget.set_used_in_prompt(tags.tag_for(widget.row) not in unused)
            errors, _ = widget.refresh_details(frames)
            blocking.extend(f"{widget.row.display_name}: {e}" for e in errors)

        blocking.extend(self.params_panel.problems())

        self._refresh_scene_indicator()

        width, height = self.params_panel.resolution()
        self.summary_label.setText(
            f"{width}×{height}  ·  {frames}f  ·  {len(state.refs.all_rows())} refs")

        # A run in flight no longer blocks: it is only this app's own submission path and
        # a local pose pass that can hold up the next one.
        busy = self.submitting or self.posing
        self.queue_button.setEnabled(self.reachable and not busy and not blocking)
        self._refresh_queue_controls()

        if blocking:
            self.queue_button.setToolTip("Fix before queueing:\n- " + "\n- ".join(blocking))
        elif not self.reachable:
            self.queue_button.setToolTip("The ComfyUI server is not reachable")
        elif self.posing:
            self.queue_button.setToolTip("Rendering a pose clip")
        elif self.submitting:
            self.queue_button.setToolTip("Sending the previous run to the server")
        else:
            self.queue_button.setToolTip(
                "Add another run to the queue" if self.runs else "")

    def _refresh_queue_controls(self) -> None:
        """The Cancel button carries the depth of the queue, since nothing else shows it."""
        outstanding = len(self.runs)
        self.cancel_button.setEnabled(bool(outstanding) or self.posing)
        self.cancel_button.setText(
            f"Cancel ({outstanding})" if outstanding > 1 else "Cancel")
        if outstanding > 1:
            self.cancel_button.setToolTip(
                f"{outstanding} runs outstanding. Cancel takes the oldest first, so "
                "pressing it again works back through the queue in the order it was "
                "submitted.")
        else:
            self.cancel_button.setToolTip("Cancel the run")

    def _probe_all_media(self) -> None:
        pending = []
        for widget in self.ref_panel.all_row_widgets():
            row = widget.row
            if row.local_path and row.has_audio is None and row.fps is None:
                self.media_probe.probe(row.uid, row.local_path, row.kind)
            elif not row.needs_upload and row.server_missing is None:
                filename, subfolder = row.server_location()
                pending.append((row.uid, filename, subfolder))
        if pending and self.reachable:
            self._verify_refs_requested.emit(pending)


    def _live_rows(self) -> list:
        """Every reference row the window is holding."""
        return [widget.row for widget in self.ref_panel.all_row_widgets()]

    def _row_by_uid(self, uid: int):
        return next((row for row in self._live_rows() if row.uid == uid), None)

    def _on_server_refs_verified(self, results) -> None:
        by_uid = dict(results)
        for widget in self.ref_panel.all_row_widgets():
            if widget.row.uid in by_uid:
                widget.row.server_missing = not by_uid[widget.row.uid]
        self._refresh_derived()

    def _on_media_unreadable(self, uid: int, reason: str) -> None:
        """Nothing here can decode the file, so nothing further should be attempted on it.

        ComfyUI runs the same decoders. Uploading it anyway buys a stack trace on the
        server several minutes later instead of a message on the row now.
        """
        row = self._row_by_uid(uid)
        if row is None:
            return
        row.unreadable_reason = reason
        self._log(f"{row.display_name} cannot be read: {reason}")
        self._refresh_derived()

    def _on_media_probed(self, uid: int, info: dict) -> None:
        row = self._row_by_uid(uid)
        if row is not None:
            row.unreadable_reason = None
            for key, value in info.items():
                if value is not None:
                    setattr(row, key, value)
        # This is where the trim editor gets the frame rate and length it was waiting for.
        self.player.refresh_trim()
        self._refresh_derived()

    # ------------------------------------------------------------------ reachability

    def _poll_reachability(self) -> None:
        QMetaObject.invokeMethod(self.job_runner, "check_reachable", Qt.QueuedConnection)

    def _on_reachability(self, reachable: bool, detail: str) -> None:
        was_reachable = self.reachable
        self.reachable = reachable

        if reachable:
            self.connection_label.setText(f"connected - {detail}")
            self.connection_label.setProperty("role", "ok")
            self.banner_bar.hide()
        else:
            self.connection_label.setText("not connected")
            self.connection_label.setProperty("role", "error")
            self._show_banner(
                f"ComfyUI is not reachable at {self.settings['server_url']} - {detail}", error=True)
        style.restyle(self.connection_label)

        if reachable and not was_reachable:
            self._log(f"Connected to {self.settings['server_url']} ({detail})")
            # Reference existence could not be checked while the server was down.
            self._probe_all_media()
        self._refresh_derived()

    def _on_output_dir_found(self, path: str) -> None:
        """Remember where the server writes, so results need not be copied anywhere.

        Not written into settings: it describes the server that happens to be connected,
        and persisting it would leave a stale path behind after switching to another one.
        The Settings field stays authoritative when it is filled in.
        """
        if path == self.detected_output_dir:
            return
        self.detected_output_dir = path
        self.settings_panel.set_detected_output_dir(path)
        if path:
            self._log(f"ComfyUI writes its results to {path}")

    def _output_dir(self) -> str:
        """What was configured, or failing that what the server said. Empty means /view.

        The setting wins because it is the override for the case detection cannot cover:
        the server reporting a path that means something different on this machine, as a
        container or a network share will.
        """
        return self.settings.get("server_output_dir", "") or self.detected_output_dir

    def _on_object_info(self, object_info: dict) -> None:
        """The server's node schemas arrived; offer the samplers it actually has.

        Repopulating rather than validating: the combos ship with stock ComfyUI's lists,
        and a server with sampler packs installed has more to offer than this build knew
        about when it was written.
        """
        self.params_panel.set_server_options(object_info)

    def _on_model_problems(self, problems: list) -> None:
        if problems:
            self._show_banner(
                "Missing model files on the ComfyUI server:\n- " + "\n- ".join(problems), error=True)
            for problem in problems:
                self._log(problem)

    def _show_banner(self, text: str, error: bool = False) -> None:
        self.banner_label.setText(text)
        self.banner_label.setProperty("role", "error" if error else "warn")
        style.restyle(self.banner_label)
        self.banner_bar.show()

    def _on_queue_changed(self, remaining: int) -> None:
        self.queue_label.setText(f"queue {remaining}" if remaining else "")

    # ------------------------------------------------------------------------- scenes

    def _scene_is_dirty(self) -> bool:
        """Whether the editor has drifted from the loaded scene's saved definition."""
        if self.current_scene is None:
            return False
        return not self.current_scene.matches_state(
            self._collect_state(), self.params_panel.randomize)

    def _refresh_scenes(self) -> None:
        """Rebuild the catalogue. Only when the set of scenes changes."""
        self.scene_panel.refresh(
            self.scene_store.scenes, self.scene_store.project_names(),
            self.current_scene, self._scene_is_dirty())
        self._refresh_scene_indicator()

    def _refresh_scene_indicator(self) -> None:
        """Update just the loaded-scene marker. Cheap enough to run on every keystroke."""
        dirty = self._scene_is_dirty()
        self.scene_panel.update_current(self.current_scene, dirty)

        if self.current_scene is None:
            self.scene_button.setText("No scene")
            self.scene_button.setProperty("role", "hint")
        else:
            marker = " *" if dirty else ""
            self.scene_button.setText(f"Scene: {self.current_scene.name}{marker}")
            self.scene_button.setProperty("role", "warn" if dirty else "ok")
        style.restyle(self.scene_button)

    def _set_current_scene(self, scene: Scene | None) -> None:
        self.current_scene = scene
        self.settings["current_scene"] = scene.name if scene else ""
        self._refresh_scenes()
        self._save_timer.start()

    def _on_scene_save_as(self, name: str, description: str = "") -> None:
        if self.scene_store.name_exists(name):
            existing = self.scene_store.find(name)
            confirm = QMessageBox.question(
                self, "Replace scene",
                f"A scene called “{name}” already exists. Replace it?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            if confirm != QMessageBox.Yes:
                return
            existing.capture_from_state(self._collect_state(), self.params_panel.randomize)
            existing.description = clean_description(description)
            self._write_scene(existing, f"Updated scene “{name}”")
            return

        scene = Scene.from_state(
            name, self._collect_state(), self.params_panel.randomize, description)
        self._write_scene(scene, f"Saved scene “{name}”")

    def _on_scene_update(self, scene: Scene) -> None:
        scene.capture_from_state(self._collect_state(), self.params_panel.randomize)
        self._write_scene(scene, f"Updated scene “{scene.name}”")

    def _write_scene(self, scene: Scene, message: str) -> None:
        try:
            self.scene_store.save(scene)
        except SceneWriteError as exc:
            QMessageBox.warning(
                self, "Scene not saved",
                f"“{scene.name}” could not be written:\n\n{exc}")
            return
        self._log(message)
        self._set_current_scene(scene)
        self.scene_panel.select(scene)

    def _on_scene_revert(self, scene: Scene) -> None:
        """Discard the edits and reload. Reverting IS the discard, so no save prompt."""
        confirm = QMessageBox.question(
            self, "Revert scene",
            f"Discard the unsaved changes and reload “{scene.name}”?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirm == QMessageBox.Yes:
            self._apply_scene(scene)
            self._log(f"Reverted to the saved “{scene.name}”")

    def _on_scene_load(self, scene: Scene, then_run: bool = False) -> None:
        if scene is not self.current_scene and not self._confirm_discarding_scene_edits():
            return
        self._apply_scene(scene)
        self._log(f"Loaded scene “{scene.name}”")

        if then_run:
            # Reference checks and media probes resolve asynchronously, so give them a
            # moment to land rather than queueing into a half-checked state.
            self._run_scene_when_ready = True
            QTimer.singleShot(1200, self._queue_loaded_scene)

    def _apply_scene(self, scene: Scene) -> None:
        """Put a scene into the editor. Render settings are left exactly as they are."""
        scene.apply_to_state(self.state)
        self.settings["randomize_seed"] = scene.randomize_seed
        self._load_state_into_ui()
        self._set_current_scene(scene)

        missing = scene.missing_local_files()
        if missing:
            self._show_banner(
                f"“{scene.name}” references {len(missing)} file(s) that no longer exist:\n- "
                + "\n- ".join(missing[:5]), error=True)

    def _on_scene_run(self, scene: Scene) -> None:
        self._on_scene_load(scene, then_run=True)

    def _queue_loaded_scene(self) -> None:
        if not self._run_scene_when_ready:
            return
        self._run_scene_when_ready = False

        if not self.queue_button.isEnabled():
            reason = self.queue_button.toolTip() or "the scene is not ready to run"
            QMessageBox.warning(self, "Cannot run this scene", reason)
            return
        self._on_queue_clicked()

    def _on_scene_details(self, scene: Scene, new_name: str, description: str) -> None:
        if self.scene_store.name_exists(new_name, exclude=scene):
            QMessageBox.warning(
                self, "Name in use", f"A scene called “{new_name}” already exists.")
            return
        try:
            self.scene_store.set_details(scene, new_name, description)
        except SceneWriteError as exc:
            QMessageBox.warning(self, "Scene not updated", str(exc))
            return
        if scene is self.current_scene:
            self.settings["current_scene"] = scene.name
        self._log(f"Updated the details of “{scene.name}”")
        self._refresh_scenes()
        self.scene_panel.select(scene)
        self._save_timer.start()

    def _on_scene_duplicate(self, scene: Scene) -> None:
        try:
            copy = self.scene_store.duplicate(scene)
        except SceneWriteError as exc:
            QMessageBox.warning(self, "Scene not duplicated", str(exc))
            return
        self._log(f"Duplicated “{scene.name}” as “{copy.name}”")
        self._refresh_scenes()
        self.scene_panel.select(copy)

    def _on_scene_delete(self, scene: Scene) -> None:
        self.scene_store.delete(scene)
        self._log(f"Deleted scene “{scene.name}”")
        if scene is self.current_scene:
            self._set_current_scene(None)
        else:
            self._refresh_scenes()

    # ------------------------------------------------------------------------ projects

    def _on_project_create(self, name: str) -> None:
        created = self.scene_store.create_project(name)
        if not created:
            return
        self._log(f"Created project “{created}”")
        self.scene_panel.expand_project(created)
        self._refresh_scenes()

    def _on_project_rename(self, old: str, new: str) -> None:
        used = self.scene_store.rename_project(old, new)
        if used != old:
            self._log(f"Renamed project “{old}” to “{used}”")
        self.scene_panel.expand_project(used)
        self._refresh_scenes()

    def _on_project_delete(self, name: str) -> None:
        released = self.scene_store.delete_project(name)
        self._log(f"Deleted project “{name}”"
                  + (f"; {released} scene(s) moved to Ungrouped" if released else ""))
        self._refresh_scenes()

    def _on_scene_moved(self, scene: Scene, project: str, position: int) -> None:
        """Put a dragged scene where it was dropped.

        ``position < 0`` means the end, which is what a drop onto a project row -- rather
        than between two of its scenes -- is taken to mean.
        """
        self.scene_store.set_project(scene, project,
                                     None if position < 0 else position)
        if project:
            self.scene_panel.expand_project(project)
        self._log(f"“{scene.name}” → {project or 'Ungrouped'}")
        self._refresh_scenes()
        self.scene_panel.select(scene)

    def _confirm_discarding_scene_edits(self) -> bool:
        """Ask before throwing away unsaved edits to the loaded scene."""
        if not self._scene_is_dirty():
            return True

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Unsaved changes")
        box.setText(f"“{self.current_scene.name}” has unsaved changes.")
        box.setInformativeText("Save them before loading a different scene?")
        save = box.addButton("Save", QMessageBox.AcceptRole)
        discard = box.addButton("Discard", QMessageBox.DestructiveRole)
        box.addButton(QMessageBox.Cancel)
        box.setDefaultButton(save)
        box.exec()

        if box.clickedButton() is save:
            self._on_scene_update(self.current_scene)
            return True
        return box.clickedButton() is discard

    # ----------------------------------------------------------------------- queueing

    def _on_queue_clicked(self) -> None:
        # A run already in flight is no reason to refuse another: ComfyUI queues them and
        # works through them in order. Only this app's own single-file path to the server
        # blocks -- one submission at a time, and a pose pass owns the run bar while it
        # renders.
        if self.submitting or self.posing:
            return
        if self.params_panel.randomize:
            self.params_panel.roll_seed()

        state = self._collect_state()
        problems = graph_builder.validate_state(state)
        blocking = [p for p in problems if not graph_builder.is_deferred(p)]
        if blocking:
            QMessageBox.warning(self, "Not ready", "\n".join(blocking))
            return

        self.submitting = True
        self.queue_button.setEnabled(False)
        self._log("Queueing a run")
        _print_prompt(state.prompt_text)

        # Two phases. Any posed reference without a current skeleton is rendered first,
        # locally, and only then does the run go out -- the graph must name a file that
        # exists. Rows whose skeleton is already cached skip straight to the second phase.
        if self._start_pose_pass(then_submit=True):
            return
        self._submit_now()

    def _submit_now(self) -> None:
        """Second phase: hand the snapshot to the job thread."""
        state = self._collect_state()
        # Only when there is nothing to report on already: a run in flight owns the bar,
        # and the new one is not worth interrupting its progress to announce.
        if not self.runs:
            self.stage_label.setText("uploading references...")
            self.progress_bar.setRange(0, 0)
            self.progress_bar.setFormat("preparing")

        # A snapshot, not the live state: the user can keep editing while this uploads.
        request = JobRequest.snapshot(
            state,
            self.settings.setdefault("upload_cache", {}),
            self._output_dir(),
        )
        # On the copy, never the live rows: a posed row submits its skeleton and starts at
        # zero, while the row on screen keeps naming the user's own file at its own mark.
        pose_mod.swap_in(request.state, self._rendered_poses())
        scaling_mod.swap_in(request.state, self._rescaled_images())
        self._submit_requested.emit(request)

    # ----------------------------------------------------------------------- scaling

    def _rescaled_images(self) -> dict[int, str]:
        """uid -> rescaled copy, for every image sending less than it holds.

        Done here rather than in a worker: resizing a still is tens of milliseconds and
        the results are cached by content and target size, so a repeat run of the same
        references does no work at all. A failure is reported and the row falls back to
        its full-size original, because sending a large reference is a worse run rather
        than a broken one.
        """
        resolved: dict[int, str] = {}
        for row in self._collect_state().refs.images:
            if not scaling_mod.wants_scaling(row):
                continue
            try:
                resolved[row.uid] = str(scaling_mod.render(row))
            except (scaling_mod.ScaleError, OSError) as exc:
                self._log(f"Could not resize {row.display_name}, sending it whole: {exc}")
        if resolved:
            self._log(f"Prepared {len(resolved)} reference image(s) before upload")
        return resolved

    # ------------------------------------------------------------------------ posing

    def _pose_settings(self):
        return settings_mod.pose_settings(self.settings)

    def _posed_rows(self) -> list:
        """The video rows with a local pass to run: a skeleton, a rescale, or both."""
        return [row for row in self._collect_state().refs.videos
                if pose_mod.needs_render(row) and Path(row.local_path).is_file()]

    def _rendered_poses(self) -> dict[int, str]:
        """uid -> rendered clip, for every such row that already has a current one."""
        return {row.uid: row.pose_path for row in self._posed_rows()
                if row.pose_path and Path(row.pose_path).is_file()}

    def _start_pose_pass(self, *, then_submit: bool, rows=None) -> bool:
        """Render whatever skeletons are missing. False when there was nothing to do."""
        if self.posing:
            return True
        duration = self.params_panel.duration_seconds()
        settings = self._pose_settings()

        jobs = []
        for row in (rows if rows is not None else self._posed_rows()):
            try:
                destination, cached = pose_mod.resolve(row, duration, settings)
            except OSError as exc:                 # the file went away between checks
                self._log(f"Cannot pose {row.display_name}: {exc}")
                continue
            start, length = pose_mod.section_for(row, duration)
            if cached:
                row.pose_path, row.pose_section = str(destination), (start, length)
                continue
            jobs.append(PoseJob(uid=row.uid, source=row.local_path,
                                display_name=row.display_name, start=start,
                                length=length, destination=str(destination),
                                canvas=pose_mod.canvas_for(row), skeleton=row.poses))

        if not jobs:
            self._refresh_derived()
            return False

        self.posing = True
        self._pose_then_submit = then_submit
        self.queue_button.setEnabled(False)
        self.cancel_button.setEnabled(True)      # a local pass is the one thing we can
        self.progress_bar.setRange(0, 100)       # actually stop on the spot
        self.progress_bar.setValue(0)
        self.stage_label.setText("posing references...")
        self._log(f"Rendering {len(jobs)} reference clip(s)")
        self._pose_requested.emit(jobs, settings)
        return True

    def _on_pose_progress(self, done: int, total: int, label: str) -> None:
        self.progress_bar.setRange(0, max(1, total))
        self.progress_bar.setValue(done)
        self.progress_bar.setFormat(f"{label}  %v/%m")
        self.stage_label.setText(label)

    def _on_pose_rendered(self, uid: int, path: str) -> None:
        row = self._row_by_uid(uid)
        if row is not None:
            row.pose_path = path
            row.pose_section = pose_mod.section_for(
                row, self.params_panel.duration_seconds())
        self._refresh_derived()

    def _on_pose_failed(self, uid: int, message: str) -> None:
        row = self._row_by_uid(uid)
        who = f"{row.display_name}: " if row is not None else ""
        self._log(f"Pose failed - {who}{message}")
        for widget in self.ref_panel.all_row_widgets():
            if row is not None and widget.row is row:
                widget.show_server_error(message)

    def _on_pose_finished(self, ok: bool) -> None:
        self.posing = False
        submit = self._pose_then_submit
        self._pose_then_submit = False
        # A pass is the only thing that adds to the cache, so it is the only place the
        # settings panel's count can go stale behind the user's back.
        self.settings_panel.refresh_pose_cache()

        if ok and submit:
            self._submit_now()
            return

        # Either something failed or the user stopped it. Either way the run does not go
        # out: a graph naming a skeleton that was never rendered would fail on the server
        # several minutes later, which is a worse way to find out.
        self.submitting = False
        # Whatever was already queued is untouched by a pose pass that came to nothing.
        self._settle()
        if not ok and submit:
            self.stage_label.setText("pose failed - not queued")

    def _on_pose_provider(self, provider: str) -> None:
        self._log(f"Pose estimation is running on {provider}")
        self.statusBar().showMessage(f"Pose estimation: {provider}", 8000)

    def _on_pose_preview(self, row) -> None:
        """The Pose thumbnail was clicked with nothing rendered yet: render it now."""
        self._start_pose_pass(then_submit=False, rows=[row])

    # -----------------------------------------------------------------------------

    def _on_cancel_clicked(self) -> None:
        """Cancel the oldest outstanding run, so repeated presses clear a queue in order."""
        self.stage_label.setText("cancelling...")
        # A local pose pass is stopped by a plain flag the worker checks between frames;
        # only a run that has reached the server needs the queued round trip.
        if self.posing:
            self.pose_runner.cancel()
            return

        run = self.runs.active
        if run is None:
            return
        # `started` is the difference between deleting a queued job and interrupting the
        # server: until the server says our run is executing, whatever is executing is
        # somebody else's and must not be taken down with it.
        self._interrupt_requested.emit(run.prompt_id, run.started)

        # A run the server never began emits no `execution_interrupted`, so nothing else
        # would ever close it out.
        if not run.started:
            self._end_run(run.prompt_id, history.STATUS_CANCELLED, "cancelled")
            self._log(f"Cancelled a queued run ({len(self.runs)} still queued)")

    def _on_upload_started(self, name: str) -> None:
        self.stage_label.setText(f"uploading {name}...")

    def _on_upload_cached(self, digest: str, server_name: str) -> None:
        self.settings.setdefault("upload_cache", {})[digest] = server_name
        self._save_timer.start()

    def _accept_run(self, prompt_id: str, built, record: RunRecord) -> None:
        """The server took a run. Put it at the back of the queue and show it.

        Nothing here assumes it is the one about to execute: it is only the head of an
        empty queue that starts immediately, and even then the server's own
        `execution_start` is what says so.
        """
        self.submitting = False
        first = not self.runs
        self.runs.add(runqueue.QueuedRun(prompt_id=prompt_id, record=record, built=built))

        self.history_store.append(record)
        self.history_panel.add_record(record)

        if first:
            self.fetch_requested_for = None
            self._reset_run_meters()
            self.progress_bar.setRange(0, 100)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("queued")
            self.stage_label.setText("queued")
            self._elapsed_timer.start()
        self._refresh_derived()

    def _reset_run_meters(self) -> None:
        """Zero the progress state, for a run that is about to start."""
        self.total_steps = 0
        self.eta.reset()
        # The two step sources restart independently at every stage boundary, so each gets
        # its own tracker rather than reading the other's rewinds as its own.
        expected = graph_builder.clamp_steps(self.state.steps)
        self.stages.reset(expected)
        self.preview_stages.reset(expected)
        self.stage = ""

    def _on_submitted(self, prompt_id: str, built) -> None:
        state = self.state
        record = RunRecord(
            prompt_id=prompt_id,
            submitted_at=datetime.now().isoformat(timespec="seconds"),
            prompt_text=state.prompt_text,
            prompt_sections=dict(state.prompt_sections),
            aspect_ratio=state.aspect_ratio,
            megapixels=state.megapixels,
            width=getattr(built, "width", 0),
            height=getattr(built, "height", 0),
            duration_seconds=state.duration_seconds,
            frames=getattr(built, "frames", 0),
            seed=state.seed,
            steps=state.steps,
            sampler_name=state.sampler_name,
            scheduler=state.scheduler,
            schedule=state.schedule,
            upscale_method=state.upscale_method,
            shift_video=state.shift_video,
            ref_image_size=state.ref_image_size,
            refs=history.refs_snapshot(state.refs, built.tags),
            graph=built.graph,
        )
        self._accept_run(prompt_id, built, record)
        waiting = len(self.runs) - 1
        self._log(f"Submitted {prompt_id}"
                  + (f" ({waiting} already queued ahead of it)" if waiting else ""))
        if getattr(built, "pruned", None):
            self._log("Left out of the graph, nothing consumes them: "
                      + ", ".join(built.pruned))

    def _on_requeued(self, prompt_id: str, source: RunRecord) -> None:
        """A stored graph was resubmitted; the new record describes that graph, not the editor."""
        record = RunRecord(
            prompt_id=prompt_id,
            submitted_at=datetime.now().isoformat(timespec="seconds"),
            prompt_text=source.prompt_text,
            prompt_sections=dict(source.prompt_sections or {}),
            aspect_ratio=source.aspect_ratio,
            megapixels=source.megapixels,
            width=source.width,
            height=source.height,
            duration_seconds=source.duration_seconds,
            frames=source.frames,
            seed=source.seed,
            steps=source.steps,
            sampler_name=source.sampler_name,
            scheduler=source.scheduler,
            schedule=source.schedule,
            upscale_method=source.upscale_method,
            shift_video=source.shift_video,
            ref_image_size=source.ref_image_size,
            refs=list(source.refs or []),
            graph=source.graph,
        )
        self._accept_run(prompt_id, None, record)
        self._log(f"Re-queued as {prompt_id}")

    def _on_submit_failed(self, failure: JobFailure) -> None:
        """The server refused it, so nothing was queued and nothing outstanding changed."""
        self.submitting = False
        self._settle()
        self._log(f"Submit failed: {failure.message}")
        if failure.detail:
            self._log(failure.detail)

        if failure.node_errors:
            self._highlight_node_errors(failure.node_errors, failure.labels)

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Warning)
        box.setWindowTitle("The run was not queued")
        box.setText(failure.message)
        if failure.detail:
            box.setDetailedText(failure.detail)
        box.exec()

    def _highlight_node_errors(self, node_errors: dict, labels: dict) -> None:
        """Show a rejection against the reference row that caused it.

        The builder's labels are the only link from a node id back to a row, which is why
        JobFailure carries them -- by the time this runs the built graph is gone.
        """
        for node_id, info in (node_errors or {}).items():
            label = (labels or {}).get(node_id, "")
            if not label:
                continue
            messages = [e.get("message", "") for e in info.get("errors") or []]
            text = "; ".join(m for m in messages if m)
            for widget in self.ref_panel.all_row_widgets():
                if widget.row.display_name and widget.row.display_name in label:
                    widget.show_server_error(text or "rejected by the server")

    # ------------------------------------------------------------------- run progress

    def _is_ours(self, prompt_id: str) -> bool:
        # ComfyUI broadcasts some events to every client, so a browser tab on the same
        # server would otherwise drive this window's progress bar. Any outstanding run of
        # ours, not just the head: the server executes them one at a time, and taking its
        # word for which one beats assuming.
        return bool(prompt_id) and self.runs.holds(prompt_id)

    def _on_execution_start(self, prompt_id: str) -> None:
        run = self.runs.find(prompt_id) if prompt_id else None
        if run is None:
            return
        run.started = True
        run.started_at = time.monotonic()
        if run.record is not None and run.record.status == history.STATUS_QUEUED:
            run.record.status = history.STATUS_RUNNING
            self.history_panel.update_record(run.record)

        # The meters belong to whichever run is executing, so they are zeroed here rather
        # than at submission -- by which time a queued batch would have shared one set.
        self.fetch_requested_for = None
        self._reset_run_meters()
        self._elapsed_timer.start()
        self.stage_label.setText("starting")
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("starting")

    def _on_execution_cached(self, prompt_id: str, nodes: list) -> None:
        if not self._is_ours(prompt_id):
            return
        self._log(f"{len(nodes)} node(s) served from cache")

    def _on_executing(self, prompt_id: str, node_id) -> None:
        if not self._is_ours(prompt_id):
            return
        if node_id is None:
            return
        stage = self._stage_name(str(node_id))
        self.stage_label.setText(stage)

        # Sampling is over once the graph moves on, so freeze the preview on its last
        # step rather than leaving it looping through decode and save. Keyed on the node's
        # class rather than its id, so it holds for any workflow.
        if self.eta.step and not self._is_sampler(str(node_id)):
            self.player.end_live_preview()

    def _is_sampler(self, node_id: str) -> bool:
        graph = getattr(self.active_built, "graph", None) or {}
        return (graph.get(node_id) or {}).get("class_type", "") in SAMPLER_CLASSES

    #: role -> what to say while that node is executing. Anything not here falls back to
    #: the build's own node label, then to the bare id.
    STAGE_NAMES = {
        "loadmodel": "loading the diffusion model",
        "loadclip": "loading the text encoder",
        "reference": "encoding references",
        "sampleradvanced": "sampling",
        "progressivesampler": "sampling",
        "imagedecode": "decoding video",
        "audiodecode": "decoding audio",
        "vidcombine": "saving",
    }

    def _stage_name(self, node_id: str) -> str:
        for role, text in self.STAGE_NAMES.items():
            if self.roles.optional(role) == node_id:
                return text
        labels = getattr(self.active_built, "labels", {}) or {}
        return labels.get(node_id, f"node {node_id}")

    def _on_progress(self, prompt_id: str, node_id: str, value: int, maximum: int) -> None:
        if not self._is_ours(prompt_id) or maximum <= 0:
            return
        # The staged sampler counts from one again at every resolution stage; the tracker
        # is what turns those passes back into a single walk across the run.
        position, total = self.stages.note(value, maximum)
        self.total_steps = total
        self.stage = self._stage_name(node_id)
        self.eta.note_step(position, total, time.monotonic())

        self.progress_bar.setRange(0, total)
        self.progress_bar.setValue(position)
        self._refresh_eta()

    def _refresh_eta(self) -> None:
        """The progress caption: stage, position, and how long is left."""
        self.progress_bar.setFormat(self.eta.describe(self.stage))

    def _on_preview_clip(self, clip) -> None:
        """A sampler step's live preview from the Model Preview Override node."""
        if self.active_prompt_id is None or not self.settings.get("show_previews"):
            return
        self.player.show_live_preview(clip)

        # The node measures its own step time at the sampler, which beats anything
        # inferred from the arrival of progress messages.
        if clip.avg_step_ms:
            self.eta.note_measured_step_ms(clip.avg_step_ms)
            if clip.total:
                position, total = self.preview_stages.note(clip.step, clip.total)
                self.eta.note_step(position, total, time.monotonic())
                self.progress_bar.setRange(0, total)
                self.progress_bar.setValue(position)
            self._refresh_eta()

    def _on_preview_frame(self, payload: bytes) -> None:
        if not self.settings.get("show_previews"):
            return
        image = QImage()
        if not image.loadFromData(payload):
            return
        self.preview_label.setPixmap(QPixmap.fromImage(image).scaled(
            self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.preview_label.show()

    def _tick_elapsed(self) -> None:
        if self.run_started_at is None:
            return
        elapsed = int(time.monotonic() - self.run_started_at)
        self.elapsed_label.setText(f"{elapsed // 60}:{elapsed % 60:02d}")

    # ------------------------------------------------------------------------ results

    def _on_executed(self, prompt_id: str, node_id: str, output: dict) -> None:
        if not self._is_ours(prompt_id) or node_id != self.roles.vidcombine:
            return
        refs = comfy_http.outputs_from_node_output(output)
        if not refs or self.fetch_requested_for == prompt_id:
            return
        self.fetch_requested_for = prompt_id
        self.stage_label.setText("downloading")
        self._fetch_requested.emit(prompt_id, refs[0], self._output_dir())

    def _on_execution_success(self, prompt_id: str) -> None:
        if not self._is_ours(prompt_id):
            return
        record = self.active_record
        downloading = self.fetch_requested_for == prompt_id

        # Off the queue here rather than when its video lands: the server has already
        # moved on to the next one, and leaving this run at the head would point the
        # progress bar, the stage names and the elapsed clock at a run that is over.
        self._end_run(prompt_id, history.STATUS_SUCCESS,
                      "done" if downloading and record and record.local_path else "")

        if not downloading:
            # The `executed` payload can be missed if the socket blipped; /history has it.
            self.fetch_requested_for = prompt_id
            self._fetch_from_history_requested.emit(
                prompt_id, self._output_dir())

    def _on_execution_error(self, prompt_id: str, data: dict) -> None:
        run = self.runs.find(prompt_id) if prompt_id else None
        if run is None:
            return
        labels = getattr(run.built, "labels", {}) or {}
        message = comfy_http.format_execution_error(data, labels)

        self._end_run(prompt_id, history.STATUS_FAILED, "failed", error=message)
        self._log(message)

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle("The run failed")
        box.setText(message)
        traceback = data.get("traceback")
        if traceback:
            box.setDetailedText("\n".join(traceback) if isinstance(traceback, list) else str(traceback))
        box.exec()

    def _on_execution_interrupted(self, prompt_id: str, _data: dict) -> None:
        if not self._is_ours(prompt_id):
            return
        self._end_run(prompt_id, history.STATUS_CANCELLED, "cancelled")
        self._log("Run cancelled"
                  + (f"; {len(self.runs)} still queued" if self.runs else ""))

    def _on_download_progress(self, done: int, total: int) -> None:
        if total:
            self.progress_bar.setRange(0, total)
            self.progress_bar.setValue(done)
            self.progress_bar.setFormat("downloading %p%")

    def _on_downloaded(self, prompt_id: str, path: str) -> None:
        run = self.runs.find(prompt_id)
        record = self.history_store.find(prompt_id) or (run.record if run else None)
        if record:
            record.local_path = self.history_store.relative(Path(path))
            if record.status in (history.STATUS_RUNNING, history.STATUS_QUEUED):
                # The download can outrace execution_success, in which case this is where
                # the run is marked finished.
                record.status = history.STATUS_SUCCESS
                record.finished_at = datetime.now().isoformat(timespec="seconds")
            if record.elapsed_s is None and run is not None and run.started_at:
                record.elapsed_s = time.monotonic() - run.started_at
            self.history_store.update(record)
            self.history_panel.update_record(record)

        # Off the queue if it is still on it, so the count is right before the tab decision.
        self._end_run(prompt_id, None, "")
        # Quietly while a batch is still working through: a result that raises the Results
        # tab every few minutes takes the frame away from whatever is being looked at, and
        # the run that is still going is the one with something to show.
        alone = not self.runs
        self.player.load(path, autoplay=alone, raise_tab=alone)
        self._log(f"Saved {path}")
        self._settle("done")

    def _on_download_failed(self, prompt_id: str, message: str) -> None:
        self._log(f"Could not fetch the result: {message}")

        # Record it even though the run itself succeeded, so history does not show a
        # clean success with no video and no explanation.
        record = self.history_store.find(prompt_id)
        if record:
            record.error = f"The run finished but its video could not be fetched: {message}"
            self.history_store.update(record)
            self.history_panel.update_record(record)

        self._end_run(prompt_id, None, "done (not downloaded)")
        QMessageBox.warning(
            self, "Result not downloaded",
            f"The run finished but its video could not be fetched:\n\n{message}")

    def _end_run(self, prompt_id: str, status: str | None, label: str,
                 error: str | None = None) -> None:
        """Take one run off the queue and settle the UI on whatever is left behind it."""
        run = self.runs.remove(prompt_id)
        record = (run.record if run is not None else None) \
            or self.history_store.find(prompt_id)

        if record is not None and status:
            record.status = status
            record.finished_at = datetime.now().isoformat(timespec="seconds")
            if error:
                record.error = error
            if record.elapsed_s is None and run is not None and run.started_at:
                record.elapsed_s = time.monotonic() - run.started_at
            self.history_store.update(record)
            self.history_panel.update_record(record)

        # This run's preview has nothing more to show. Stopped rather than cleared, so the
        # last step stays up until the next run replaces it.
        self.player.end_live_preview()
        self.preview_label.hide()
        self._settle(label)

    def _settle(self, label: str = "") -> None:
        """Put the run bar into the state the queue now calls for.

        Idle when nothing is left; otherwise back to "queued", because the next run's own
        `execution_start` is what zeroes the meters and starts describing it. Nothing here
        touches the live preview: this is also the path a *failed submission* takes, and
        whatever is still running has every right to keep animating.
        """
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)

        if self.runs:
            queued = len(self.runs)
            self.progress_bar.setFormat(
                f"{label} - {queued} queued" if label else f"{queued} queued")
            self.stage_label.setText("queued")
        else:
            self._elapsed_timer.stop()
            self.progress_bar.setFormat(label or "idle")
            self.stage_label.setText("")

        self._refresh_derived()

    # ------------------------------------------------------------------------ history

    def _on_replay(self, record: RunRecord) -> None:
        path = record.video_path()
        if path:
            self.player.load(str(path))
        else:
            QMessageBox.information(
                self, "Not available",
                "The saved copy of this run's video is no longer on disk.")

    def _on_requeue(self, record: RunRecord) -> None:
        """Resubmit a stored graph. It joins the back of the queue like anything else."""
        if not record.graph or self.submitting or self.posing:
            return
        self._log(f"Re-queueing {record.prompt_id}")
        if not self.runs:
            self.stage_label.setText("re-queueing...")
        self.submitting = True
        self.queue_button.setEnabled(False)
        self._requeue_requested.emit(record)

    def _on_restore(self, record: RunRecord) -> None:
        self.state.prompt_sections = (
            prompt_mod.normalise(record.prompt_sections)
            if record.prompt_sections
            # A record from before the prompt was split keeps only the joined string.
            else prompt_mod.parse(record.prompt_text)
        )
        self.state.aspect_ratio = record.aspect_ratio or self.state.aspect_ratio
        self.state.megapixels = record.megapixels or self.state.megapixels
        self.state.duration_seconds = record.duration_seconds or self.state.duration_seconds
        self.state.seed = record.seed or self.state.seed
        # All absent from records written before they were exposed, so a falsy value
        # means "unknown" and the current setting stands.
        self.state.steps = record.steps or self.state.steps
        self.state.sampler_name = record.sampler_name or self.state.sampler_name
        self.state.scheduler = record.scheduler or self.state.scheduler
        self.state.schedule = record.schedule or self.state.schedule
        self.state.upscale_method = record.upscale_method or self.state.upscale_method
        self.state.shift_video = record.shift_video or self.state.shift_video
        self.state.ref_image_size = record.ref_image_size or self.state.ref_image_size

        restored = RefSet()
        for entry in record.refs or []:
            name = entry.get("comfy_name")
            if not name:
                continue
            # `use_pose` is recorded but deliberately not restored: what was uploaded for
            # a posed row *is* the skeleton clip, so the restored row already names it.
            # Turning the flag back on would pose the pose.
            row = RefRow(kind=entry["kind"], comfy_name=name)
            target = restored.list_for(row.kind)
            if len(target) < refs_mod.KIND_LIMITS[row.kind]:
                target.append(row)
        if restored.all_rows():
            self.state.refs = restored

        self._load_state_into_ui()
        self._refresh_derived()
        self._save_timer.start()

    # ----------------------------------------------------------------------- verifying

    def _on_check_clicked(self) -> None:
        if not self.reachable:
            QMessageBox.information(self, "Not connected", "The ComfyUI server is not reachable.")
            return
        self.params_panel.check_button.setEnabled(False)
        self.params_panel.check_button.setText("Checking...")
        self._check_requested.emit(self._collect_state())

    def _on_check_finished(self, ok: bool, report: str) -> None:
        self.params_panel.check_button.setEnabled(True)
        self.params_panel.check_button.setText("Check the workflow")
        self._log(report)

        box = QMessageBox(self)
        box.setIcon(QMessageBox.Information if ok else QMessageBox.Warning)
        box.setWindowTitle("Workflow checked" if ok else "Workflow problems")
        box.setText(
            "The workflow satisfies the role contract and validates against the server."
            if ok else
            "The workflow has problems that will stop or spoil a run."
        )
        box.setDetailedText(report)
        box.exec()

    # ------------------------------------------------------------------------ plumbing

    def _log(self, message: str) -> None:
        stamped = f"{datetime.now():%H:%M:%S}  {message}"
        self.log_lines.append(stamped)
        del self.log_lines[:-500]
        log.info(message)

    def _show_log(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("HARMON3 log")
        dialog.resize(760, 420)

        layout = QVBoxLayout(dialog)
        view = QPlainTextEdit("\n".join(self.log_lines))
        view.setReadOnly(True)
        layout.addWidget(view)

        buttons = QDialogButtonBox(QDialogButtonBox.Close)
        copy_button = buttons.addButton("Copy all", QDialogButtonBox.ActionRole)
        copy_button.clicked.connect(lambda: _copy_to_clipboard("\n".join(self.log_lines)))
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def _on_previews_toggled(self, enabled: bool) -> None:
        self.settings["show_previews"] = enabled
        # Told to the websocket thread so a disabled preview costs no decoding at all.
        self.ws_client.previews_enabled = enabled
        if not enabled:
            self.preview_label.hide()
            self.player.live_preview.clear()
        self._save_timer.start()

    def _on_sage_toggled(self, enabled: bool) -> None:
        """Sage Attention lives on the graph, not the interface, so it goes into the state.

        Applied on the next run rather than immediately: it is a property of the graph
        being built, and there is nothing to reconnect or move on disk.
        """
        self.state.sage_attention = enabled
        self.settings["sage_attention"] = enabled
        self._log(f"Sage Attention {'on' if enabled else 'off'} for the next run")
        self._save_timer.start()

    def _on_pose_model_changed(self, model: str) -> None:
        self.settings["pose_model"] = model
        self._forget_pose_clips(f"Pose model set to {model}")

    def _on_pose_style_changed(self, pose_style: str) -> None:
        self.settings["pose_style"] = pose_style
        self._forget_pose_clips(f"Pose style set to {pose_style}")

    def _on_pose_cache_clear(self) -> None:
        """Delete every rendered skeleton clip, once the user has said so out loud.

        Asked rather than assumed: this is minutes of GPU time on the floor, and unlike
        every other button in this panel it cannot be undone by putting the setting back.
        Refused outright mid-pass, where the renderer is holding the files being deleted.
        """
        if self.posing:
            QMessageBox.information(
                self, "Pose pass running",
                "Wait for the pose pass to finish, or cancel it, before clearing.")
            return

        count, size = pose_mod.cache_usage()
        if not count:
            self.settings_panel.refresh_pose_cache()
            return

        confirmed = QMessageBox.question(
            self, "Clear pose clips",
            f"Delete {count} rendered skeleton clip(s), freeing {size / (1 << 20):.1f} MB?"
            "\n\nThey are only a cache. Anything still needed is rendered again on the "
            "next run, which takes time but loses nothing.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if confirmed != QMessageBox.Yes:
            return

        removed, freed, failures = pose_mod.clear_cache()
        pose_mod.forget_all(self._collect_state())
        self._refresh_derived()
        self.settings_panel.refresh_pose_cache()

        self._log(f"Cleared {removed} pose clip(s), {freed / (1 << 20):.1f} MB")
        if failures:
            # A clip open elsewhere cannot be deleted on Windows, and saying nothing would
            # leave the count quietly disagreeing with the button that was just pressed.
            self._log(f"{len(failures)} could not be deleted: {failures[0]}")
            self.settings_panel.set_status(
                f"{removed} cleared; {len(failures)} still in use.", "warn")
        else:
            self.settings_panel.set_status(
                f"Cleared {removed} clip(s), {freed / (1 << 20):.1f} MB.", "ok")

    def _on_export_bundle(self) -> None:
        """Write out what the node is about to be given, from the editor as it stands.

        Built through the *same* three steps the submit path takes -- snapshot, pose
        substitution, scale substitution -- rather than a second arrangement that could
        drift from it. A bundle that showed something the run would not send would be
        worse than no bundle, because it would be believed.
        """
        state = self._collect_state()
        notes = []

        # Whatever is already rendered. A posed row with no current clip is not rendered
        # here: that is a pass with a progress bar and a stop button, and a diagnostic
        # should not quietly start one.
        poses = self._rendered_poses()
        for row in self._posed_rows():
            if row.uid not in poses:
                notes.append(
                    f"{row.display_name}: its clip has not been rendered yet, so the "
                    "original file is shown instead - queue once, or click its Pose "
                    "thumbnail, and export again")

        try:
            request = JobRequest.snapshot(
                state, dict(self.settings.get("upload_cache", {})),
                self._output_dir())
            pose_mod.swap_in(request.state, poses)
            scaling_mod.swap_in(request.state, self._rescaled_images())
            result = bundle_mod.export(state, request.state, notes=notes)
        # ValueError alongside the obvious two, for a state whose aspect ratio is not one
        # this build knows -- a diagnostic is the last place to hand back a traceback.
        except (bundle_mod.BundleError, OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not export", str(exc))
            self._log(f"Reference export failed: {exc}")
            return

        self._log(f"Exported {len(result.copied)} reference file(s) to {result.directory}")
        for problem in result.missing:
            self._log(f"  not exported - {problem}")

        self.settings_panel.set_status(
            f"Wrote {len(result.copied)} file(s) to {result.directory.name}."
            + (f" {len(result.missing)} could not be exported - see the log."
               if result.missing else ""),
            "warn" if result.missing else "ok")

    def _forget_pose_clips(self, message: str) -> None:
        """Let go of the clips rendered under the previous pose settings.

        The pointers have to be dropped by hand. ``forget_stale`` only compares the
        section, so a row would go on offering a skeleton drawn the old way -- while the
        cache itself is fine, because everything that changes the drawing is in the key.
        The next pass either finds the new clip already rendered or renders it, and the old
        one is still there if the setting is put back.
        """
        pose_mod.forget_all(self._collect_state())
        self._log(message)
        self._refresh_derived()
        self._save_timer.start()

    def _on_settings_applied(self, staged: dict) -> None:
        messages = []

        if not self._apply_scenes_dir(staged.get("scenes_dir", "")):
            return
        messages.append("Scenes folder updated.")

        server = comfy_http.normalise_base_url(staged.get("server_url", ""))
        if server != self.settings.get("server_url"):
            self.settings["server_url"] = server
            self._server_changed.emit(server)
            self.ws_client.set_base_url(server)
            self._poll_reachability()
            messages.append(f"Reconnecting to {server}.")

        self.settings["server_output_dir"] = staged.get("server_output_dir", "")

        prefix = config.clean_filename_prefix(staged.get("filename_prefix", ""))
        if prefix != self.settings.get("filename_prefix", ""):
            self.settings["filename_prefix"] = prefix
            self.state.filename_prefix = prefix
            messages.append(
                f"Results will be filed under {prefix}." if prefix
                else "Results go wherever the workflow's own output node says.")

        self._save_settings()
        self.settings_panel.load(self.settings)
        self.settings_panel.set_status(" ".join(messages), "ok")
        self._log("Settings applied")

    def _apply_scenes_dir(self, configured: str) -> bool:
        """Point the catalogue at a new folder, offering to bring the scenes along."""
        new_dir = config.resolve_scenes_dir(configured)
        old_dir = self.scene_store.dir
        if new_dir == old_dir:
            return True

        probe = SceneStore(new_dir)
        problem = probe.writable()
        if problem:
            QMessageBox.warning(
                self, "Cannot use that folder",
                f"{new_dir}\n\nHARMON3 {problem}.")
            return False

        pending = self.scene_store.count_files()
        if pending:
            box = QMessageBox(self)
            box.setIcon(QMessageBox.Question)
            box.setWindowTitle("Move scenes?")
            box.setText(f"{pending} scene file(s) are in the current folder.")
            box.setInformativeText(
                f"From:\t{old_dir}\nTo:\t{new_dir}\n\n"
                "Move them, or leave them behind and start fresh in the new folder?")
            move = box.addButton("Move them", QMessageBox.AcceptRole)
            leave = box.addButton("Leave them", QMessageBox.DestructiveRole)
            box.addButton(QMessageBox.Cancel)
            box.setDefaultButton(move)
            box.exec()

            if box.clickedButton() is move:
                moved, problems = self.scene_store.move_to(new_dir)
                self._log(f"Moved {moved} scene file(s) to {new_dir}")
                if problems:
                    QMessageBox.warning(
                        self, "Some scenes were not moved", "\n".join(problems))
            elif box.clickedButton() is leave:
                self.scene_store.dir = new_dir
            else:
                return False
        else:
            self.scene_store.dir = new_dir

        self.settings["scenes_dir"] = configured
        self.scene_store.load_all()
        # The scene that was loaded may not exist in the new folder.
        self.current_scene = self.scene_store.find(self.settings.get("current_scene", ""))
        self.settings["current_scene"] = self.current_scene.name if self.current_scene else ""
        self._refresh_scenes()
        return True

    def _save_settings(self) -> None:
        state = self._collect_state()
        self.settings = settings_mod.capture_from_state(self.settings, state)
        self.settings["randomize_seed"] = self.params_panel.randomize
        self.settings["last_browse_dirs"] = self.ref_panel.last_dirs()
        settings_mod.save_settings(self.settings)

    def _restore_geometry(self) -> None:
        store = QSettings(str(config.UI_STATE_PATH), QSettings.IniFormat)
        geometry = store.value("window/geometry")
        if geometry:
            self.restoreGeometry(geometry)

        # Two levels of invalidation, because they fail differently. A version mismatch
        # makes restoreState say so and return False. A *pane set* mismatch does not:
        # a state naming a dock that no longer exists restores as a hole and reports
        # success, so the schema is checked separately and falls back to the default.
        state = store.value("window/state")
        schema_ok = str(store.value("layout/schema", "")) == docks.LAYOUT_SCHEMA
        if not (state and schema_ok and self.restoreState(state, docks.LAYOUT_VERSION)):
            docks.apply_default_layout(self)
        # Written by every version before the panes became dockable.
        store.remove("splitter")

        # restoreState brings toolbar visibility back with it, and a notice that was up at
        # shutdown is stale by definition. The reachability poll queued in __init__ puts it
        # straight back if the server really is down.
        self.banner_bar.hide()

        index = store.value("other/tab")
        if index is not None and 0 <= int(index) < self.other_tabs.count():
            self.other_tabs.setCurrentIndex(int(index))

        # A layout preference, so it lives here beside the panes rather than in
        # settings.json -- which is scene data, captured into every scene that is saved.
        # QSettings hands ini values back as strings.
        helpers = store.value("prompt/helpers")
        self.prompt_editor.set_helpers_visible(
            True if helpers is None else str(helpers).lower() in ("true", "1"))

    def _save_geometry(self) -> None:
        store = QSettings(str(config.UI_STATE_PATH), QSettings.IniFormat)
        store.setValue("window/geometry", self.saveGeometry())
        store.setValue("layout/schema", docks.LAYOUT_SCHEMA)
        store.setValue("window/state", self.saveState(docks.LAYOUT_VERSION))
        # saveState knows about tabbed *panes*, not about a tab widget inside one.
        store.setValue("other/tab", self.other_tabs.currentIndex())
        store.setValue("prompt/helpers", self.prompt_editor.helpers_visible())
        store.sync()

    def closeEvent(self, event) -> None:
        self._save_timer.stop()
        self._save_settings()
        self._save_geometry()

        self.player.clear()

        self.ws_client.stop()
        self.ws_thread.quit()
        self.ws_thread.wait(3000)

        # The pose worker checks this between frames, so it stops within one -- which is
        # why this one can afford to be asked nicely and then waited on.
        self.pose_runner.cancel()
        self.pose_thread.quit()
        if not self.pose_thread.wait(3000):
            log.info("Pose thread still busy at shutdown; exiting anyway")

        # Queued, never blocking: the job thread may be part-way through an upload or a
        # download with a five-minute timeout, and a blocking call would freeze the
        # window until it finished. If the wait times out the session is left to the
        # process teardown, which is fine at this point.
        QMetaObject.invokeMethod(self.job_runner, "shutdown", Qt.QueuedConnection)
        self.job_thread.quit()
        if not self.job_thread.wait(3000):
            log.info("Job thread still busy at shutdown; exiting anyway")

        super().closeEvent(event)


def _print_prompt(text: str) -> None:
    """Echo the combined prompt to the console, for checking what the model receives.

    Goes to both stdout and the log so it shows up whether the app was started from a
    terminal (run_debug.bat) or not.
    """
    banner = "=" * 60
    print(f"\n{banner}\nPROMPT SENT TO THE MODEL\n{banner}\n{text}\n{banner}\n", flush=True)
    log.info("Prompt sent to the model:\n%s", text)


def _copy_to_clipboard(text: str) -> None:
    from PySide6.QtGui import QGuiApplication
    QGuiApplication.clipboard().setText(text)
