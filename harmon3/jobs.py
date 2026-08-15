"""Job worker: upload references, build and validate the graph, submit, download results.

Everything here runs on a dedicated QThread with a thread-confined requests.Session.
Slots are invoked through queued connections and never return values; results come back
as signals. Nothing in this module touches a widget.
"""

from __future__ import annotations

import copy
import logging
import uuid
from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot

from . import comfy_http, config, graph_builder, mathmirror, validator
from .comfy_http import ComfyClient, ComfyError, ComfyUnreachable, OutputRef
from .graph_builder import BuildState, BuiltGraph  # noqa: F401  (BuiltGraph documents the submitted signal)

log = logging.getLogger(__name__)

@dataclass
class JobRequest:
    """A self-contained snapshot handed to the worker thread.

    ``state`` and ``upload_cache`` must be copies, never the UI's live objects: the
    worker rewrites reference filenames and cache entries while the user keeps typing,
    and sharing them would let a mid-flight edit change the graph being submitted (or
    corrupt the settings file being serialised at the same moment).
    """

    state: BuildState
    #: sha256 -> server filename, seeded from the UI's cache so unchanged refs skip upload
    upload_cache: dict = field(default_factory=dict)
    server_output_dir: str = ""

    @classmethod
    def snapshot(cls, state: BuildState, upload_cache: dict, server_output_dir: str = ""):
        return cls(
            state=copy.deepcopy(state),
            upload_cache=dict(upload_cache),
            server_output_dir=server_output_dir,
        )


@dataclass
class JobFailure:
    message: str
    detail: str = ""
    node_errors: dict = field(default_factory=dict)
    unreachable: bool = False
    #: node id -> human label, so a rejection can be shown against the row that caused it
    labels: dict = field(default_factory=dict)


class JobRunner(QObject):
    """Owns all outbound HTTP.

    One *submission* at a time -- references upload serially over one session -- but any
    number of submitted runs can be outstanding on the server at once. Which of them the
    server is working on is not this object's problem: it hands prompts over and reports
    what came back. The window's ``runqueue`` is what tracks the rest.
    """

    reachability_changed = Signal(bool, str)          # reachable, detail
    #: Where the connected server writes results, when that is knowable and readable from
    #: here. Empty when it is not, which is the normal case for a remote ComfyUI.
    output_dir_found = Signal(str)
    # `object`, not dict/list: node schemas contain uint64 bounds that Qt's QVariantMap
    # conversion cannot represent, and it raises OverflowError rather than delivering.
    object_info_ready = Signal(object)                # class -> schema
    model_problems = Signal(object)                   # preflight failures

    upload_started = Signal(str)                      # display name
    upload_cache_updated = Signal(str, str)           # sha256, server filename

    submitted = Signal(str, object)                   # prompt_id, BuiltGraph
    requeued = Signal(str, object)                    # prompt_id, the source RunRecord
    submit_failed = Signal(object)                    # JobFailure

    download_progress = Signal(int, int)              # done, total
    downloaded = Signal(str, str)                     # prompt_id, local path
    download_failed = Signal(str, str)                # prompt_id, message

    check_finished = Signal(bool, str)                # ok, report
    server_refs_verified = Signal(object)             # [(row uid, exists)]

    def __init__(self, base_url: str, client_id: str, parent: QObject | None = None):
        super().__init__(parent)
        self.client = ComfyClient(base_url)
        self.client_id = client_id
        self._object_info: dict[str, dict | None] = {}

    # -- connection ----------------------------------------------------------------

    @Slot(str)
    def set_server(self, base_url: str) -> None:
        self.client.set_base_url(base_url)
        self._object_info = {}
        self.check_reachable()

    @Slot()
    def check_reachable(self) -> None:
        try:
            stats = self.client.system_stats()
        except (ComfyUnreachable, ComfyError) as exc:
            self.reachability_changed.emit(False, str(exc))
            return

        system = stats.get("system") or {}
        detail = f"ComfyUI {system.get('comfyui_version', '?')}"
        devices = stats.get("devices") or []
        if devices:
            detail += f" - {devices[0].get('name', '?').split(':')[0]}"
        self.reachability_changed.emit(True, detail)

        # Checked for existence here rather than by the UI: this is the thread that is
        # allowed to touch the filesystem, and a path the server reports is only useful
        # if this machine can actually read it.
        detected = comfy_http.output_dir_from_stats(stats)
        self.output_dir_found.emit(
            detected if detected and Path(detected).is_dir() else "")

        if not self._object_info:
            self.refresh_object_info()

    @Slot()
    def refresh_object_info(self) -> None:
        try:
            workflow = config.load_workflow()
        except (FileNotFoundError, ValueError) as exc:
            self.model_problems.emit([str(exc)])
            return

        try:
            self._object_info = self.client.object_info_many(
                config.classes_for(workflow.graph), refresh=True)
        except (ComfyUnreachable, ComfyError) as exc:
            self.reachability_changed.emit(False, str(exc))
            return

        self.object_info_ready.emit(dict(self._object_info))
        self.model_problems.emit(validator.model_preflight(
            workflow.graph, self._object_info, workflow.roles))

    @Slot(object)
    def verify_server_refs(self, entries) -> None:
        """Confirm that server-file references really exist in ComfyUI's input folder.

        The shipped workflow names two images that may not be present on whichever
        machine runs ComfyUI. Checking up front turns a cryptic 400 at submit time into
        a row-level error the user can act on.
        """
        results = []
        for uid, filename, subfolder in entries or []:
            try:
                exists = self.client.file_exists(filename, subfolder, "input")
            except (ComfyUnreachable, ComfyError):
                continue  # unknown, not missing - leave it unchecked and retry later
            results.append((uid, exists))
        if results:
            self.server_refs_verified.emit(results)

    # -- submission ----------------------------------------------------------------

    @Slot(object)
    def submit_job(self, request: JobRequest) -> None:
        try:
            self._resolve_uploads(request)
        except (ComfyUnreachable, ComfyError) as exc:
            self.submit_failed.emit(JobFailure(
                message=f"Preparing references failed: {exc}",
                unreachable=isinstance(exc, ComfyUnreachable),
            ))
            return

        problems = graph_builder.validate_state(request.state)
        if problems:
            self.submit_failed.emit(JobFailure(
                message="The job is not ready to run.", detail="\n".join(problems)))
            return

        workflow = config.load_workflow()
        built = graph_builder.build_graph(workflow.graph, request.state, workflow.roles)

        if self._object_info:
            report = validator.validate(built.graph, self._object_info, built.labels)
            if not report.ok:
                self.submit_failed.emit(JobFailure(
                    message="The workflow did not pass validation.",
                    detail="\n".join(report.errors)))
                return

        self._submit_built(built, request, retry_on_stale_upload=True)

    def _submit_built(self, built: BuiltGraph, request: JobRequest,
                      retry_on_stale_upload: bool) -> None:
        prompt_id = str(uuid.uuid4())
        try:
            self.client.submit(built.graph, self.client_id, prompt_id,
                               extra_data=graph_builder.extra_data_for(built.graph))
        except ComfyUnreachable as exc:
            self.submit_failed.emit(JobFailure(message=str(exc), unreachable=True))
            return
        except ComfyError as exc:
            # A file that vanished from ComfyUI's input dir makes a cached upload a lie.
            # Purge the cache and try once more before bothering the user.
            if retry_on_stale_upload and _is_stale_upload_error(exc):
                log.info("Stale upload detected; re-uploading references and retrying")
                request.upload_cache.clear()
                for row in request.state.refs.all_rows():
                    if row.needs_upload:
                        row.comfy_name = None
                try:
                    self._resolve_uploads(request)
                except (ComfyUnreachable, ComfyError) as upload_exc:
                    self.submit_failed.emit(JobFailure(message=f"Re-upload failed: {upload_exc}"))
                    return
                workflow = config.load_workflow()
                rebuilt = graph_builder.build_graph(
                    workflow.graph, request.state, workflow.roles)
                self._submit_built(rebuilt, request, retry_on_stale_upload=False)
                return

            detail = "\n".join(comfy_http.describe_node_errors(exc.node_errors, built.labels))
            self.submit_failed.emit(JobFailure(
                message=str(exc), detail=detail, node_errors=exc.node_errors,
                labels=dict(built.labels)))
            return

        self.submitted.emit(prompt_id, built)

    def _resolve_uploads(self, request: JobRequest) -> None:
        """Give every reference row a filename that exists in ComfyUI's input directory."""
        for row in request.state.refs.all_rows():
            if row.needs_upload and not row.comfy_name:
                row.comfy_name = self._upload(request, Path(row.local_path),
                                              row.display_name, row.mime_type)

    def _upload(self, request: JobRequest, path: Path, label: str, mime: str) -> str:
        if not path.is_file():
            raise ComfyError(f"{path} no longer exists")
        # A last guard against putting something undecodable in ComfyUI's input folder,
        # where it would sit until a run failed on it. The UI blocks these earlier; this
        # catches the file that was emptied or truncated after that check.
        if path.stat().st_size == 0:
            raise ComfyError(f"{label} is empty, so there is nothing to upload")

        # Always hash, even for a file uploaded before: it may have been edited in place,
        # and the hash is the only thing that can tell. The cache makes the repeat cheap.
        digest = comfy_http.sha256_of(path)
        cached = request.upload_cache.get(digest)
        if cached:
            return cached

        self.upload_started.emit(label)
        result = self.client.upload(
            path, comfy_http.content_addressed_name(path, digest), mime)

        request.upload_cache[digest] = result.reference
        self.upload_cache_updated.emit(digest, result.reference)
        return result.reference

    # -- results -------------------------------------------------------------------

    @Slot(str, object, str)
    def fetch_result(self, prompt_id: str, ref: OutputRef, server_output_dir: str) -> None:
        """Make the finished video available locally, preferring not to move it at all.

        When ComfyUI's output directory is readable from here the file is used exactly
        where the workflow's output node wrote it. Copying it into ``runs/videos`` would
        double the disk cost of every run for nothing: this app only ever reads a result,
        and ComfyUI's own output folder is the place a user already looks for one.

        The download stays the fallback, because a remote server's output path is a path
        on another machine. Nothing else changes for that case.
        """
        if server_output_dir:
            local = Path(server_output_dir) / ref.subfolder / ref.filename
            if local.is_file():
                self.downloaded.emit(prompt_id, str(local))
                return
            log.info("%s is not readable from here; falling back to /view", local)

        destination = config.VIDEO_CACHE_DIR / f"{prompt_id}{Path(ref.filename).suffix or '.mp4'}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.client.download(ref, destination, progress=self.download_progress.emit)
        except (ComfyUnreachable, ComfyError) as exc:
            self.download_failed.emit(prompt_id, str(exc))
            return
        self.downloaded.emit(prompt_id, str(destination))

    @Slot(str, str)
    def fetch_result_from_history(self, prompt_id: str, server_output_dir: str) -> None:
        """Fallback when the `executed` message was missed."""
        try:
            output_node = config.load_workflow().roles.vidcombine
            refs = self.client.outputs_from_history(prompt_id, output_node)
        except (FileNotFoundError, ValueError) as exc:
            self.download_failed.emit(prompt_id, str(exc))
            return
        except (ComfyUnreachable, ComfyError) as exc:
            self.download_failed.emit(prompt_id, str(exc))
            return
        if not refs:
            self.download_failed.emit(prompt_id, "the run produced no video output")
            return
        self.fetch_result(prompt_id, refs[0], server_output_dir)

    @Slot(object)
    def requeue(self, record) -> None:
        """Resubmit a stored graph verbatim under a fresh prompt id."""
        graph = record.graph
        new_id = str(uuid.uuid4())
        try:
            self.client.submit(graph, self.client_id, new_id,
                               extra_data=graph_builder.extra_data_for(graph))
        except ComfyUnreachable as exc:
            self.submit_failed.emit(JobFailure(message=str(exc), unreachable=True))
            return
        except ComfyError as exc:
            self.submit_failed.emit(JobFailure(
                message=str(exc),
                detail="\n".join(comfy_http.describe_node_errors(exc.node_errors)),
                node_errors=exc.node_errors))
            return
        self.requeued.emit(new_id, record)

    @Slot(str, bool)
    def interrupt(self, prompt_id: str = "", executing: bool = True) -> None:
        """Cancel one run.

        Deleting it from the queue is what cancels a run that has not begun; /interrupt is
        what stops one that has. Both are needed, and only ``executing`` decides whether
        the second is safe: /interrupt takes down whatever the server is working on, which
        on a shared server is somebody else's job until ours has actually started.
        """
        try:
            if prompt_id:
                self.client.cancel_queued(prompt_id)
            if executing:
                self.client.interrupt()
        except (ComfyUnreachable, ComfyError) as exc:
            log.info("Interrupt failed: %s", exc)

    # -- verification --------------------------------------------------------------

    @Slot(object)
    def check_workflow(self, state: BuildState) -> None:
        """Report anything wrong with the workflow the app is about to build against.

        This replaces the old geometry probe. That probe existed because the server
        computed width, height and length from nodes in the graph and this side mirrored
        the arithmetic, so the two could disagree; the workflow no longer carries those
        nodes, the app writes the numbers as literals, and there is nothing left to
        cross-check. What can still be wrong is the contract itself.
        """
        try:
            workflow = config.load_workflow()
        except (FileNotFoundError, ValueError) as exc:
            self.check_finished.emit(False, str(exc))
            return

        problems = graph_builder.geometry_warnings(workflow.graph, workflow.roles)
        built = graph_builder.build_graph(workflow.graph, state, workflow.roles)

        if self._object_info:
            report = validator.validate(built.graph, self._object_info, built.labels)
            problems.extend(report.errors)

        lines = [
            f"{len(workflow.roles.describe(workflow.graph))} roles bound, "
            f"{len(workflow.graph)} nodes in the workflow",
            f"{built.width}x{built.height}, {built.frames} frames "
            f"({mathmirror.true_seconds(built.frames):.2f}s at {config.FPS} fps)",
        ]
        if built.pruned:
            lines.append("pruned as unreachable: " + ", ".join(built.pruned))
        lines.extend(problems)
        self.check_finished.emit(not problems, "\n".join(lines))

    @Slot()
    def shutdown(self) -> None:
        self.client.close()


def _is_stale_upload_error(exc: ComfyError) -> bool:
    """True when the rejection looks like a reference file missing from the server."""
    for info in (exc.node_errors or {}).values():
        for error in info.get("errors") or []:
            if error.get("type") == "custom_validation_failed":
                return True
            text = f"{error.get('message', '')} {error.get('details', '')}".lower()
            if "invalid" in text and ("image" in text or "audio" in text or "video" in text):
                return True
    return False
