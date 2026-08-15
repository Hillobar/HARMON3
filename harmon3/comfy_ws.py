"""Persistent websocket connection to ComfyUI, republished as Qt signals.

Runs on its own QThread for the lifetime of the app and reconnects on its own. Blocking
recv with a short timeout keeps the stop flag responsive without polling the socket.

Every signal is emitted from the worker thread, so all connections to UI slots are
automatically queued.
"""

from __future__ import annotations

import json
import logging
import struct
import threading

from PySide6.QtCore import QObject, Signal

from . import preview
from .comfy_http import websocket_url

log = logging.getLogger(__name__)

RECV_TIMEOUT = 1.0
BACKOFF_START = 0.5
BACKOFF_MAX = 8.0

#: Binary frame header: 4-byte big-endian event type, then 4-byte big-endian image type.
EVENT_PREVIEW_IMAGE = 1
_HEADER = struct.Struct(">II")


class ComfyWsClient(QObject):
    """Consumes ComfyUI's /ws stream and re-emits the messages the app acts on."""

    connected = Signal(str)                    # session id
    disconnected = Signal(str)                 # reason
    queue_changed = Signal(int)                # queue_remaining

    # Payloads are declared `object` rather than dict/list: Qt would otherwise marshal
    # them through QVariantMap, which cannot represent the uint64 values ComfyUI puts in
    # its schemas and seeds, and raises OverflowError instead of delivering the signal.
    execution_start = Signal(str)                # prompt_id
    execution_cached = Signal(str, object)       # prompt_id, node ids
    executing = Signal(str, object)              # prompt_id, node_id or None
    progress = Signal(str, str, int, int)        # prompt_id, node_id, value, max
    executed = Signal(str, str, object)          # prompt_id, node_id, ui output
    execution_success = Signal(str)              # prompt_id
    execution_error = Signal(str, object)        # prompt_id, error payload
    execution_interrupted = Signal(str, object)  # prompt_id, payload

    preview_frame = Signal(bytes)              # raw PNG/JPEG bytes (ComfyUI's own stream)
    preview_clip = Signal(object)              # PreviewClip from the Model Preview Override node

    def __init__(self, base_url: str, client_id: str, parent: QObject | None = None):
        super().__init__(parent)
        self.base_url = base_url
        self.client_id = client_id
        #: Skips decoding entirely when the preview pane is off, so a run costs nothing.
        self.previews_enabled = False
        self._stop = threading.Event()
        self._socket = None
        self._socket_lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------------

    def run(self) -> None:
        """Connect-and-listen loop. Entry point for the worker thread."""
        import websocket

        backoff = BACKOFF_START
        while not self._stop.is_set():
            url = websocket_url(self.base_url, self.client_id)
            try:
                socket = websocket.create_connection(url, timeout=RECV_TIMEOUT)
            except Exception as exc:
                self.disconnected.emit(str(exc))
                if self._stop.wait(backoff):
                    break
                backoff = min(backoff * 2, BACKOFF_MAX)
                continue

            backoff = BACKOFF_START
            with self._socket_lock:
                self._socket = socket
            log.info("Websocket connected to %s", url)

            try:
                self._listen(socket)
            finally:
                with self._socket_lock:
                    self._socket = None
                try:
                    socket.close()
                except Exception:
                    pass

        self.disconnected.emit("stopped")

    def stop(self) -> None:
        """Ask the loop to exit and unblock a pending recv."""
        self._stop.set()
        with self._socket_lock:
            if self._socket is not None:
                try:
                    self._socket.close()
                except Exception:
                    pass

    def set_base_url(self, base_url: str) -> None:
        """Point at a different server; the loop reconnects on its next iteration."""
        if base_url == self.base_url:
            return
        self.base_url = base_url
        with self._socket_lock:
            if self._socket is not None:
                try:
                    self._socket.close()
                except Exception:
                    pass

    # -- receive loop --------------------------------------------------------------

    def _listen(self, socket) -> None:
        import websocket

        while not self._stop.is_set():
            try:
                opcode, frame = socket.recv_data()
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as exc:
                if not self._stop.is_set():
                    log.info("Websocket closed: %s", exc)
                    self.disconnected.emit(str(exc))
                return

            if opcode == websocket.ABNF.OPCODE_BINARY:
                self._handle_binary(frame)
            elif opcode == websocket.ABNF.OPCODE_TEXT:
                try:
                    message = json.loads(frame.decode("utf-8"))
                except (ValueError, UnicodeDecodeError):
                    continue
                self._handle_message(message)

    def _handle_binary(self, frame: bytes) -> None:
        # Always consumed, even when previews are hidden, so the socket cannot back up.
        if len(frame) < _HEADER.size:
            return
        event_type, _image_type = _HEADER.unpack_from(frame)
        if event_type == EVENT_PREVIEW_IMAGE:
            self.preview_frame.emit(frame[_HEADER.size:])

    def _handle_message(self, message: dict) -> None:
        msg_type = message.get("type")
        data = message.get("data") or {}
        prompt_id = str(data.get("prompt_id") or "")

        if msg_type == "status":
            status = data.get("status") or {}
            sid = data.get("sid")
            if sid:
                self.connected.emit(str(sid))
            remaining = (status.get("exec_info") or {}).get("queue_remaining")
            if remaining is not None:
                self.queue_changed.emit(int(remaining))

        elif msg_type == "execution_start":
            self.execution_start.emit(prompt_id)

        elif msg_type == "execution_cached":
            self.execution_cached.emit(prompt_id, [str(n) for n in data.get("nodes") or []])

        elif msg_type == "executing":
            node = data.get("node")
            self.executing.emit(prompt_id, str(node) if node is not None else None)

        elif msg_type == "progress":
            self.progress.emit(
                prompt_id, str(data.get("node") or ""),
                int(data.get("value") or 0), int(data.get("max") or 0),
            )

        elif msg_type == "executed":
            self.executed.emit(prompt_id, str(data.get("node") or ""), data.get("output") or {})

        elif msg_type == "execution_success":
            self.execution_success.emit(prompt_id)

        elif msg_type == "execution_error":
            self.execution_error.emit(prompt_id, data)

        elif msg_type == "execution_interrupted":
            self.execution_interrupted.emit(prompt_id, data)

        elif msg_type == preview.MESSAGE_TYPE:
            self._handle_preview_override(data)

        # progress_state duplicates `progress` with whole-graph detail, and feature_flags
        # only matters if we negotiate one; both are deliberately ignored.

    def _handle_preview_override(self, data: dict) -> None:
        """Decode a live sampler preview.

        Decoding runs here rather than on the GUI thread. It is a few tens of
        milliseconds per step against messages that arrive about once a second, and the
        node's own encoder is bounded and drops frames when full, so this cannot back up.

        These messages carry no prompt_id -- ComfyUI addresses them to whichever client
        submitted the running prompt, so anything that arrives is ours by construction.
        """
        if not self.previews_enabled:
            return
        try:
            clip = preview.decode_message(data)
        except Exception as exc:
            log.debug("Preview decode failed: %s", exc)
            return
        if clip:
            self.preview_clip.emit(clip)
