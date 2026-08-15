"""Decoding the live sampler preview sent by the Model Preview Override node.

That node does not use ComfyUI's binary preview stream. It emits its own JSON message,
``kj_preview_override``, addressed to the client that submitted the prompt -- which is
this app -- carrying a base64 payload per sampler step:

    fragmented H.264 MP4   when the server's PyAV has NVENC and preview_frames > 1
    animated WebP          same, without NVENC
    JPEG                   when preview_frames == 1

Decoding happens on the websocket thread, so this module produces QImages (safe off the
GUI thread) rather than QPixmaps (which are not).
"""

from __future__ import annotations

import base64
import binascii
import io
import logging
from dataclasses import dataclass, field

from PySide6.QtCore import QBuffer, Qt
from PySide6.QtGui import QImage, QImageReader

log = logging.getLogger(__name__)

MESSAGE_TYPE = "kj_preview_override"

#: Frames are scaled down to this longest side before being held. The node caps its own
#: output at max_resolution (1024 by default), and a 24-frame clip at that size is ~42 MB
#: per step held in memory for a preview only a few hundred pixels wide.
MAX_PREVIEW_EDGE = 640

#: Guard against a malformed or hostile payload turning into an unbounded decode.
MAX_FRAMES = 256

try:
    import av
except ImportError:      # pragma: no cover - av is a hard dependency, this is belt and braces
    av = None
    log.warning("PyAV is not installed; MP4 sampler previews will be skipped")


@dataclass
class PreviewClip:
    """One sampler step's preview, decoded and ready to show."""

    frames: list = field(default_factory=list)
    fps: float = 12.0
    step: int = 0
    total: int = 0
    sigma: float | None = None
    #: Averaged step time reported by the node, in milliseconds.
    avg_step_ms: float | None = None
    step_ms: float | None = None

    def __bool__(self) -> bool:
        return bool(self.frames)


def decode_message(data: dict) -> PreviewClip | None:
    """Turn a ``kj_preview_override`` payload into a clip, or None if it carries no image.

    The node's first message of a run has no image -- it exists to deliver the sigma
    schedule -- so a None result here is normal, not a failure.
    """
    if not isinstance(data, dict):
        return None

    encoded = data.get("image")
    if not encoded:
        return None

    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as exc:
        log.debug("Undecodable preview payload: %s", exc)
        return None

    mime = (data.get("mime") or "").lower()
    frames = _decode_mp4(payload) if mime == "video/mp4" else _decode_with_qt(payload)
    if not frames:
        return None

    return PreviewClip(
        frames=frames,
        fps=_positive(data.get("fps")) or 12.0,
        step=_as_int(data.get("step")),
        total=_as_int(data.get("total")),
        sigma=_as_float(data.get("sigma")),
        avg_step_ms=_as_float(data.get("avg_step_ms")),
        step_ms=_as_float(data.get("step_ms")),
    )


def first_frame(path) -> QImage | None:
    """The opening frame of a video file, for use as a thumbnail."""
    if av is None:
        return None
    try:
        with av.open(str(path)) as container:
            if not container.streams.video:
                return None
            for frame in container.decode(container.streams.video[0]):
                return _image_from_frame(frame)
    except Exception as exc:
        log.debug("Could not read a thumbnail from %s: %s", path, exc)
    return None


def _decode_mp4(payload: bytes) -> list:
    """Decode the node's fragmented MP4. Qt cannot read one from memory; PyAV can."""
    if av is None:
        return []

    frames = []
    try:
        with av.open(io.BytesIO(payload)) as container:
            if not container.streams.video:
                return []
            stream = container.streams.video[0]
            stream.thread_type = "AUTO"
            for frame in container.decode(stream):
                image = _image_from_frame(frame)
                if image is not None:
                    frames.append(image)
                if len(frames) >= MAX_FRAMES:
                    break
    except Exception as exc:
        log.debug("MP4 preview decode failed: %s", exc)
        return []
    return frames


def _image_from_frame(frame) -> QImage | None:
    """Convert a decoded frame straight from its plane buffer.

    Deliberately not frame.to_ndarray(): that needs numpy, which PyAV does not require
    and this app does not otherwise want. A plane is already a contiguous buffer, and
    QImage takes the row stride directly, so the padding ffmpeg adds for alignment costs
    nothing to handle.
    """
    rgb = frame.reformat(format="rgb24")
    plane = rgb.planes[0]
    # QImage does not take ownership of the buffer, so copy() before it goes out of scope.
    image = QImage(bytes(plane), rgb.width, rgb.height,
                   plane.line_size, QImage.Format_RGB888).copy()
    return _scaled(image)


def _decode_with_qt(payload: bytes) -> list:
    """Read a still or an animated WebP. QImageReader walks multi-frame images natively."""
    # setData copies into the buffer's own storage. Constructing QBuffer around a
    # QByteArray built inline instead hands it a pointer to a temporary that Python then
    # frees, and the reader walks freed memory -- a hard crash, not an exception.
    buffer = QBuffer()
    buffer.setData(payload)
    buffer.open(QBuffer.ReadOnly)

    reader = QImageReader(buffer)
    reader.setDecideFormatFromContent(True)

    frames = []
    while True:
        image = reader.read()
        if image.isNull():
            break
        scaled = _scaled(image)
        if scaled is not None:
            frames.append(scaled)
        if len(frames) >= MAX_FRAMES or not reader.supportsAnimation():
            break
    buffer.close()
    return frames


def _scaled(image: QImage) -> QImage | None:
    if image.isNull():
        return None
    if max(image.width(), image.height()) <= MAX_PREVIEW_EDGE:
        return image
    return image.scaled(
        MAX_PREVIEW_EDGE, MAX_PREVIEW_EDGE,
        Qt.KeepAspectRatio, Qt.SmoothTransformation,
    )


def _as_int(value) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return 0


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _positive(value) -> float | None:
    number = _as_float(value)
    return number if number and number > 0 else None
