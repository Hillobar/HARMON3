"""Local media probing, used to warn about reference videos before they are submitted.

Reference videos are the one input that can silently ruin a run: the model assumes 24 fps
and needs at least 5 frames, and a video with no audio track quietly withholds its
<Audio j> tag, shifting every audio ordinal after it.

PyAV, when present, is what ComfyUI's own GetVideoComponents uses, so it sees exactly
what the server will. Without it the app falls back to QtMultimedia metadata, which is
already a dependency and good enough for the warnings.
"""

from __future__ import annotations

import logging

from pathlib import Path

from PySide6.QtCore import QObject, QTimer, QUrl, Signal
from PySide6.QtGui import QImageReader
from PySide6.QtMultimedia import QMediaPlayer

from .. import imaging

log = logging.getLogger(__name__)


def _empty_reason(path: str) -> str | None:
    """Why a file is obviously unusable before anything tries to decode it."""
    try:
        size = Path(path).stat().st_size
    except OSError as exc:
        return str(exc)
    return "the file is empty" if size == 0 else None

try:
    import av
except ImportError:
    av = None


class MediaProbe(QObject):
    """Probes local files and reports what it learns, keyed by reference-row uid."""

    probed = Signal(int, dict)  # row uid, {duration_s, fps, frame_count, has_audio, width, height}
    #: Nothing available could read this file at all. Not "some metadata was missing" --
    #: the file is not usable media, and sending it anywhere is a wasted round trip
    #: ending in a decoder error on the server.
    unreadable = Signal(int, str)   # row uid, why

    def __init__(self, parent: QObject | None = None):
        super().__init__(parent)
        self._players: dict[int, QMediaPlayer] = {}

    def probe(self, uid: int, path: str, kind: str) -> None:
        empty = _empty_reason(path)
        if empty:
            self.unreadable.emit(uid, empty)
            return

        if kind == "image":
            # Before PyAV, not after it. PyAV opens a still perfectly happily -- as a
            # one-frame video -- and reports the dimensions *stored* in the file, with no
            # notion of the EXIF Orientation tag. So a portrait photograph came back
            # 4000x3000, and every size derived from that was computed for a landscape
            # picture: the rescaled copy was a tall image squashed into a wide box.
            #
            # Qt's image reader is the one that decides for an image, and `imaging` is
            # where it is asked the way the server would ask.
            size = imaging.oriented_size(path)
            if size is not None:
                self.probed.emit(uid, {"width": size[0], "height": size[1]})
            else:
                self.unreadable.emit(
                    uid, QImageReader(path).errorString() or "not a readable image")
            return

        if av is not None:
            info = _probe_with_av(path)
            if info:
                self.probed.emit(uid, info)
                return

        self._probe_with_qt(uid, path)

    def _probe_with_qt(self, uid: int, path: str) -> None:
        # A row can be re-probed before its first probe reports (adding a second video
        # re-runs the sweep), so retire any player already working on this uid rather
        # than dropping it on the floor still holding a decoder and a file handle.
        self._release(uid)

        player = QMediaPlayer(self)
        self._players[uid] = player

        def finish():
            metadata = player.metaData()
            from PySide6.QtMultimedia import QMediaMetaData

            duration_ms = player.duration() or metadata.value(QMediaMetaData.Duration) or 0
            fps = metadata.value(QMediaMetaData.VideoFrameRate)
            resolution = metadata.value(QMediaMetaData.Resolution)

            info = {
                "duration_s": (duration_ms or 0) / 1000.0 or None,
                "fps": float(fps) if fps else None,
                "has_audio": bool(player.audioTracks()),
            }
            if info["fps"] and info["duration_s"]:
                info["frame_count"] = int(round(info["fps"] * info["duration_s"]))
            if resolution is not None and hasattr(resolution, "width"):
                info["width"], info["height"] = resolution.width(), resolution.height()

            self.probed.emit(uid, info)
            self._release(uid)

        def on_status(status):
            if status in (QMediaPlayer.LoadedMedia, QMediaPlayer.BufferedMedia):
                finish()
            elif status == QMediaPlayer.InvalidMedia:
                # Both readers have now refused it, so this is a verdict rather than a
                # gap: PyAV could not open it either, or it would not have got here.
                self.unreadable.emit(uid, "no installed decoder can read this file")
                self._release(uid)

        player.mediaStatusChanged.connect(on_status)
        player.setSource(QUrl.fromLocalFile(path))
        # Never let a stuck decoder leak a player. Keyed on the player, not the uid, so a
        # replacement probe for the same row is not torn down by its predecessor's timer.
        QTimer.singleShot(8000, lambda: self._release(uid, only=player))

    def _release(self, uid: int, only: QMediaPlayer | None = None) -> None:
        player = self._players.get(uid)
        if player is None or (only is not None and player is not only):
            return
        self._players.pop(uid, None)
        player.setSource(QUrl())
        player.deleteLater()


def _probe_with_av(path: str) -> dict | None:
    try:
        with av.open(path) as container:
            if not container.streams.video:
                return None
            stream = container.streams.video[0]
            rate = stream.average_rate or stream.guessed_rate
            fps = float(rate) if rate else None

            frame_count = stream.frames or None
            duration_s = None
            if stream.duration is not None and stream.time_base:
                duration_s = float(stream.duration * stream.time_base)
            elif container.duration:
                duration_s = container.duration / 1_000_000.0
            if not frame_count and fps and duration_s:
                frame_count = int(round(fps * duration_s))

            return {
                "duration_s": duration_s,
                "fps": fps,
                "frame_count": frame_count,
                "has_audio": bool(container.streams.audio),
                "width": stream.codec_context.width or None,
                "height": stream.codec_context.height or None,
            }
    except Exception as exc:
        log.debug("PyAV could not probe %s: %s", path, exc)
        return None
