"""The live sampler preview: the emerging video, animating, while the run is in flight."""

from __future__ import annotations

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QHBoxLayout, QLabel, QSizePolicy, QVBoxLayout, QWidget

from ..preview import PreviewClip
from . import style

#: Slowest playback the node is allowed to ask for, so a stray fps of 1 does not stall it.
MIN_FPS = 2.0
MAX_FPS = 30.0


class LivePreviewWidget(QWidget):
    """Plays the most recent step's clip on a loop until the next one replaces it.

    Each sampler step arrives as a short clip of its own. Rather than restarting playback
    every time -- which reads as a stutter -- the new frames are swapped in underneath and
    the loop keeps running.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frames: list = []
        self._index = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(4)

        self.canvas = QLabel()
        self.canvas.setAlignment(Qt.AlignCenter)
        self.canvas.setMinimumHeight(160)
        self.canvas.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.canvas.setStyleSheet("background: #000;")
        layout.addWidget(self.canvas, 1)

        caption = QHBoxLayout()
        caption.setContentsMargins(2, 0, 2, 0)

        self.step_label = style.metric("")
        caption.addWidget(self.step_label)

        self.detail_label = style.hint("")
        caption.addWidget(self.detail_label, 1)

        self.note_label = style.hint("live preview")
        caption.addWidget(self.note_label)
        layout.addLayout(caption)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._advance)

    # -- content -------------------------------------------------------------------

    def show_clip(self, clip: PreviewClip) -> None:
        restart = not self._frames
        self._frames = clip.frames
        if restart or self._index >= len(self._frames):
            self._index = 0

        interval = int(1000 / min(max(clip.fps, MIN_FPS), MAX_FPS))
        if not self._timer.isActive() or self._timer.interval() != interval:
            self._timer.start(interval)

        if clip.total:
            self.step_label.setText(f"{clip.step}/{clip.total}")
        if clip.sigma is not None:
            self.detail_label.setText(f"sigma {clip.sigma:.3f}")

        if len(self._frames) == 1:
            self._timer.stop()
            self._render(0)
        else:
            self._render(self._index)

    def stop(self) -> None:
        """Freeze on the current frame, keeping it visible."""
        self._timer.stop()
        self.note_label.setText("last preview")

    def clear(self) -> None:
        self._timer.stop()
        self.note_label.setText("live preview")
        self._frames = []
        self._index = 0
        self.canvas.setPixmap(QPixmap())
        self.step_label.clear()
        self.detail_label.clear()

    def has_content(self) -> bool:
        return bool(self._frames)

    def refresh(self) -> None:
        """Re-draw the current frame, for when the widget has been moved or resized."""
        self._render(self._index)

    # -- playback ------------------------------------------------------------------

    def _advance(self) -> None:
        if not self._frames:
            self._timer.stop()
            return
        self._index = (self._index + 1) % len(self._frames)
        self._render(self._index)

    def _render(self, index: int) -> None:
        if not self._frames:
            return
        image = self._frames[min(index, len(self._frames) - 1)]
        self.canvas.setPixmap(QPixmap.fromImage(image).scaled(
            self.canvas.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render(self._index)
