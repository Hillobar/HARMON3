"""Showing references in the result frame: one, or two side by side.

Used when a reference thumbnail is clicked -- one reference, or two side by side for
comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PySide6.QtCore import Qt, QTimer, QUrl
from PySide6.QtGui import QPixmap
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from .. import imaging, scaling
from ..refs import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS
from . import style

MAX_PANES = 2


@dataclass
class MediaItem:
    path: str
    caption: str = ""
    #: The reference row this came from, when it came from one. Carried so the result
    #: frame can offer the trim editor for whatever it is showing.
    row: object = None

    @property
    def is_video(self) -> bool:
        return Path(self.path).suffix.lower() in VIDEO_EXTENSIONS

    @property
    def is_audio(self) -> bool:
        return Path(self.path).suffix.lower() in AUDIO_EXTENSIONS

    @property
    def plays(self) -> bool:
        return self.is_video or self.is_audio


class MediaPane(QWidget):
    """One slot: a still, or a clip playing on a loop."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._pixmap: QPixmap | None = None
        #: What is on screen, so the pane can re-read its row's size ceiling.
        self._item: MediaItem | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self.stack = QStackedWidget()
        self.stack.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)

        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background: #000;")
        self.stack.addWidget(self.image_label)

        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("background: #000;")
        self.stack.addWidget(self.video_widget)
        layout.addWidget(self.stack, 1)

        self.caption = style.hint("")
        self.caption.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.caption)

        self.player = QMediaPlayer(self)
        self.audio = QAudioOutput(self)
        self.audio.setVolume(0.8)
        self.player.setAudioOutput(self.audio)
        self.player.setVideoOutput(self.video_widget)
        # References are short; looping makes a clip comparable to a still beside it.
        self.player.setLoops(QMediaPlayer.Infinite)

        #: Set while a newly opened clip is still to be parked on its first frame.
        self._park_when_loaded = False
        # Connected once, for the life of the pane, and never disconnected. Wiring this up
        # per item and tearing it down from inside its own slot mutates the connection
        # list while Qt is emitting through it, which is a hard crash rather than an
        # exception.
        self.player.mediaStatusChanged.connect(self._on_status)

    def show_item(self, item: MediaItem, play: bool = False) -> None:
        """Open a reference in this pane. Paused on its first frame unless asked to play.

        Selecting a reference is browsing, not playback: a list you click through should
        not start talking at you, and marking an in point wants a still picture anyway.
        """
        self._item = item
        self.caption.setText(item.caption)

        if item.is_video:
            self._pixmap = None
            self.stack.setCurrentWidget(self.video_widget)
            self.player.setSource(QUrl.fromLocalFile(item.path))
            self._start(play)
            return

        if item.is_audio:
            # Audio has nothing to draw, so the pane shows what it is holding instead.
            self._pixmap = None
            self.image_label.setPixmap(QPixmap())
            self.image_label.setText(f"♪  {Path(item.path).name}")
            self.image_label.setProperty("role", "hint")
            style.restyle(self.image_label)
            self.stack.setCurrentWidget(self.image_label)
            self.player.setSource(QUrl.fromLocalFile(item.path))
            self._start(play)
            return

        self.player.stop()
        self.player.setSource(QUrl())
        self.image_label.setText("")
        self._pixmap = imaging.load_pixmap(item.path)
        self.stack.setCurrentWidget(self.image_label)
        self._rescale()

    def _start(self, play: bool) -> None:
        """Play, or park on the first frame so there is a picture without any sound.

        A pause issued before the media has loaded is ignored, so it has to be applied
        again once it lands -- hence the flag rather than a single call here.
        """
        self._park_when_loaded = not play
        if play:
            self.player.play()
        else:
            self.player.pause()

    def _on_status(self, status) -> None:
        if status != QMediaPlayer.LoadedMedia or not self._park_when_loaded:
            return
        # Cleared first: pausing and seeking can emit mediaStatusChanged again, and this
        # must be a no-op the second time rather than a loop.
        self._park_when_loaded = False
        # Out of the emission entirely. Seeking a player from inside its own status
        # signal is what took the process down, not the seek itself.
        QTimer.singleShot(0, self._park)

    def _park(self) -> None:
        if not self.player.source().isEmpty():
            self.player.pause()
            self.player.setPosition(0)

    def clear(self) -> None:
        self.player.stop()
        self.player.setSource(QUrl())
        self._item = None
        self._pixmap = None
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("")
        self.caption.clear()

    def _rescale(self) -> None:
        """Fit the still to the pane, through its row's size ceiling if it has one.

        Genuinely resampled down and back up rather than merely relabelled, so lowering
        the slider far enough to matter *looks* like it matters. Above the pane's own
        size it will not: an image encoded at 1600px and shown in a 700px pane is
        indistinguishable from the original, and pretending otherwise -- by faking
        pixelation -- would misreport what the model is going to receive. That the picture
        stops changing is itself the useful signal, and the readout carries the number.
        """
        if self._pixmap is None or self._pixmap.isNull():
            return

        source = self._pixmap
        target = self._scaled_size()
        if target is not None and target[0] < source.width():
            # Only when the ceiling is below what is being displayed; otherwise this is an
            # expensive resample of a large pixmap that changes nothing on screen.
            fitted = self._pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio,
                                         Qt.SmoothTransformation)
            if target[0] < fitted.width():
                source = self._pixmap.scaled(target[0], target[1], Qt.IgnoreAspectRatio,
                                             Qt.SmoothTransformation)

        self.image_label.setPixmap(source.scaled(
            self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self._refresh_caption()

    def _scaled_size(self):
        row = getattr(self._item, "row", None)
        return row.scaled_size() if row is not None and row.scales else None

    def _refresh_caption(self) -> None:
        """Say what will actually be sent, while the slider is being moved."""
        if self._item is None:
            return
        caption = self._item.caption
        target = self._scaled_size()
        if target is not None:
            caption = (f"{caption}  -  sending {target[0]}x{target[1]}, "
                       f"{scaling.format_megapixels(*target)}").strip(" -")
        self.caption.setText(caption)

    def refresh_scale(self) -> None:
        """Re-render for a row whose ceiling has just moved."""
        self._rescale()
        self._refresh_caption()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._rescale()


class MediaView(QWidget):
    """One or two panes, laid out side by side."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._items: list[MediaItem] = []

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.panes = [MediaPane() for _ in range(MAX_PANES)]
        for pane in self.panes:
            layout.addWidget(pane, 1)
            pane.hide()

    def show_items(self, items: list[MediaItem]) -> None:
        self._items = [item for item in items if Path(item.path).is_file()][:MAX_PANES]
        for pane, item in zip(self.panes, self._items):
            pane.show_item(item)
            pane.show()
        for pane in self.panes[len(self._items):]:
            pane.clear()
            pane.hide()

    def items(self) -> list[MediaItem]:
        return list(self._items)

    def has_content(self) -> bool:
        return bool(self._items)

    def playing_pane(self) -> MediaPane | None:
        """The pane the transport should drive: the first one holding something that plays."""
        for pane, item in zip(self.panes, self._items):
            if item.plays:
                return pane
        return None

    def set_volume(self, volume: float) -> None:
        for pane in self.panes:
            pane.audio.setVolume(volume)

    def refresh_scale(self) -> None:
        """Re-render every pane, for a size ceiling that has just moved."""
        for pane in self.panes:
            pane.refresh_scale()

    def clear(self) -> None:
        self._items = []
        for pane in self.panes:
            pane.clear()
            pane.hide()
