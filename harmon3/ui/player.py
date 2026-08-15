"""The result frame: whatever the app has to show, one timeline, and the transport.

References are added and owned by the reference panel. Clicking one brings it here to be
looked at and marked: the frame is where the in point is set, and that mark is where the
section of the reference the model receives begins. A reference therefore stays what it
always was -- a file plus a mark -- and this frame is the surface for setting the mark
rather than a place files are kept.

It opens *paused*, on its first frame. Clicking through a reference list is a browsing
action, and a clip that starts talking the moment it is selected makes that unpleasant.

There is one track, in ``timeline.py``, and it is both the scrub bar and the mark editor.
There were two, showing the same clip a line apart, and the frame is worth more than the
duplication. Nothing else here narrates: no caption line, no path line, no "sending this
much" line. The reference list says which file it is and what will be sent, the track
draws the section, and the line under the transport stays empty unless playback actually
fails.

QMediaPlayer needs an explicit QAudioOutput or it plays silently -- which matters here,
because MiniMax H3 generates the soundtrack along with the picture. If the platform
cannot decode the file, the widget degrades to an "open externally" button rather than
showing a black rectangle.

The frame holds three things -- reference clips, the live sampler preview, and the finished
video -- on three tabs rather than swapping one surface between them. Swapping meant that
opening a reference threw away the result you were looking at, with no way back except
running it again; on tabs all three stay, and you choose. The app still raises the tab that
has just gained something, because that is nearly always what you want to look at, with one
exception: preview clips arrive many times a second, so the live preview raises itself once
per run and then leaves the choice alone.

Two of the three play sound. The transport therefore drives whichever player belongs to the
tab in front rather than owning one outright, so Play, Stop, the timeline and the volume
always act on the thing being looked at. Switching tabs *parks* what you left -- paused, on
its frame -- rather than stopping it, so coming back finds it where you put it.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QEvent, Qt, QTimer, QUrl, Signal
from PySide6.QtGui import QDesktopServices
from PySide6.QtMultimedia import QAudioOutput, QMediaPlayer
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWidgets import (
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QStackedWidget,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from . import style
from .live_preview import LivePreviewWidget
from .media_view import MediaView
from .timeline import WHEEL_HINT, TimelineBar, WheelStepper

#: The three tabs, in the order they are added.
TAB_REFERENCE = 0
TAB_LATENT = 1
TAB_RESULTS = 2


def park(player: QMediaPlayer, position: int = 0) -> None:
    """Stop playback with the frame at ``position`` left on screen.

    ``QMediaPlayer.stop()`` tears the decode pipeline down and the video surface goes
    black with it -- and while a player is stopped, seeking moves the position without
    presenting anything, so scrubbing after a stop shows nothing at all. Pausing keeps the
    pipeline up, and a seek while paused delivers the frame it lands on. Every "stop" in
    this widget is therefore a pause and a rewind, and the picture stays.
    """
    if player.source().isEmpty():
        return
    player.pause()
    player.setPosition(max(0, position))


class VideoPlayer(QGroupBox):
    """Playback surface, one timeline, and the transport controls."""

    error_occurred = Signal(str)
    #: An in point was marked here; the reference row has already been updated.
    trim_changed = Signal()
    def __init__(self, parent=None):
        # No group title: this widget lives in a pane whose title bar already names it.
        super().__init__("", parent)
        self.setProperty("role", "pane")
        self._path: Path | None = None
        #: The player the transport is currently driving, if any.
        self._bound: QMediaPlayer | None = None
        #: The file the trim editor was opened on, so a row repointed elsewhere closes it.
        self._trim_path: str | None = None
        #: None until the Open button has a handler; True once it opens the file itself
        #: (after a decode failure) rather than its folder.
        self._open_mode: bool | None = None
        #: Whether the live preview has already claimed the frame for the current run.
        #: Preview clips arrive many times a second; raising the tab on each of them would
        #: make it impossible to look at anything else while a run is going.
        self._live_raised = False
        #: The tab the frame is on, kept so a change knows which one to park.
        self._left_tab = TAB_RESULTS

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(6)

        #: Wheel deltas over the picture, accumulated separately from the track's own so
        #: half a frame rolled on one does not finish itself off on the other.
        self._wheel = WheelStepper()

        self.tabs = QTabWidget()
        # Modest, because the frame carries the timeline and the transport underneath it:
        # a tall floor here squeezes those out instead.
        self.tabs.setMinimumHeight(150)

        # -- Reference: one or two clips or stills, brought here from the reference list.
        reference_page = QWidget()
        reference_layout = QVBoxLayout(reference_page)
        reference_layout.setContentsMargins(0, 0, 0, 0)
        self.ref_hint = QLabel(
            "Click a reference to open it here and mark where it starts.")
        self.ref_hint.setAlignment(Qt.AlignCenter)
        self.ref_hint.setProperty("role", "hint")
        # Wrapped, and free to be narrower than its text: an empty-state line is not worth
        # a floor under the width of the pane it is sitting in.
        self.ref_hint.setWordWrap(True)
        self.ref_hint.setMinimumWidth(0)
        reference_layout.addWidget(self.ref_hint)
        self.media_view = MediaView()
        reference_layout.addWidget(self.media_view, 1)

        # -- Latent: the sampler's own view of the clip, while it is still being made.
        self.live_preview = LivePreviewWidget()

        # -- Results: the finished video, or the line that says how one gets here.
        self.results_page = QStackedWidget()
        self.placeholder = QLabel("Queue a run to see the result here.")
        self.placeholder.setAlignment(Qt.AlignCenter)
        self.placeholder.setProperty("role", "hint")
        self.placeholder.setWordWrap(True)
        self.placeholder.setMinimumWidth(0)
        self.results_page.addWidget(self.placeholder)
        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("background: #000;")
        self.video_widget.setToolTip(WHEEL_HINT)
        self.results_page.addWidget(self.video_widget)

        self.tabs.addTab(reference_page, "Reference")
        self.tabs.addTab(self.live_preview, "Latent")
        self.tabs.addTab(self.results_page, "Results")
        self.tabs.setCurrentIndex(TAB_RESULTS)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        layout.addWidget(self.tabs, 1)

        self.player = QMediaPlayer(self)
        # Without this the video plays but is silent.
        self.audio_output = QAudioOutput(self)
        self.audio_output.setVolume(0.8)
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_widget)
        self.player.errorOccurred.connect(self._on_error)
        self.player.mediaStatusChanged.connect(self._on_media_status)

        # One timeline, for everything: it scrubs whatever is on screen, and carries the
        # in point as well when what is on screen is a reference.
        self.timeline = TimelineBar()
        self.timeline.changed.connect(self.trim_changed.emit)

        layout.addWidget(self.timeline)
        layout.addWidget(self._build_controls())

        # Nothing routine goes here -- captions, paths and what-will-be-sent notes all
        # cost a line of the frame to say what the reference list and the track already
        # show. It stays for failures, which have nowhere else to appear, and takes up no
        # room while there are none.
        self.status_label = style.hint("")
        self.status_label.setWordWrap(True)
        self.status_label.hide()
        layout.addWidget(self.status_label)

        self._watch_for_wheel()
        self._update_transport()

    def _set_status(self, message: str = "", *, role: str = "hint") -> None:
        """Show a failure under the transport, or take the line back when it clears."""
        self.status_label.setText(message)
        self.status_label.setProperty("role", role)
        self.status_label.setVisible(bool(message))
        style.restyle(self.status_label)

    def _build_controls(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.play_button = QPushButton("Play")
        self.play_button.clicked.connect(self.toggle_play)
        layout.addWidget(self.play_button)

        # Pause leaves the frame where it stopped, which is what you want mid-inspection;
        # Stop silences the clip and takes it back to the start, which is what you want
        # when the sound is simply in the way. Hence both, and Stop only when there is
        # something playing for it to act on.
        self.stop_button = QPushButton("Stop")
        self.stop_button.setToolTip("Stop playback and rewind to the start")
        self.stop_button.clicked.connect(self.stop)
        layout.addWidget(self.stop_button)

        # No scrub bar here: the timeline above is the scrub bar, and a second one showing
        # the same clip was the same information twice, one line apart.
        self.time_label = QLabel("0:00 / 0:00")
        self.time_label.setMinimumWidth(84)
        self.time_label.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        layout.addWidget(self.time_label, 1)

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(80)
        self.volume_slider.setFixedWidth(80)
        self.volume_slider.setToolTip("Volume")
        self.volume_slider.valueChanged.connect(self._on_volume)
        layout.addWidget(self.volume_slider)

        self.open_button = QPushButton("Open folder")
        layout.addWidget(self.open_button)
        self._set_open_mode(external=False)

        return row

    def _watch_for_wheel(self) -> None:
        """Catch the wheel over every surface the frame can show.

        A filter on each one rather than letting the event bubble up to this widget:
        QVideoWidget renders through a native surface and the wheel does not reliably
        arrive here from it, which shows up as the gesture working over some of the frame
        and silently not over the picture -- the one place it is aimed at.

        Installed once, over the whole subtree, because the panes and their labels are all
        built in __init__ and none of them is replaced afterwards.

        The page area rather than the tab widget itself, so the tab bar is left out: it is
        filtered separately below, to take the wheel away from it entirely.
        """
        # Direct children only: the panes have stacks of their own further down, and a
        # recursive search could hand back one of those instead of the tab widget's.
        pages = self.tabs.findChild(QStackedWidget, options=Qt.FindDirectChildrenOnly)
        pages.installEventFilter(self)
        for child in pages.findChildren(QWidget):
            child.installEventFilter(self)
        self.tabs.tabBar().installEventFilter(self)

    def eventFilter(self, watched, event):
        if event.type() == QEvent.Wheel:
            # QTabBar rolls through its tabs on a wheel, which would flip what the frame is
            # showing while the gesture is aimed at stepping frames. Stepping from over the
            # tab bar would be just as arbitrary, so it does neither.
            if watched is self.tabs.tabBar():
                event.accept()
                return True
            self.wheelEvent(event)
            if event.isAccepted():
                return True
        return super().eventFilter(watched, event)

    def wheelEvent(self, event):
        """Step frames from over the picture, the same as from over the track.

        One handler covers the result, a reference clip, two clips side by side and the
        live preview, without any of them knowing the timeline exists.

        The volume slider keeps its own wheel, which is what a slider under the pointer
        should do -- it is outside the stack this filters.
        """
        frames = self._wheel.frames(event)
        if frames:
            self.timeline.step_frames(frames)
        event.accept()

    # -- source --------------------------------------------------------------------

    def show_media(self, items: list) -> None:
        """Open references on the Reference tab, replacing whatever was there."""
        if not items:
            return
        park(self.player)
        self.media_view.show_items(items)
        self.ref_hint.hide()
        self.tabs.setCurrentIndex(TAB_REFERENCE)
        self._set_status()
        self._update_transport()

    def shown_media(self) -> list:
        """What the Reference tab currently holds, for shift-click to build on."""
        if self._current_tab() != TAB_REFERENCE:
            return []
        return self.media_view.items()

    def _current_tab(self) -> int:
        return self.tabs.currentIndex()

    def edit_trim(self, row) -> None:
        """Bind the timeline to a reference with no local copy to play here.

        A row restored from history names a file on the server and nothing on this
        machine, so there is no clip to mark against -- but its in point is still worth
        being able to set, so the number stays while the playhead goes.
        """
        self.timeline.bind(row, None)
        self._trim_path = row.local_path

    def refresh_scale(self, row=None) -> None:
        """Re-render the shown stills for a size ceiling that has just moved.

        Cheap enough to run from a slider being dragged: a pane whose ceiling is above
        what it is displaying skips the resample entirely.
        """
        if row is not None and all(item.row is not row for item in self.shown_media()):
            return
        self.media_view.refresh_scale()

    def refresh_trim(self) -> None:
        """Re-read the bound row, for when a probe has just supplied its length or rate."""
        if self.timeline.row is not None:
            self.timeline.refresh()

    def set_target_frames(self, frames: int) -> None:
        """Tell the timeline how long the generated clip is, which is how long a section
        of a reference runs for."""
        self.timeline.set_target_frames(frames)

    def forget_rows_not_in(self, rows) -> None:
        """Let the timeline go when its reference has been removed or repointed."""
        row = self.timeline.row
        if row is None:
            return
        # Compared both ways round: a row that has just *gained* a local file has a clip
        # to mark against now, which the timeline was bound to it without.
        if row.local_path != self._trim_path or all(existing is not row for existing in rows):
            self.timeline.unbind()
            self._trim_path = None

    def show_live_preview(self, clip) -> None:
        """Display a sampler step's preview, raising the Latent tab once per run.

        Once, because these arrive many times a second: raising on every one would take the
        frame back from anything else being looked at, several times a second, for the
        length of the run.
        """
        self.live_preview.show_clip(clip)
        if not self._live_raised:
            self._live_raised = True
            self.tabs.setCurrentIndex(TAB_LATENT)
        self._update_transport()

    def end_live_preview(self) -> None:
        """Stop animating. The tab keeps the last step until the next run replaces it."""
        self.live_preview.stop()
        self._live_raised = False

    def load(self, path: str | Path, autoplay: bool = True, raise_tab: bool = True) -> None:
        """Show a finished video on the Results tab.

        ``raise_tab`` is for a result that arrives mid-batch: it goes into the tab and
        waits there rather than taking the frame away from whatever is being looked at
        while the rest of the queue works through.
        """
        self.live_preview.clear()
        self._live_raised = False
        self._path = Path(path)
        self._set_status()

        # A previous file may have failed to decode and rebound this button; a new file
        # deserves the normal behaviour back until it fails on its own account.
        self._set_open_mode(external=False)
        self.results_page.setCurrentWidget(self.video_widget)
        if raise_tab:
            self.tabs.setCurrentIndex(TAB_RESULTS)

        self.player.setSource(QUrl.fromLocalFile(str(self._path)))
        self._update_transport()
        if autoplay:
            self.player.play()

    def clear(self) -> None:
        self.player.stop()
        self.player.setSource(QUrl())
        self.media_view.clear()
        self.ref_hint.show()
        self.live_preview.clear()
        self._live_raised = False
        self._path = None
        self.results_page.setCurrentWidget(self.placeholder)
        self.tabs.setCurrentIndex(TAB_RESULTS)
        self._set_status()
        self._update_transport()

    def current_path(self) -> Path | None:
        return self._path

    # -- transport -----------------------------------------------------------------

    def toggle_play(self) -> None:
        player = self._bound
        if player is None:
            return
        if player.playbackState() == QMediaPlayer.PlayingState:
            player.pause()
            return
        if player.mediaStatus() == QMediaPlayer.EndOfMedia:
            player.setPosition(0)
        player.play()

    def stop(self) -> None:
        """Stop whatever the frame is playing and rewind it, leaving the picture up."""
        for player in self._live_players():
            park(player, self._rewind_target(player))

    def _live_players(self) -> list:
        """The players whose output the tab in front is actually showing."""
        return self._players_for_tab(self._current_tab())

    def _rewind_target(self, player: QMediaPlayer) -> int:
        """Where Stop goes back to: the in point of a marked reference, else the start."""
        row = self.timeline.row
        if row is not None and row.marked and player is self.timeline.player:
            return self.timeline.window_ms()[0]
        return 0

    def _on_tab_changed(self, index: int) -> None:
        """Park what was left behind, then hand the transport to what is now in front.

        Parked rather than stopped: a stop tears the decode pipeline down and the surface
        goes black with it, so a tab flipped away from and back would come back to nothing.
        Nothing here touches the live preview's animation -- the run owns that, and a run
        is still going whether or not its tab is the one being looked at.
        """
        for player in self._players_for_tab(self._left_tab):
            park(player, self._rewind_target(player))
        self._left_tab = index
        self._update_transport()

    def _players_for_tab(self, tab: int) -> list:
        """The players a tab is showing, if it is showing anything at all.

        The Results tab is in front from the start, holding its "queue a run" line; there
        is no player behind that until a file has been loaded.
        """
        if tab == TAB_REFERENCE:
            return [pane.player for pane in self.media_view.panes]
        if tab == TAB_RESULTS and self._path is not None:
            return [self.player]
        return []

    def refresh_surfaces(self) -> None:
        """Re-present the current frame after the widget has been reparented.

        Floating the pane this lives in moves it into a new native window, and the video
        surfaces are recreated empty by the move -- the picture goes black while the player
        still believes it is showing something. Parking each one presents its frame again.
        """
        for player in self._live_players():
            park(player, player.position())
        self.live_preview.refresh()

    def reveal(self) -> None:
        if self._path and self._path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._path.parent)))

    def open_externally(self) -> None:
        if self._path and self._path.exists():
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(self._path)))

    # -- wiring --------------------------------------------------------------------

    def _update_transport(self) -> None:
        """Point the controls at whatever the frame is showing, and enable them if it plays."""
        tab = self._current_tab()
        if tab == TAB_REFERENCE:
            pane = self.media_view.playing_pane()
            self._bind(pane.player if pane is not None else None)
        elif tab == TAB_RESULTS:
            self._bind(self.player if self._path is not None else None)
        else:
            self._bind(None)

        playable = self._bound is not None
        for widget in (self.play_button, self.volume_slider):
            widget.setEnabled(playable)
        # The one control that is about the file on disk rather than about playback.
        self.open_button.setEnabled(self._path is not None)
        self.stop_button.setVisible(playable)

        if not playable:
            self.time_label.setText("0:00 / 0:00")
            self.play_button.setText("Play")

        self._sync_timeline()

    def _bind(self, player: QMediaPlayer | None) -> None:
        """Drive the transport from `player`, releasing whichever one it was on before."""
        if player is self._bound:
            return
        if self._bound is not None:
            self._bound.positionChanged.disconnect(self._on_position)
            self._bound.durationChanged.disconnect(self._on_duration)
            self._bound.playbackStateChanged.disconnect(self._on_playback_state)

        self._bound = player
        if player is None:
            return

        player.positionChanged.connect(self._on_position)
        player.durationChanged.connect(self._on_duration)
        player.playbackStateChanged.connect(self._on_playback_state)
        self._on_duration(player.duration())
        self._on_position(player.position())
        self._on_playback_state(player.playbackState())

    def _sync_timeline(self) -> None:
        """Point the timeline at what the frame is showing, and at its player.

        A reference gets its in point on the track as well; anything else -- a finished
        video, a live preview, two clips side by side -- gets a plain scrub bar. Two clips
        because there is no single thing a handle would belong to, and comparing is not
        marking.
        """
        items = self.shown_media()
        row = items[0].row if len(items) == 1 else None
        if row is not None and not getattr(row, "supports_trim", False):
            row = None

        self.timeline.bind(row, self._bound)
        self._trim_path = row.local_path if row is not None else None

    # -- signals -------------------------------------------------------------------

    def _on_position(self, position: int) -> None:
        duration = self._bound.duration() if self._bound is not None else 0
        self._update_time_label(position, duration)

    def _on_duration(self, duration: int) -> None:
        position = self._bound.position() if self._bound is not None else 0
        self._update_time_label(position, duration)

    def _on_volume(self, value: int) -> None:
        volume = value / 100.0
        self.audio_output.setVolume(volume)
        self.media_view.set_volume(volume)

    def _on_playback_state(self, state) -> None:
        self.play_button.setText("Pause" if state == QMediaPlayer.PlayingState else "Play")

    def _on_media_status(self, status) -> None:
        """Keep a picture up at the two moments Qt would otherwise leave the frame black.

        Reaching the end of a clip stops the player, and a file that has been loaded but
        never played has presented no frame at all. Both are recoverable by parking on the
        first frame, which is also where Play would start from next.

        Deferred to a zero timer rather than done here: pausing and seeking a player from
        inside its own status signal mutates state Qt is still emitting through, and takes
        the process down without so much as a traceback.
        """
        if status == QMediaPlayer.EndOfMedia:
            QTimer.singleShot(0, self._park_result)
        elif status == QMediaPlayer.LoadedMedia:
            if self.player.playbackState() != QMediaPlayer.PlayingState:
                QTimer.singleShot(0, self._park_result)

    def _park_result(self) -> None:
        # Re-checked on arrival: Play may have been pressed in the meantime, and yanking
        # it back to the start then would be its own bug.
        if self.player.playbackState() != QMediaPlayer.PlayingState:
            park(self.player)

    def _on_error(self, error, error_string: str) -> None:
        if error == QMediaPlayer.NoError:
            return
        message = error_string or "this system cannot decode the video"
        self._set_status(
            f"Playback failed ({message}). The file is saved - use Open externally.",
            role="warn",
        )
        self._set_open_mode(external=True)
        self.error_occurred.emit(message)

    def _set_open_mode(self, external: bool) -> None:
        if self._open_mode == external:
            return
        if self._open_mode is not None:
            self.open_button.clicked.disconnect()
        self._open_mode = external
        if external:
            self.open_button.setText("Open externally")
            self.open_button.clicked.connect(self.open_externally)
        else:
            self.open_button.setText("Open folder")
            self.open_button.clicked.connect(self.reveal)

    def _update_time_label(self, position: int, duration: int) -> None:
        self.time_label.setText(f"{_format_ms(position)} / {_format_ms(duration)}")


def _format_ms(milliseconds: int) -> str:
    seconds = max(0, milliseconds) // 1000
    return f"{seconds // 60}:{seconds % 60:02d}"
