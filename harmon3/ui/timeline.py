"""The result frame's timeline: one track that scrubs, and marks.

There used to be two -- a transport slider for whatever was on screen, and a separate trim
track underneath it for the reference being marked -- showing the same clip twice, one
above the other. This is the single track that does both jobs. When the frame is showing a
finished video it is a scrub bar with a playhead; when it is showing a reference it also
carries that reference's in point, with the section that will be sent drawn from the mark
and everything outside it cut away.

Marking is a *viewing* activity -- you find the moment by looking at it, then mark it --
which is why it happens here beside the picture rather than on the reference row, and why
it behaves the way a video editor's does: drag the handle, or press I at the playhead, and
playback loops inside the section so it can be judged rather than guessed at.

There is no out point and no on/off switch. Every video and audio reference is cut to the
generated length, because that is what the model keeps of one either way, so the section
runs from the mark for exactly as long as the clip being generated: the *Duration*
parameter is the out point, and the drawn window follows it. What is drawn is clamped to
the end of the file, so a reference that runs out shows it.

Everything on screen is in milliseconds, because that is what QMediaPlayer speaks. What is
stored on the row is not -- a video's mark is in frames and an audio's is in seconds, each
being the exact unit its slicing node takes -- so every edit is converted on the way out
and back.
"""

from __future__ import annotations

from PySide6.QtCore import QRect, Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QPainter, QPen, QShortcut
from PySide6.QtMultimedia import QMediaPlayer
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import config, mathmirror
from ..refs import VIDEO
from . import style

HANDLE_W = 7
#: How close to the handle a click counts as grabbing it rather than scrubbing.
GRAB_PX = 9
RULER_H = 14
TRACK_H = 26
BAR_H = RULER_H + TRACK_H + 4

#: Ruler steps in milliseconds; the first that leaves at most MAX_TICKS marks is used.
TICK_STEPS_MS = (100, 250, 500, 1000, 2000, 5000, 10_000, 15_000, 30_000,
                 60_000, 120_000, 300_000, 600_000, 1_800_000)
MAX_TICKS = 8

#: How far behind the in point playback may drift before it is pulled back. Position
#: updates arrive every few tens of milliseconds, so this has to survive one of them.
LOOP_SLACK_MS = 250

#: A wheel notch, in the eighths of a degree Qt reports. A mouse sends one of these per
#: click; a trackpad sends a stream of much smaller ones, which is why the remainder is
#: carried rather than rounded away.
WHEEL_NOTCH = 120

#: What Shift multiplies a step by.
COARSE_STEP = 10

#: Said on the track and over the picture, because a gesture with nothing on screen to
#: suggest it is a gesture nobody finds.
WHEEL_HINT = (f"Wheel over the picture or the track to step a frame at a time, "
              f"Shift for {COARSE_STEP}.")


class WheelStepper:
    """Turns wheel deltas into whole frames, keeping what is left over.

    A mouse reports 120 per click and a trackpad reports a stream of 3s and 5s. Rounding
    each event on its own would make a trackpad do nothing at all, so the fraction of a
    frame is kept and spent on a later event.
    """

    def __init__(self):
        self._carry = 0

    def frames(self, event) -> int:
        """How many frames this wheel event is worth. Positive is forwards."""
        delta = event.angleDelta().y() or event.angleDelta().x()
        if not delta:
            return 0
        self._carry += delta
        # Truncated towards zero, not floored: divmod on a negative carry would hand back
        # a step in the wrong direction and a positive remainder.
        notches = int(self._carry / WHEEL_NOTCH)
        self._carry -= notches * WHEEL_NOTCH
        if not notches:
            return 0
        if event.modifiers() & Qt.ShiftModifier:
            notches *= COARSE_STEP
        return notches


def format_timecode(milliseconds: int) -> str:
    """m:ss.cc -- fine enough to mark a frame, short enough for a compact readout."""
    milliseconds = max(0, int(milliseconds))
    minutes, remainder = divmod(milliseconds, 60_000)
    seconds, millis = divmod(remainder, 1000)
    return f"{minutes}:{seconds:02d}.{millis // 10:02d}"


def _tick_step(duration_ms: int) -> int:
    for step in TICK_STEPS_MS:
        if duration_ms / step <= MAX_TICKS:
            return step
    return TICK_STEPS_MS[-1]


class TimelineTrack(QWidget):
    """The track itself: a ruler, a playhead, and -- when marking -- a section.

    Only the in edge is a handle. The other end of the section is where it runs out, set
    by the generated length rather than by this widget, so it is drawn as an edge rather
    than as something to grab.
    """

    #: The in handle moved. Carries its new position, in milliseconds.
    in_moved = Signal(int)
    #: The playhead was dragged somewhere.
    scrubbed = Signal(int)
    #: A drag finished, so the edit can be committed rather than spamming on every pixel.
    edited = Signal()
    #: The wheel asked to move some number of frames. Frames rather than milliseconds,
    #: because the point of it is to land on a frame boundary rather than near one.
    stepped = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._wheel = WheelStepper()
        self._duration = 0
        self._in = 0
        self._out = 0
        self._position = 0
        self._marking = False
        self._playhead = True
        self._drag = ""
        self._hover = ""
        #: One nudge of the arrow keys, in milliseconds. A frame, when that is known.
        self.step_ms = 40

        self.setFixedHeight(BAR_H)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

    # -- state ---------------------------------------------------------------------

    def set_duration(self, milliseconds: int) -> None:
        self._duration = max(0, int(milliseconds))
        self.update()

    def set_window(self, start_ms: int, end_ms: int) -> None:
        self._in, self._out = int(start_ms), int(end_ms)
        self.update()

    def set_position(self, milliseconds: int) -> None:
        self._position = int(milliseconds)
        self.update()

    def position(self) -> int:
        return self._position

    def set_marking(self, marking: bool) -> None:
        """Whether there is a section on this track at all.

        Off for a finished video: it has a playhead and nothing to mark, and drawing a
        handle on it would offer an edit that has nowhere to go.
        """
        self._marking = bool(marking)
        self.update()

    def set_playhead_visible(self, visible: bool) -> None:
        self._playhead = bool(visible)
        self.update()

    def is_dragging(self) -> bool:
        return bool(self._drag)

    # -- geometry ------------------------------------------------------------------

    def _track(self) -> QRect:
        # Inset by half a handle at each end so a handle at 0 or at the end is fully drawn.
        return QRect(HANDLE_W, RULER_H, max(1, self.width() - 2 * HANDLE_W), TRACK_H)

    def _x_for(self, milliseconds: int) -> int:
        track = self._track()
        if self._duration <= 0:
            return track.left()
        ratio = min(1.0, max(0.0, milliseconds / self._duration))
        return track.left() + int(round(ratio * (track.width() - 1)))

    def _ms_at(self, x: int) -> int:
        track = self._track()
        if self._duration <= 0 or track.width() <= 1:
            return 0
        ratio = (x - track.left()) / (track.width() - 1)
        return int(round(min(1.0, max(0.0, ratio)) * self._duration))

    def _hit(self, x: int) -> str:
        if self._duration <= 0:
            return ""
        if self._marking and abs(x - self._x_for(self._in)) <= GRAB_PX:
            return "in"
        return "playhead"

    # -- painting ------------------------------------------------------------------

    def paintEvent(self, _event):
        painter = QPainter(self)
        track = self._track()

        painter.fillRect(track, QColor(style.FIELD))
        painter.setPen(QColor(style.BORDER))
        painter.drawRect(track.adjusted(0, 0, -1, -1))

        if self._duration <= 0:
            painter.setPen(QColor(style.TEXT_DIM))
            painter.drawText(track, Qt.AlignCenter, "nothing to scrub")
            return

        self._paint_ruler(painter, track)
        if self._marking:
            self._paint_selection(painter, track)
            self._paint_handle(painter, track)
        if self._playhead:
            x = self._x_for(self._position)
            painter.setPen(QPen(QColor(style.MAGENTA), 1))
            painter.drawLine(x, 1, x, track.bottom() + 2)

    def _paint_ruler(self, painter: QPainter, track: QRect) -> None:
        step = _tick_step(self._duration)
        font = painter.font()
        font.setPointSize(7)
        painter.setFont(font)

        at = 0
        while at <= self._duration:
            x = self._x_for(at)
            painter.setPen(QColor(style.BORDER_HI))
            painter.drawLine(x, track.top() - 3, x, track.top() - 1)
            minutes, seconds = divmod(at // 1000, 60)
            label = f"{minutes}:{seconds:02d}" if step >= 1000 else f"{at / 1000:.1f}s"
            box = QRect(x + 2, 0, 44, RULER_H - 3)
            if box.right() <= track.right():
                painter.setPen(QColor(style.TEXT_DIM))
                painter.drawText(box, Qt.AlignLeft | Qt.AlignVCenter, label)
            at += step

    def _paint_selection(self, painter: QPainter, track: QRect) -> None:
        left, right = self._x_for(self._in), self._x_for(self._out)
        fill = QColor(style.ACCENT)
        fill.setAlpha(58)
        painter.fillRect(QRect(left, track.top(), max(1, right - left), track.height()), fill)

        # Everything outside the section is what will not be sent, so it reads as cut away.
        shade = QColor(0, 0, 0, 96)
        painter.fillRect(QRect(track.left(), track.top(), left - track.left(),
                               track.height()), shade)
        painter.fillRect(QRect(right, track.top(), track.right() - right + 1,
                               track.height()), shade)

    def _paint_handle(self, painter: QPainter, track: QRect) -> None:
        lit = self._drag == "in" or (not self._drag and self._hover == "in")
        colour = QColor(style.ACCENT if lit else style.BORDER_HI)

        x = self._x_for(self._in)
        rect = QRect(x - HANDLE_W // 2, track.top() - 3, HANDLE_W, track.height() + 6)
        painter.fillRect(rect, colour)
        painter.setPen(QColor(style.BG_DEEP))
        painter.drawLine(x, rect.top() + 5, x, rect.bottom() - 5)

        # The far edge is where the section runs out, which the duration parameter decides
        # rather than this widget -- so it is drawn as a line, not as something to grab.
        end = self._x_for(self._out)
        painter.setPen(QPen(colour, 1))
        painter.drawLine(end, track.top(), end, track.bottom())

    # -- interaction ---------------------------------------------------------------

    def mousePressEvent(self, event):
        if event.button() != Qt.LeftButton or self._duration <= 0:
            return
        self.setFocus(Qt.MouseFocusReason)
        self._drag = self._hit(int(event.position().x()))
        self._apply_drag(int(event.position().x()))

    def mouseMoveEvent(self, event):
        x = int(event.position().x())
        if self._drag:
            self._apply_drag(x)
            return
        hover = self._hit(x)
        if hover != self._hover:
            self._hover = hover
            self.setCursor(Qt.SizeHorCursor if hover == "in" else Qt.PointingHandCursor)
            self.update()

    def mouseReleaseEvent(self, event):
        if self._drag:
            self._drag = ""
            self.update()
            self.edited.emit()

    def leaveEvent(self, event):
        self._hover = ""
        self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event):
        if self._duration <= 0:
            super().wheelEvent(event)
            return
        frames = self._wheel.frames(event)
        event.accept()          # accepted even at zero, or a trackpad scrolls the panel
        if frames:
            self.stepped.emit(frames)

    def keyPressEvent(self, event):
        if event.key() not in (Qt.Key_Left, Qt.Key_Right):
            super().keyPressEvent(event)
            return
        step = self.step_ms * (COARSE_STEP if event.modifiers() & Qt.ShiftModifier else 1)
        if event.key() == Qt.Key_Left:
            step = -step
        self.scrubbed.emit(max(0, min(self._duration, self._position + step)))

    def _apply_drag(self, x: int) -> None:
        milliseconds = self._ms_at(x)
        if self._drag == "in":
            self._in = milliseconds
            self.update()
            self.in_moved.emit(self._in)
        elif self._drag == "playhead":
            self._position = milliseconds
            self.update()
            self.scrubbed.emit(milliseconds)


class TimelineBar(QWidget):
    """The track, plus the two controls that only mean anything while marking.

    Bound to whatever the result frame is showing: a player alone for a finished video, or
    a reference row and the player showing it. A row with no local copy -- one restored
    from history, say -- can still be bound without a player: the number stays editable,
    and only the parts that need pixels go quiet.
    """

    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.row = None
        self._player: QMediaPlayer | None = None
        self._duration_ms = 0
        self._syncing = False
        #: The generated length, which is what a section runs for. Pushed in by the window
        #: whenever the duration parameter changes; seeded so a bar shown before that
        #: happens still draws something truthful.
        self._target_frames = mathmirror.frames_from_seconds(config.DEFAULT_DURATION)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(3)

        self.track = TimelineTrack()
        self.track.in_moved.connect(self._on_in_moved)
        self.track.scrubbed.connect(self._on_scrubbed)
        self.track.stepped.connect(self.step_frames)
        self.track.edited.connect(self._commit)
        layout.addWidget(self.track)
        layout.addLayout(self._build_marks())

        shortcut = QShortcut(QKeySequence("I"), self)
        shortcut.setContext(Qt.WidgetWithChildrenShortcut)
        shortcut.activated.connect(self.mark_in)

    def _build_marks(self) -> QHBoxLayout:
        """The marking controls, which are hidden unless there is a reference to mark."""
        marks = QHBoxLayout()
        marks.setContentsMargins(0, 0, 0, 0)
        marks.setSpacing(6)

        self.mark_label = QLabel("Starts at")
        self.mark_label.setProperty("role", "hint")
        marks.addWidget(self.mark_label)

        self.start_spin = QDoubleSpinBox()
        self.start_spin.setKeyboardTracking(False)
        self.start_spin.setFixedWidth(84)
        self.start_spin.setToolTip("The in point, in the unit this reference is cut by")
        self.start_spin.valueChanged.connect(self._on_spin_changed)
        marks.addWidget(self.start_spin)

        self.mark_in_button = self._button(
            marks, "Mark in", "Set the in point at the playhead  (I)", self.mark_in)
        self.play_window_button = self._button(
            marks, "Play section", "Play the section that will be sent, on a loop",
            self.play_window)
        self.reset_button = self._button(
            marks, "Reset", "Start at the beginning of the reference again", self.reset)
        marks.addStretch(1)

        self._mark_widgets = (self.mark_label, self.start_spin, self.mark_in_button,
                              self.play_window_button, self.reset_button)
        for widget in self._mark_widgets:
            widget.hide()
        return marks

    def _button(self, layout, text: str, tip: str, slot) -> QToolButton:
        button = QToolButton()
        button.setText(text)
        button.setToolTip(tip)
        button.clicked.connect(slot)
        layout.addWidget(button)
        return button

    # -- binding -------------------------------------------------------------------

    def bind(self, row, player: QMediaPlayer | None) -> None:
        """Follow `row` (None for anything that is not a markable reference) and `player`."""
        if row is self.row and player is self._player:
            self.refresh()
            return
        self._release()
        self.row = row if (row is None or row.supports_trim) else None
        self._player = player
        if player is not None:
            player.durationChanged.connect(self._on_player_duration)
            player.positionChanged.connect(self._on_player_position)
            self._duration_ms = player.duration()
        self.refresh()

    @property
    def player(self) -> QMediaPlayer | None:
        """The player this track is following, if it is following one at all."""
        return self._player

    def unbind(self) -> None:
        """Stop following anything. The track stays, empty, because it is the transport."""
        self._release()
        self.refresh()

    def _release(self) -> None:
        if self._player is not None:
            try:
                self._player.durationChanged.disconnect(self._on_player_duration)
                self._player.positionChanged.disconnect(self._on_player_position)
            except (RuntimeError, TypeError):      # already gone; nothing to release
                pass
        self.row = None
        self._player = None
        self._duration_ms = 0

    def set_target_frames(self, frames: int) -> None:
        """Follow the generated length, which is what a section runs for.

        Changing *Duration* moves the far edge of every section, so the track has to be
        redrawn -- there is nothing on the row to change.
        """
        frames = max(1, int(frames))
        if frames == self._target_frames:
            return
        self._target_frames = frames
        if self.row is not None:
            self.refresh()

    # -- units ---------------------------------------------------------------------

    def _fps(self) -> float:
        """The frame rate the mark is counted in, measured if it was never probed."""
        row = self.row
        if row is None:
            return 0.0
        if row.fps:
            return float(row.fps)
        if row.frame_count and self._duration_ms > 0:
            return row.frame_count / (self._duration_ms / 1000.0)
        return 0.0

    def _usable(self) -> bool:
        """Whether an edit here can be expressed in the unit the row stores."""
        if self.row is None:
            return False
        return self._fps() > 0 if self.row.kind == VIDEO else True

    def _to_ms(self, value: float) -> int:
        if self.row is not None and self.row.kind == VIDEO:
            fps = self._fps()
            return int(round(value / fps * 1000)) if fps else 0
        return int(round(value * 1000))

    def _from_ms(self, milliseconds: int) -> float:
        if self.row is not None and self.row.kind == VIDEO:
            fps = self._fps()
            return float(round(milliseconds / 1000.0 * fps)) if fps else 0.0
        return round(milliseconds / 1000.0, 2)

    def duration_ms(self) -> int:
        """The length of whatever is bound, from the player or from the probe."""
        if self._duration_ms > 0:
            return self._duration_ms
        row = self.row
        if row is None:
            return 0
        if row.duration_s:
            return int(row.duration_s * 1000)
        if row.kind == VIDEO and row.frame_count and row.fps:
            return int(row.frame_count / row.fps * 1000)
        return 0

    def section_ms(self) -> int:
        """How long the section runs, in milliseconds of *this* reference's timeline.

        A video's section is counted in source frames, so at 30 fps a 124-frame section is
        4.1 seconds of the file even though it becomes 5.2 seconds of output. The track
        being drawn on is the file's, so this is the length that belongs on it.
        """
        row = self.row
        if row is None:
            return 0
        return self._to_ms(row.trim_length(self._target_frames))

    def window_ms(self) -> tuple[int, int]:
        """The section in milliseconds, clamped to the file."""
        row = self.row
        if row is None:
            return 0, self.duration_ms()
        duration = self.duration_ms()
        start = self._to_ms(row.trim_start)
        end = start + self.section_ms()
        return start, min(end, duration) if duration > 0 else end

    # -- editing -------------------------------------------------------------------

    def mark_in(self) -> None:
        if self._player is not None and self.row is not None:
            self._set_start(self._player.position())

    def play_window(self) -> None:
        if self._player is None:
            return
        start, _ = self.window_ms()
        self._player.setPosition(start)
        self._player.play()

    def reset(self) -> None:
        if self.row is None:
            return
        self.row.trim_start = 0.0
        self.refresh()
        self.changed.emit()

    def _set_start(self, start_ms: int) -> None:
        if self.row is None or not self._usable():
            return
        self.row.trim_start = self._from_ms(max(0, start_ms))
        self.refresh()
        self.changed.emit()

    def _on_spin_changed(self, _value) -> None:
        if self._syncing or self.row is None:
            return
        self.row.trim_start = self.start_spin.value()
        self.refresh()
        self.changed.emit()

    def _on_in_moved(self, milliseconds: int) -> None:
        """Follow the handle live, without announcing the edit until the drag ends."""
        if self.row is None or not self._usable():
            return
        self.row.trim_start = self._from_ms(max(0, milliseconds))

        # Seeking to the mark being dragged is what makes marking possible at all: you
        # need to see the frame you are cutting on.
        if self._player is not None:
            self._player.setPosition(milliseconds)
        self._refresh_readouts()

    def _on_scrubbed(self, milliseconds: int) -> None:
        if self._player is not None:
            self._player.setPosition(milliseconds)
        self.track.set_position(milliseconds)

    def step_frames(self, frames: int) -> None:
        """Move the playhead by whole frames. Positive is forwards.

        Playback is stopped first. Stepping through frames and running at speed are two
        different things to be doing, and leaving the clip playing means the next position
        update overwrites where the step just landed -- so the frame you asked for appears
        and is gone before you see it.
        """
        if not frames or self.duration_ms() <= 0:
            return

        if self._player is not None \
                and self._player.playbackState() == QMediaPlayer.PlayingState:
            self._player.pause()

        # From the player rather than from the track: the track is redrawn from position
        # updates that arrive late, so stepping off its value would drift backwards.
        now = (self._player.position() if self._player is not None
               else self.track.position())
        step = self._frame_ms()
        # A millisecond short of the end on purpose: landing exactly on the duration puts
        # the player at EndOfMedia, and the frame parks itself back at zero -- so stepping
        # onto the last frame would look like the clip jumping to the start.
        last = max(0, self.duration_ms() - 1)
        target = max(0, min(last, int(round(now + frames * step))))
        self._on_scrubbed(target)

    def _frame_ms(self) -> float:
        """One frame, in milliseconds of whatever is on screen.

        A bound reference knows its own rate. Anything else -- a finished result, a pose
        clip, a preview -- is something this app generated at config.FPS, which is a far
        better guess than a fixed number and exact for the case that matters most.
        """
        fps = self._fps() or float(config.FPS)
        return 1000.0 / fps if fps > 0 else 40.0

    def _commit(self) -> None:
        """A drag finished. Now the rest of the window can be told."""
        if self.row is None:
            return
        self.refresh()
        self.changed.emit()

    # -- following the player ------------------------------------------------------

    def _on_player_duration(self, milliseconds: int) -> None:
        self._duration_ms = max(0, int(milliseconds))
        self.refresh()

    def _on_player_position(self, milliseconds: int) -> None:
        self.track.set_position(milliseconds)
        self._confine(milliseconds)

    def _confine(self, milliseconds: int) -> None:
        """Keep playback inside the section, the way an editor's in/out loop does."""
        if self.track.is_dragging() or self._player is None or self.row is None:
            return
        if self._player.playbackState() != QMediaPlayer.PlayingState:
            return
        start, end = self.window_ms()
        if end > start and (milliseconds >= end or milliseconds < start - LOOP_SLACK_MS):
            self._player.setPosition(start)

    # -- display -------------------------------------------------------------------

    def refresh(self) -> None:
        """Re-show what is bound: the spin box's range, the track, and the tooltip."""
        row = self.row
        marking = row is not None
        usable = self._usable()
        for widget in self._mark_widgets:
            widget.setVisible(marking)

        if marking:
            self._syncing = True
            try:
                limit = row.trim_limit()
                if row.kind == VIDEO:
                    self.start_spin.setDecimals(0)
                    self.start_spin.setSingleStep(1)
                    self.start_spin.setSuffix(" f")
                    self.start_spin.setRange(0, limit or 999999)
                else:
                    self.start_spin.setDecimals(2)
                    self.start_spin.setSingleStep(0.1)
                    self.start_spin.setSuffix(" s")
                    self.start_spin.setRange(0, limit or 86400.0)
                self.start_spin.setValue(row.trim_start)
                self.start_spin.setEnabled(usable)
            finally:
                self._syncing = False

        # The arrow keys and the wheel move by the same amount, from the same place.
        self.track.step_ms = int(round(self._frame_ms()))
        self.track.set_duration(self.duration_ms())
        self.track.set_marking(marking and usable)
        self.track.set_playhead_visible(self._player is not None)
        self.track.setEnabled(self._player is not None or (marking and usable))

        # Marking happens at the playhead, and playing needs something to play, so both
        # go quiet for a reference with no local copy to show.
        live = self._player is not None and usable
        for button in (self.mark_in_button, self.play_window_button):
            button.setEnabled(live)
        self._refresh_readouts()

    def _refresh_readouts(self) -> None:
        start_ms, end_ms = self.window_ms()
        self.track.set_window(start_ms, end_ms)
        if self.row is not None:
            self._syncing = True
            try:
                self.start_spin.setValue(self.row.trim_start)
            finally:
                self._syncing = False
        self.track.setToolTip(self.describe())

    def describe(self) -> str:
        """What will actually be sent -- the track's tooltip, so it costs no room."""
        row = self.row
        if row is None:
            return f"Drag to scrub.\n{WHEEL_HINT}"
        how = ("Drag the handle to move the in point, or press I to set it at the "
               f"playhead.\nDrag anywhere else to scrub.\n{WHEEL_HINT}\n")
        if not self._usable():
            return how + "Waiting for the frame rate to be read."

        start_ms, end_ms = self.window_ms()
        span = f"{format_timecode(start_ms)} - {format_timecode(end_ms)}"

        # The section is as long as the generated clip unless the reference runs out
        # first, and saying which of those you are looking at is the point of drawing it
        # at all: the fix is to start earlier, or to generate less.
        asked = self.section_ms()
        short = asked - (end_ms - start_ms)
        if asked > 0 and short > 1:
            return (f"{how}Sending {span} - short by {format_timecode(short)}; "
                    "the reference runs out before the clip does.")

        _, length = row.trim_span(self._target_frames)
        if row.kind == VIDEO:
            seconds = row.trim_seconds(self._target_frames)
            tail = f"{int(length)} frames" + (f", {seconds[1]:.2f}s" if seconds else "")
        else:
            tail = f"{length:.2f}s"
        return f"{how}Sending {span}  ({tail})."
