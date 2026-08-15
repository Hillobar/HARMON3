"""The reference loaders: dynamic lists of images, videos and audio.

Each row carries the prompt tag the model will use for it. Tags are recomputed by the
main window on every structural change, because they depend on the whole set rather than
on any single row.

Rows are dragged by the grip on their left to reorder them, which is the only way to
choose which reference gets which ordinal: the model assigns `<Picture 1>` and the rest
purely from the order it receives them in. Reordering therefore renumbers tags, and the
prompt editor's existing "Rewrite tags in prompt" banner is what offers to follow --
`refs.tag_migration` matches rows by uid, so a move reports exactly what changed.

Reordering is within a kind. An image can only ever be a `<Picture i>`, so dropping one
among the videos would be asking for a tag that cannot exist.
"""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QMimeData, QPoint, QSize, Qt, Signal  # noqa: F401  (QSize used by THUMB)
from PySide6.QtGui import QColor, QDrag, QPainter, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QFileDialog,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSlider,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .. import imaging, preview, refs as refs_mod, scaling
from ..refs import AUDIO, IMAGE, KIND_LIMITS, VIDEO, VIDEO_EXTENSIONS, RefRow, RefSet
from . import style

THUMB = QSize(56, 40)

FILTERS = {
    IMAGE: "Images (*.png *.jpg *.jpeg *.webp *.bmp *.gif *.tif *.tiff);;All files (*)",
    VIDEO: "Videos (*.mp4 *.webm *.mkv *.mov *.avi *.m4v);;All files (*)",
    AUDIO: "Audio (*.wav *.mp3 *.flac *.ogg *.m4a *.aac *.opus);;All files (*)",
}

TITLES = {IMAGE: "Images", VIDEO: "Videos", AUDIO: "Audio"}

#: Carries the dragged row's uid. A private type rather than text/plain so a stray drag
#: from a text editor cannot be mistaken for a reorder, and so each list can refuse a row
#: belonging to another kind.
MIME_REF_ROW = "application/x-harmon3-ref-row"

#: How thick the "it would land here" line is.
DROP_INDICATOR_HEIGHT = 2

_thumb_cache: dict[str, QPixmap] = {}


def thumbnail_for(path: str) -> QPixmap | None:
    """A thumbnail for any local reference, video included.

    Decoding a video's first frame is not free, and rows are rebuilt on every structural
    change, so results are kept.
    """
    if path in _thumb_cache:
        return _thumb_cache[path]

    pixmap = imaging.load_pixmap(path)
    if pixmap.isNull() and Path(path).suffix.lower() in VIDEO_EXTENSIONS:
        image = preview.first_frame(path)
        if image is not None:
            pixmap = QPixmap.fromImage(image)

    result = None if pixmap.isNull() else pixmap
    if result is not None:
        _thumb_cache[path] = result
    return result


class DragHandle(QWidget):
    """The grip a reference row is reordered by.

    A dedicated handle rather than dragging the row itself: a row carries a size slider, a
    thumbnail that opens the result frame and two checkboxes, and a press-and-move on any
    of those means something already. The handle is the one place where it can only mean
    "move this".

    The dots are painted rather than set as text. A glyph would depend on the mono font
    actually carrying it -- this app's does not resolve to an exact match on a machine
    without JetBrains Mono installed, and a substituted box in place of the handle is
    worse than no handle at all.
    """

    drag_started = Signal(object)          # the RefRowWidget being dragged

    DOT = 2
    GAP = 3
    ROWS = 3
    COLUMNS = 2

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setCursor(Qt.OpenHandCursor)
        self.setToolTip(
            "Drag to reorder.\n\n"
            "The model numbers references by the order it receives them, so moving a row\n"
            "changes which tag it carries."
        )
        self.setFixedSize(
            self.COLUMNS * self.DOT + (self.COLUMNS - 1) * self.GAP + 6,
            self.ROWS * self.DOT + (self.ROWS - 1) * self.GAP + 6,
        )
        self._press_at: QPoint | None = None
        self._hovered = False

    def enterEvent(self, event):
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing, False)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(style.ACCENT if self._hovered else style.TEXT_DIM))

        span_x = self.COLUMNS * self.DOT + (self.COLUMNS - 1) * self.GAP
        span_y = self.ROWS * self.DOT + (self.ROWS - 1) * self.GAP
        left = (self.width() - span_x) // 2
        top = (self.height() - span_y) // 2
        for column in range(self.COLUMNS):
            for row in range(self.ROWS):
                painter.drawRect(left + column * (self.DOT + self.GAP),
                                 top + row * (self.DOT + self.GAP),
                                 self.DOT, self.DOT)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self._press_at = event.position().toPoint()
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Only past the platform's drag threshold, so a click that wobbles by a pixel does
        # not become a drag the user never asked for.
        if self._press_at is not None and (
            (event.position().toPoint() - self._press_at).manhattanLength()
            >= QApplication.startDragDistance()
        ):
            self._press_at = None
            self.drag_started.emit(self.parent())
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        self._press_at = None
        super().mouseReleaseEvent(event)


class TagChip(QLabel):
    """A prompt tag that inserts itself into the prompt when clicked."""

    clicked = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("role", "tag")
        style.mono(self, size=8)
        self.setCursor(Qt.PointingHandCursor)
        self.setToolTip("Click to insert this tag into the prompt at the cursor")

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.text():
            self.clicked.emit(self.text())
        super().mousePressEvent(event)


class Thumbnail(QLabel):
    """A reference thumbnail. Click to show it in the result frame."""

    clicked = Signal(str, bool)     # which ("source" | "pose"), additive

    def __init__(self, which: str, parent=None):
        super().__init__(parent)
        self.which = which
        self.setFixedSize(THUMB)
        self.setAlignment(Qt.AlignCenter)
        self.setCursor(Qt.PointingHandCursor)
        self._set_tooltip()

    def _set_tooltip(self) -> None:
        what = "the source" if self.which == "source" else "the pose clip"
        self.setToolTip(f"Click to show {what} in the result frame.\n"
                        "Shift-click to show it alongside what is already there.")

    def set_pending(self, note: str) -> None:
        """Say that clicking will *make* the thing rather than show it."""
        self.setToolTip(note)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            additive = bool(event.modifiers() & Qt.ShiftModifier)
            self.clicked.emit(self.which, additive)
        super().mousePressEvent(event)


class RefRowWidget(QFrame):
    """One reference. The same widget serves all three kinds; only the extras differ."""

    changed = Signal()
    removed = Signal(object)
    soundtrack_toggled = Signal()
    tag_clicked = Signal(str)
    #: The list owns the file dialog, so it can apply and update the per-kind last folder.
    replace_requested = Signal(object)
    preview_requested = Signal(object, str, bool)   # row, which, additive
    #: This image's size ceiling moved. Carries the row, so the result frame can
    #: re-render it at the new size while the slider is still being dragged.
    scale_changed = Signal(object)
    #: The grip was dragged. The list owns the drag, because it is the only thing that
    #: knows what a valid destination is.
    drag_started = Signal(object)

    def __init__(self, row: RefRow, parent=None):
        super().__init__(parent)
        self.row = row
        self.setProperty("role", "row")

        outer = QVBoxLayout(self)
        outer.setContentsMargins(6, 5, 6, 5)
        outer.setSpacing(3)

        top = QHBoxLayout()
        top.setSpacing(8)

        self.drag_handle = DragHandle(self)
        self.drag_handle.drag_started.connect(lambda _w: self.drag_started.emit(self))
        top.addWidget(self.drag_handle, 0, Qt.AlignTop)

        self.tag_chip = TagChip()
        self.tag_chip.clicked.connect(self.tag_clicked.emit)
        top.addWidget(self.tag_chip, 0, Qt.AlignTop)

        self.thumb = Thumbnail("source")
        self.thumb.clicked.connect(
            lambda which, additive: self.preview_requested.emit(self.row, which, additive))
        top.addWidget(self.thumb, 0, Qt.AlignTop)

        # The second slot appears only once a skeleton has actually been rendered, so a
        # posed row shows the source and what will be sent in its place, side by side.
        self.pose_thumb = Thumbnail("pose")
        self.pose_thumb.clicked.connect(
            lambda which, additive: self.preview_requested.emit(self.row, which, additive))
        self.pose_thumb.hide()
        top.addWidget(self.pose_thumb, 0, Qt.AlignTop)

        text_column = QVBoxLayout()
        text_column.setSpacing(1)
        self.name_label = style.ElidedLabel(row.display_name)
        text_column.addWidget(self.name_label)

        # Elided rather than clipped: the metadata line carries the trim window as well as
        # the dimensions now, and a narrow references column would otherwise cut it off
        # mid-word with nothing to say what was lost.
        self.detail_label = style.ElidedLabel("")
        self.detail_label.setProperty("role", "hint")
        text_column.addWidget(self.detail_label)
        top.addLayout(text_column, 1)

        # No Trim button: everything with a timeline is cut to the generated length, so
        # there is nothing to switch on, and clicking the row already opens it in the
        # result frame where the in point is marked.
        replace_button = QToolButton()
        replace_button.setText("...")
        replace_button.setToolTip("Choose a different file")
        replace_button.clicked.connect(lambda: self.replace_requested.emit(self))
        top.addWidget(replace_button, 0, Qt.AlignTop)

        remove_button = QToolButton()
        remove_button.setText("X")
        remove_button.setToolTip("Remove this reference")
        remove_button.clicked.connect(lambda: self.removed.emit(self))
        top.addWidget(remove_button, 0, Qt.AlignTop)

        outer.addLayout(top)

        self.soundtrack_box = None
        self.pose_box = None
        self.scale_slider = None
        if row.supports_scale:
            outer.addLayout(self._build_scale_row())
        if row.kind == VIDEO:
            bottom = QHBoxLayout()
            bottom.setContentsMargins(0, 0, 0, 0)
            self.soundtrack_box = QCheckBox("Use its soundtrack")
            self.soundtrack_box.setChecked(row.use_soundtrack)
            self.soundtrack_box.setToolTip(
                "Feed the video's own audio to the model alongside its frames.\n"
                "This adds an <Audio> tag, which renumbers any standalone audio after it."
            )
            self.soundtrack_box.toggled.connect(self._on_soundtrack_toggled)
            bottom.addWidget(self.soundtrack_box)

            self.soundtrack_chip = TagChip()
            self.soundtrack_chip.clicked.connect(self.tag_clicked.emit)
            bottom.addWidget(self.soundtrack_chip)

            self.pose_box = QCheckBox("Pose")
            self.pose_box.setChecked(row.use_pose)
            self.pose_box.setToolTip(
                "Send a skeleton of this clip instead of the clip: the movement without\n"
                "the person. Rendered here, before the run is queued, over just the\n"
                "section that will be sent. Its soundtrack is carried across unchanged."
            )
            self.pose_box.toggled.connect(self._on_pose_toggled)
            bottom.addWidget(self.pose_box)

            bottom.addStretch(1)
            outer.addLayout(bottom)

        self.warning_label = QLabel("")
        self.warning_label.setWordWrap(True)
        self.warning_label.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.warning_label.setMinimumWidth(0)
        self.warning_label.setProperty("role", "warn")
        self.warning_label.hide()
        outer.addWidget(self.warning_label)

        self._load_thumbnail()

    # -- state ---------------------------------------------------------------------

    def _on_soundtrack_toggled(self, checked: bool) -> None:
        self.row.use_soundtrack = checked
        self.soundtrack_toggled.emit()

    def _build_scale_row(self):
        """The per-image size ceiling: a slider, its percentage, and what it comes to.

        Only images have one. A reference video never consults the node's sizing -- every
        clip is fitted to a fixed canvas -- so a slider on one would promise something it
        could not deliver.
        """
        row = QHBoxLayout()
        row.setContentsMargins(0, 0, 0, 0)
        row.setSpacing(6)

        label = style.hint("Size")
        row.addWidget(label)

        self.scale_slider = QSlider(Qt.Horizontal)
        self.scale_slider.setRange(scaling.MIN_PERCENT, scaling.MAX_PERCENT)
        self.scale_slider.setValue(scaling.clamp_percent(self.row.scale_percent))
        self.scale_slider.setFixedWidth(96)
        self.scale_slider.setToolTip(self._scale_tooltip())
        self.scale_slider.valueChanged.connect(self._on_scale_changed)
        row.addWidget(self.scale_slider)

        self.scale_percent_label = style.hint("")
        style.mono(self.scale_percent_label, size=8)
        self.scale_percent_label.setFixedWidth(34)
        row.addWidget(self.scale_percent_label)

        self.scale_readout = style.ElidedLabel("")
        self.scale_readout.setProperty("role", "hint")
        row.addWidget(self.scale_readout, 1)

        self._refresh_scale_readout()
        return row

    def _on_scale_changed(self, value: int) -> None:
        self.row.scale_percent = scaling.clamp_percent(value)
        self._refresh_scale_readout()
        self.scale_changed.emit(self.row)

    def _scale_tooltip(self) -> str:
        """What the slider means, which is not quite the same for the two kinds."""
        if self.row.kind == VIDEO:
            return (
                "The frame size this clip is fed to the model at, as a share of the\n"
                "size it would otherwise use.\n\n"
                "A share of the model's own canvas rather than of the file: every clip\n"
                "above about a megapixel is fitted to the same 1344x768, so a 4K source\n"
                "and a 1080p one already arrive identical and a share of the file would\n"
                "do nothing. Smaller frames mean fewer tokens per frame, and a video\n"
                "reference is the expensive kind.\n\n"
                "The original file is never touched - a resized section is rendered\n"
                "before the run is queued, with its soundtrack carried across.")
        return (
            "How much of this image is sent, as a percentage of its own size.\n\n"
            "The model never enlarges a reference, so this is a ceiling: shrinking one\n"
            "here is the only way to size it separately from the others. Reference\n"
            "tokens run through every sampling step, so a smaller reference is a\n"
            "faster run at some cost in identity.\n\n"
            "The original file is never touched - a resized copy is made at submit time.")

    def _refresh_scale_readout(self) -> None:
        if self.scale_slider is None:
            return
        percent = scaling.clamp_percent(self.row.scale_percent)
        self.scale_percent_label.setText(f"{percent}%")
        describe = scaling.describe_video if self.row.kind == VIDEO else scaling.describe
        self.scale_readout.setText(describe(self.row.width, self.row.height, percent))
        # Amber once it is actually cutting something, so a row that will send less than
        # it holds says so at a glance rather than only in the number.
        self.scale_readout.setProperty("role", "warn" if percent < 100 else "hint")
        style.restyle(self.scale_readout)

    def _on_pose_toggled(self, checked: bool) -> None:
        self.row.use_pose = checked
        self.refresh_pose_thumbnail()
        self.soundtrack_toggled.emit()

    def refresh_pose_thumbnail(self) -> None:
        """The skeleton slot: the rendered clip, or the way to ask for one.

        It appears as soon as Pose is ticked, not once a render exists. Showing it only
        when there was something to show left no way to *get* the first one -- the render
        on demand was reachable in code and unreachable on screen.
        """
        self.pose_thumb.setVisible(self.row.poses)
        if not self.row.poses:
            return

        path = self.row.pose_path
        if path and Path(path).is_file():
            self._load_thumbnail_into(self.pose_thumb, path, "POSE")
            self.pose_thumb._set_tooltip()
            return

        self._load_thumbnail_into(self.pose_thumb, None, "POSE?")
        self.pose_thumb.set_pending(
            "No skeleton rendered yet.\n"
            "Click to render this reference's section now and watch it.\n"
            "Queueing renders it anyway, so this is only to see it first.")

    def set_file(self, path: str) -> None:
        """Point this row at a different file, discarding everything derived from the old one."""
        self.row.local_path = path
        self.row.comfy_name = None       # force a fresh upload
        self.row.duration_s = self.row.fps = self.row.frame_count = None
        self.row.has_audio = None
        self.row.server_missing = None
        # A skeleton of the file that used to be here says nothing about this one.
        self.row.pose_path = None
        self.name_label.setText(self.row.display_name)
        self._load_thumbnail()
        self.refresh_pose_thumbnail()
        self.changed.emit()

    def current_dir(self) -> str:
        """The folder this row's file lives in, if it has one."""
        return str(Path(self.row.local_path).parent) if self.row.local_path else ""

    def set_tags(self, tag: str, soundtrack_tag: str | None) -> None:
        self.tag_chip.setText(tag)
        if self.soundtrack_box is not None:
            self.soundtrack_chip.setText(soundtrack_tag or "")
            self.soundtrack_chip.setVisible(bool(soundtrack_tag))

    def set_used_in_prompt(self, used: bool) -> None:
        self.tag_chip.setProperty("muted", "false" if used else "true")
        self.tag_chip.setToolTip(
            "Click to insert this tag into the prompt at the cursor"
            if used else
            "This reference is not mentioned anywhere in the prompt - "
            "click to insert its tag at the cursor"
        )
        style.restyle(self.tag_chip)

    def refresh_details(self, target_frames: int) -> tuple[list[str], list[str]]:
        """Update the metadata line and warnings. Returns (errors, warnings)."""
        parts = []
        row = self.row
        if row.width and row.height:
            parts.append(f"{row.width}x{row.height}")
            if row.kind == IMAGE:
                # Megapixels next to the dimensions, because what a reference costs is an
                # area rather than a width: 4000x3000 and 6000x2000 read alike and are not.
                parts.append(scaling.format_megapixels(row.width, row.height))
        if row.duration_s:
            parts.append(f"{row.duration_s:.1f}s")
        if row.fps:
            parts.append(f"{row.fps:.3g} fps")
        if row.frame_count:
            parts.append(f"{row.frame_count} frames")
        if row.kind == VIDEO and row.has_audio is False:
            parts.append("no audio")
        if not row.needs_upload:
            parts.append("already on server")
        summary = row.trim_summary(target_frames)
        if summary:
            parts.append(summary)
        if row.poses:
            parts.append("posed")
        self.detail_label.setText("  -  ".join(parts))
        self.refresh_pose_thumbnail()
        # The probe delivers width and height after the row is built, so the readout is
        # empty until this runs.
        self._refresh_scale_readout()

        errors, warnings = refs_mod.row_warnings(row, target_frames)
        messages = [f"! {m}" for m in errors] + [f"~ {m}" for m in warnings]
        self.warning_label.setText("\n".join(messages))
        self.warning_label.setVisible(bool(messages))
        self.warning_label.setProperty("role", "error" if errors else "warn")
        style.restyle(self.warning_label)

        self.setProperty("invalid", "true" if errors else "false")
        style.restyle(self)
        return errors, warnings


    def show_server_error(self, message: str) -> None:
        self.warning_label.setText(f"! {message}")
        self.warning_label.setProperty("role", "error")
        self.warning_label.show()
        style.restyle(self.warning_label)
        self.setProperty("invalid", "true")
        style.restyle(self)

    def _load_thumbnail(self) -> None:
        glyph = {IMAGE: "IMG", VIDEO: "VID", AUDIO: "AUD"}[self.row.kind]
        self._load_thumbnail_into(self.thumb, self.row.local_path, glyph)

    def _load_thumbnail_into(self, target: QLabel, path: str | None, glyph: str) -> None:
        pixmap = thumbnail_for(path) if path else None
        if pixmap is not None and not pixmap.isNull():
            target.setPixmap(pixmap.scaled(THUMB, Qt.KeepAspectRatio, Qt.SmoothTransformation))
            target.setText("")
            return
        target.setPixmap(QPixmap())
        target.setText(glyph)
        target.setProperty("role", "hint")
        style.restyle(target)


class RefListWidget(QGroupBox):
    """One kind of reference: a header with a counter and an Add button, plus the rows."""

    changed = Signal()
    structure_changed = Signal()
    tag_clicked = Signal(str)
    preview_requested = Signal(object, str, bool)
    scale_changed = Signal(object)

    def __init__(self, kind: str, parent=None):
        super().__init__(parent)
        self.kind = kind
        self.limit = KIND_LIMITS[kind]
        self.rows: list[RefRowWidget] = []
        self._last_dir = ""
        #: The line showing where a dragged row would land, parented into rows_layout
        #: while a drag is over this list and taken out again when it leaves.
        self._indicator: QFrame | None = None
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 8)
        layout.setSpacing(5)

        header = QHBoxLayout()
        self.title_label = QLabel()
        self.title_label.setProperty("role", "section")
        header.addWidget(self.title_label)
        header.addStretch(1)

        self.add_button = QPushButton("+ Add")
        self.add_button.clicked.connect(self.browse_and_add)
        header.addWidget(self.add_button)
        layout.addLayout(header)

        self.rows_container = QWidget()
        self.rows_layout = QVBoxLayout(self.rows_container)
        self.rows_layout.setContentsMargins(0, 0, 0, 0)
        self.rows_layout.setSpacing(4)
        layout.addWidget(self.rows_container)

        self.empty_label = style.hint(f"No reference {TITLES[kind].lower()}.")
        layout.addWidget(self.empty_label)

        self._update_header()

    # -- population ----------------------------------------------------------------

    def set_rows(self, rows: list[RefRow]) -> None:
        for widget in list(self.rows):
            self._detach(widget)
        for row in rows[: self.limit]:
            self._attach(RefRowWidget(row))
        self._update_header()

    def model_rows(self) -> list[RefRow]:
        return [widget.row for widget in self.rows]

    def row_widgets(self) -> list[RefRowWidget]:
        return list(self.rows)

    def browse_and_add(self) -> None:
        if len(self.rows) >= self.limit:
            return
        paths, _ = QFileDialog.getOpenFileNames(
            self, f"Add reference {TITLES[self.kind].lower()}", self._start_dir(),
            FILTERS[self.kind])
        if paths:
            self.add_paths(paths)

    def _on_replace_requested(self, widget: RefRowWidget) -> None:
        # Prefer the folder this row's file is already in; fall back to the kind's last.
        start = widget.current_dir() or self._start_dir()
        path, _ = QFileDialog.getOpenFileName(
            self, f"Choose a reference {self.kind}", start, FILTERS[self.kind])
        if path:
            self._remember(path)
            widget.set_file(path)

    def add_row(self, row: RefRow) -> bool:
        """Take a row built elsewhere -- the result frame -- into the next free slot."""
        if len(self.rows) >= self.limit:
            return False
        self._attach(RefRowWidget(row))
        if row.local_path:
            self._remember(row.local_path)
        self._update_header()
        self.structure_changed.emit()
        return True

    def add_paths(self, paths) -> None:
        added = False
        for path in paths:
            if len(self.rows) >= self.limit:
                break
            self._attach(RefRowWidget(RefRow(kind=self.kind, local_path=str(path))))
            added = True
        if added:
            self._remember(paths[0])
            self._update_header()
            self.structure_changed.emit()

    def _start_dir(self) -> str:
        """Where the file dialog should open.

        A remembered folder that has since been removed would leave the dialog somewhere
        arbitrary, so it is only used while it still exists.
        """
        return self._last_dir if self._last_dir and Path(self._last_dir).is_dir() else ""

    def _remember(self, path) -> None:
        parent = Path(path).parent
        if parent.is_dir():
            self._last_dir = str(parent)

    def last_dir(self) -> str:
        return self._last_dir

    def set_last_dir(self, path: str) -> None:
        self._last_dir = path or ""

    # -- internals -----------------------------------------------------------------

    def _attach(self, widget: RefRowWidget) -> None:
        widget.removed.connect(self._on_removed)
        widget.changed.connect(self._on_row_changed)
        widget.soundtrack_toggled.connect(self.structure_changed.emit)
        widget.tag_clicked.connect(self.tag_clicked.emit)
        widget.replace_requested.connect(self._on_replace_requested)
        widget.preview_requested.connect(self.preview_requested.emit)
        widget.scale_changed.connect(self.scale_changed.emit)
        widget.drag_started.connect(self._start_drag)
        self.rows.append(widget)
        self.rows_layout.addWidget(widget)

    def _detach(self, widget: RefRowWidget) -> None:
        self.rows.remove(widget)
        self.rows_layout.removeWidget(widget)
        widget.setParent(None)
        widget.deleteLater()

    def _on_removed(self, widget: RefRowWidget) -> None:
        self._detach(widget)
        self._update_header()
        self.structure_changed.emit()

    def _on_row_changed(self) -> None:
        self._update_header()
        self.structure_changed.emit()

    # -- reordering ----------------------------------------------------------------

    def move_row(self, uid: int, to_index: int) -> bool:
        """Move the row with ``uid`` so it sits at ``to_index``. True if anything moved.

        The index is where the row ends up *after* the move, counted in the list as the
        user sees it -- so dragging row 0 down past row 1 and dropping "before row 2"
        means index 1, not 2. Taking the row out before clamping is what makes that come
        out right, and it is the difference between a drop landing where the indicator
        said it would and landing one place further on.
        """
        current = next((i for i, w in enumerate(self.rows) if w.row.uid == uid), None)
        if current is None:
            return False

        widget = self.rows.pop(current)
        target = max(0, min(to_index if to_index <= current else to_index - 1,
                            len(self.rows)))
        if target == current:
            self.rows.insert(current, widget)
            return False

        self.rows.insert(target, widget)
        # The layout is rebuilt from self.rows rather than nudged, so the two can never
        # disagree about the order -- which is what model_rows() then reports.
        for index, row_widget in enumerate(self.rows):
            self.rows_layout.insertWidget(index, row_widget)
        return True

    def _insertion_index(self, y: int) -> int:
        """Which slot a drop at ``y`` (in rows_container coordinates) would fall into."""
        for index, widget in enumerate(self.rows):
            if y < widget.geometry().center().y():
                return index
        return len(self.rows)

    def _start_drag(self, widget: RefRowWidget) -> None:
        payload = QMimeData()
        payload.setData(MIME_REF_ROW, str(widget.row.uid).encode("ascii"))

        drag = QDrag(widget)
        drag.setMimeData(payload)
        # The row itself as the cursor, so what is being moved is never in doubt.
        drag.setPixmap(widget.grab())
        drag.setHotSpot(QPoint(widget.width() // 2, widget.height() // 2))
        drag.exec(Qt.MoveAction)
        self._clear_indicator()

    def _dragged_uid(self, event) -> int | None:
        """The uid a drag carries, if it is one of *this* list's rows.

        A row from another kind is refused: an image can only ever be a `<Picture i>`, so
        there is no ordinal for it among the videos to move to.
        """
        data = event.mimeData()
        if not data.hasFormat(MIME_REF_ROW):
            return None
        try:
            uid = int(bytes(data.data(MIME_REF_ROW)).decode("ascii"))
        except (UnicodeDecodeError, ValueError):
            return None
        return uid if any(w.row.uid == uid for w in self.rows) else None

    def dragEnterEvent(self, event) -> None:
        # Ignored rather than refused when it is not ours, so a file dropped from Explorer
        # goes on up to RefPanel, which is what adds new references.
        if self._dragged_uid(event) is None:
            event.ignore()
            return
        event.acceptProposedAction()

    def dragMoveEvent(self, event) -> None:
        if self._dragged_uid(event) is None:
            event.ignore()
            return
        position = event.position().toPoint()
        self._show_indicator(self._insertion_index(
            self.rows_container.mapFrom(self, position).y()))
        event.acceptProposedAction()

    def dragLeaveEvent(self, event) -> None:
        self._clear_indicator()
        super().dragLeaveEvent(event)

    def dropEvent(self, event) -> None:
        uid = self._dragged_uid(event)
        if uid is None:
            event.ignore()
            return
        position = event.position().toPoint()
        index = self._insertion_index(self.rows_container.mapFrom(self, position).y())
        self._clear_indicator()
        event.acceptProposedAction()
        if self.move_row(uid, index):
            # The same signal an add or a remove sends: tags are recomputed from the whole
            # set, and the prompt editor offers its rewrite from the migration that falls
            # out of it.
            self.structure_changed.emit()

    def _show_indicator(self, index: int) -> None:
        """Draw the "it lands here" line at slot ``index``.

        An overlay rather than a widget in the layout: inserting one would push the rows
        down by its own height, which changes the geometries _insertion_index is reading
        and can leave the line oscillating between two slots at a boundary.
        """
        if self._indicator is None:
            self._indicator = QFrame(self.rows_container)
            self._indicator.setProperty("role", "drop-indicator")
            style.restyle(self._indicator)

        if not self.rows:
            y = 0
        elif index < len(self.rows):
            y = self.rows[index].geometry().top() - self.rows_layout.spacing() // 2
        else:
            y = self.rows[-1].geometry().bottom() + self.rows_layout.spacing() // 2

        self._indicator.setGeometry(
            0, max(0, y), self.rows_container.width(), DROP_INDICATOR_HEIGHT)
        self._indicator.raise_()
        self._indicator.show()

    def _clear_indicator(self) -> None:
        if self._indicator is None:
            return
        self._indicator.setParent(None)
        self._indicator.deleteLater()
        self._indicator = None

    def _update_header(self) -> None:
        self.title_label.setText(
            f"{TITLES[self.kind].upper()}  {len(self.rows)}/{self.limit}")
        self.add_button.setEnabled(len(self.rows) < self.limit)
        self.add_button.setToolTip(
            "" if len(self.rows) < self.limit
            else f"The model accepts at most {self.limit} reference {TITLES[self.kind].lower()}"
        )
        self.empty_label.setVisible(not self.rows)


class RefPanel(QWidget):
    """All three reference lists, plus drag-and-drop onto the right one by extension."""

    structure_changed = Signal()
    tag_clicked = Signal(str)
    preview_requested = Signal(object, str, bool)
    scale_changed = Signal(object)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAcceptDrops(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.lists = {}
        for kind in (IMAGE, VIDEO, AUDIO):
            widget = RefListWidget(kind)
            widget.structure_changed.connect(self.structure_changed.emit)
            widget.tag_clicked.connect(self.tag_clicked.emit)
            widget.preview_requested.connect(self.preview_requested.emit)
            widget.scale_changed.connect(self.scale_changed.emit)
            self.lists[kind] = widget
            layout.addWidget(widget)
        layout.addStretch(1)

    def set_refset(self, refset: RefSet) -> None:
        self.lists[IMAGE].set_rows(refset.images)
        self.lists[VIDEO].set_rows(refset.videos)
        self.lists[AUDIO].set_rows(refset.audios)

    def refset(self) -> RefSet:
        return RefSet(
            images=self.lists[IMAGE].model_rows(),
            videos=self.lists[VIDEO].model_rows(),
            audios=self.lists[AUDIO].model_rows(),
        )

    def all_row_widgets(self) -> list[RefRowWidget]:
        widgets = []
        for kind in (IMAGE, VIDEO, AUDIO):
            widgets.extend(self.lists[kind].row_widgets())
        return widgets

    def last_dirs(self) -> dict[str, str]:
        """The last folder browsed for each kind, kept apart.

        References of different kinds usually live in different places, so collapsing
        these into one would send you back to the wrong folder on every other click.
        """
        return {
            kind: widget.last_dir()
            for kind, widget in self.lists.items()
            if widget.last_dir()
        }

    def set_last_dirs(self, dirs) -> None:
        for kind, widget in self.lists.items():
            widget.set_last_dir((dirs or {}).get(kind, ""))

    # -- drag and drop -------------------------------------------------------------

    def dragEnterEvent(self, event):
        if event.mimeData().hasUrls():
            event.acceptProposedAction()

    def dropEvent(self, event):
        buckets = {IMAGE: [], VIDEO: [], AUDIO: []}
        for url in event.mimeData().urls():
            path = url.toLocalFile()
            if not path:
                continue
            kind = refs_mod.kind_for_path(path)
            if kind:
                buckets[kind].append(path)

        for kind, paths in buckets.items():
            if paths:
                self.lists[kind].add_paths(paths)
        event.acceptProposedAction()
