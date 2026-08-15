"""Visual theme: a dark, technical palette built around cyan and magenta.

Qt stylesheets support only a subset of CSS -- no box-shadow, no letter-spacing, no
text-transform -- so "glow" is simulated with border and background colour shifts, and
the uppercase panel headings are applied in stylise() rather than declared here.
"""

from __future__ import annotations

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QColor, QFont, QFontMetrics, QPalette
from PySide6.QtWidgets import QFrame, QGroupBox, QLabel, QSizePolicy

# --------------------------------------------------------------------------------------
# Palette
# --------------------------------------------------------------------------------------

BG_DEEP = "#080b11"       # window
BG = "#0c111a"            # panel ground
SURFACE = "#121a25"       # raised rows, buttons
SURFACE_HI = "#18222f"    # hover
FIELD = "#060910"         # inset inputs
BORDER = "#1c2735"
BORDER_HI = "#2a3a4d"

TEXT = "#ccd9e6"
TEXT_DIM = "#64788c"
MUTED = TEXT_DIM          # kept for existing callers

ACCENT = "#22d3ee"        # cyan: focus, primary action, live values
ACCENT_DEEP = "#0e7490"
MAGENTA = "#e879f9"       # reference tags, secondary highlight

OK = "#34d399"
WARN = "#fbbf24"
ERROR = "#fb5c73"

# Typography is set through QFont.setFamilies rather than the stylesheet: Qt's stylesheet
# parser does not handle a comma-separated font-family fallback list reliably, and a
# family that fails to resolve there silently lands on something with no glyphs.
MONO_FAMILIES = ["JetBrains Mono", "Cascadia Mono", "Consolas", "DejaVu Sans Mono",
                 "Courier New", "monospace"]
UI_FAMILIES = ["Segoe UI Variable Text", "Segoe UI", "Inter", "Noto Sans",
               "DejaVu Sans", "sans-serif"]


def _rgba(hex_colour: str, alpha: float) -> str:
    colour = QColor(hex_colour)
    return f"rgba({colour.red()}, {colour.green()}, {colour.blue()}, {alpha})"


STYLESHEET = f"""
QWidget {{
    background: transparent;
    color: {TEXT};
}}
QMainWindow, QDialog {{ background: {BG_DEEP}; }}

/* ---- panels ------------------------------------------------------------------ */
QGroupBox {{
    background: {BG};
    border: 1px solid {BORDER};
    border-radius: 3px;
    margin-top: 13px;
    padding-top: 10px;
    font-size: 10px;
    font-weight: 700;
    color: {TEXT_DIM};
}}
QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 10px;
    padding: 0 5px;
    background: {BG_DEEP};
    color: {ACCENT};
}}

/* A panel that fills a pane: the pane's title bar names it, so it has no heading of its
   own and no room to lose to the margin a heading would need. */
QGroupBox[role="pane"] {{
    margin-top: 0;
    padding-top: 6px;
}}

QTabWidget::pane {{
    border: 1px solid {BORDER};
    border-radius: 3px;
    background: {BG};
    top: -1px;
}}
QTabBar::tab {{
    background: transparent;
    color: {TEXT_DIM};
    border: none;
    padding: 6px 16px;
    font-size: 11px;
    font-weight: 600;
}}
/* The accent runs along the edge the tab shares with its contents, so the underline reads
   as belonging to the page. Docked panes can put their tab bar on either edge. */
QTabBar::tab:top    {{ border-bottom: 2px solid transparent; }}
QTabBar::tab:bottom {{ border-top: 2px solid transparent; }}
QTabBar::tab:hover {{ color: {TEXT}; }}
QTabBar::tab:selected:top {{
    color: {ACCENT};
    border-bottom: 2px solid {ACCENT};
}}
QTabBar::tab:selected:bottom {{
    color: {ACCENT};
    border-top: 2px solid {ACCENT};
}}

QSplitter::handle {{ background: {BORDER}; }}
QSplitter::handle:horizontal {{ width: 1px; }}
QSplitter::handle:vertical {{ height: 1px; }}
QSplitter::handle:hover {{ background: {ACCENT_DEEP}; }}

QScrollArea {{ background: transparent; border: none; }}

/* ---- buttons ----------------------------------------------------------------- */
QPushButton {{
    background: {SURFACE};
    border: 1px solid {BORDER_HI};
    border-radius: 3px;
    padding: 5px 13px;
    color: {TEXT};
}}
QPushButton:hover {{
    background: {SURFACE_HI};
    border-color: {ACCENT_DEEP};
    color: {ACCENT};
}}
QPushButton:pressed {{ background: {FIELD}; }}
QPushButton:disabled {{
    background: {_rgba(SURFACE, 0.45)};
    border-color: {BORDER};
    color: {_rgba(TEXT_DIM, 0.55)};
}}
QPushButton:flat {{ background: transparent; border: 1px solid transparent; }}
QPushButton:flat:hover {{ border-color: {BORDER_HI}; }}

QPushButton#primary {{
    background: {_rgba(ACCENT, 0.14)};
    border: 1px solid {ACCENT};
    color: {ACCENT};
    font-weight: 700;
    padding: 6px 22px;
}}
QPushButton#primary:hover {{ background: {_rgba(ACCENT, 0.26)}; }}
QPushButton#primary:disabled {{
    background: transparent;
    border-color: {BORDER_HI};
    color: {_rgba(TEXT_DIM, 0.55)};
}}

QToolButton {{
    background: {SURFACE};
    border: 1px solid {BORDER_HI};
    border-radius: 3px;
    padding: 2px 6px;
    color: {TEXT_DIM};
}}
QToolButton:hover {{ border-color: {ACCENT_DEEP}; color: {ACCENT}; }}
QToolButton:disabled {{
    background: {_rgba(SURFACE, 0.45)};
    border-color: {BORDER};
    color: {_rgba(TEXT_DIM, 0.55)};
}}
/* A tool button whose setting is currently in force, e.g. Trim on a windowed reference. */
QToolButton[role="on"] {{
    background: {_rgba(ACCENT, 0.14)};
    border-color: {ACCENT};
    color: {ACCENT};
}}

/* ---- inputs ------------------------------------------------------------------ */
QLineEdit, QPlainTextEdit, QTextEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background: {FIELD};
    border: 1px solid {BORDER};
    border-radius: 3px;
    padding: 4px 7px;
    color: {TEXT};
    selection-background-color: {ACCENT_DEEP};
    selection-color: #ffffff;
}}
QLineEdit:hover, QPlainTextEdit:hover, QSpinBox:hover,
QDoubleSpinBox:hover, QComboBox:hover {{ border-color: {BORDER_HI}; }}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {ACCENT};
    background: {_rgba(ACCENT, 0.05)};
}}
QComboBox::drop-down {{ border: none; width: 18px; }}
QComboBox QAbstractItemView {{
    background: {SURFACE};
    border: 1px solid {BORDER_HI};
    selection-background-color: {_rgba(ACCENT, 0.22)};
    selection-color: {ACCENT};
    outline: none;
}}
QPlainTextEdit, QTextEdit {{ padding: 7px; line-height: 145%; }}

QCheckBox {{ spacing: 7px; color: {TEXT}; }}
QCheckBox::indicator {{
    width: 13px; height: 13px;
    border: 1px solid {BORDER_HI};
    border-radius: 2px;
    background: {FIELD};
}}
QCheckBox::indicator:hover {{ border-color: {ACCENT_DEEP}; }}
QCheckBox::indicator:checked {{
    background: {ACCENT};
    border-color: {ACCENT};
}}

/* ---- readouts and status ----------------------------------------------------- */
QLabel {{ background: transparent; }}
QLabel[role="hint"]  {{ color: {TEXT_DIM}; }}
QLabel[role="ok"]    {{ color: {OK}; }}
QLabel[role="warn"]  {{ color: {WARN}; }}
QLabel[role="error"] {{ color: {ERROR}; }}

/* The same roles on a flat button, e.g. the scene indicator in the run bar. */
QPushButton[role="hint"]  {{ color: {TEXT_DIM}; }}
QPushButton[role="ok"]    {{ color: {OK}; }}
QPushButton[role="warn"]  {{ color: {WARN}; }}
QPushButton[role="error"] {{ color: {ERROR}; }}
QPushButton[role="ok"]:hover,
QPushButton[role="warn"]:hover,
QPushButton[role="hint"]:hover {{ color: {ACCENT}; }}

/* metric() and TagChip apply the monospaced family via QFont; only colour here. */
QLabel[role="metric"] {{
    font-weight: 700;
    color: {ACCENT};
}}

/* Matches the uppercase QGroupBox headings, for panels that build their own header. */
QLabel[role="section"] {{
    color: {ACCENT};
    font-size: 10px;
    font-weight: 700;
}}
QLabel[role="metric"][role2="error"] {{ color: {ERROR}; }}

QLabel[role="tag"] {{
    background: {_rgba(MAGENTA, 0.12)};
    color: {MAGENTA};
    border: 1px solid {_rgba(MAGENTA, 0.42)};
    border-radius: 2px;
    padding: 1px 6px;
}}
QLabel[role="tag"]:hover {{ background: {_rgba(MAGENTA, 0.24)}; }}
QLabel[role="tag"][muted="true"] {{
    background: transparent;
    color: {TEXT_DIM};
    border-color: {BORDER_HI};
}}

/* Where a dragged row would land. Deliberately the accent rather than the magenta of a
   tag: it says "here", not "this is a reference". */
QFrame[role="drop-indicator"] {{
    background: {ACCENT};
    border: none;
}}

/* A fragment of the output format, clicked into the prompt. Deliberately not the magenta
   of role="tag": those name a reference that actually exists, these are vocabulary. The
   shape is shared so both still read as "click to insert". */
QLabel[role="snippet"] {{
    background: {_rgba(ACCENT, 0.10)};
    color: {ACCENT};
    border: 1px solid {_rgba(ACCENT, 0.34)};
    border-radius: 2px;
    padding: 1px 6px;
}}
QLabel[role="snippet"]:hover {{ background: {_rgba(ACCENT, 0.22)}; }}

QFrame[role="banner"] {{
    background: {_rgba(WARN, 0.09)};
    border: 1px solid {_rgba(WARN, 0.45)};
    border-radius: 3px;
}}
QFrame[role="banner-error"] {{
    background: {_rgba(ERROR, 0.09)};
    border: 1px solid {_rgba(ERROR, 0.45)};
    border-radius: 3px;
}}

QFrame[role="row"] {{
    background: {SURFACE};
    border: 1px solid {BORDER};
    border-left: 2px solid {BORDER_HI};
    border-radius: 3px;
}}
QFrame[role="row"]:hover {{ border-left-color: {ACCENT_DEEP}; }}
QFrame[role="row"][invalid="true"] {{
    border-color: {_rgba(ERROR, 0.5)};
    border-left-color: {ERROR};
}}

QFrame[frameShape="4"] {{ background: {BORDER}; max-height: 1px; border: none; }}

/* ---- progress ---------------------------------------------------------------- */
QProgressBar {{
    background: {FIELD};
    border: 1px solid {BORDER};
    border-radius: 2px;
    height: 16px;
    text-align: center;
    color: {TEXT_DIM};
}}
QProgressBar::chunk {{ background: {_rgba(ACCENT, 0.5)}; border-radius: 1px; }}

/* ---- lists ------------------------------------------------------------------- */
QTreeWidget, QTreeView, QListWidget {{
    background: {FIELD};
    border: 1px solid {BORDER};
    border-radius: 3px;
    alternate-background-color: {_rgba(SURFACE, 0.5)};
    outline: none;
}}
QTreeWidget::item, QListWidget::item {{ padding: 4px 2px; border: none; }}
QTreeWidget::item:hover {{ background: {_rgba(ACCENT, 0.06)}; }}
QTreeWidget::item:selected, QListWidget::item:selected {{
    background: {_rgba(ACCENT, 0.16)};
    color: {TEXT};
}}
QHeaderView::section {{
    background: {BG};
    color: {TEXT_DIM};
    border: none;
    border-bottom: 1px solid {BORDER_HI};
    border-right: 1px solid {BORDER};
    padding: 5px 6px;
    font-size: 10px;
    font-weight: 700;
}}

/* ---- sliders and scrollbars -------------------------------------------------- */
QSlider::groove:horizontal {{
    background: {FIELD};
    border: 1px solid {BORDER};
    height: 3px;
    border-radius: 2px;
}}
QSlider::sub-page:horizontal {{ background: {ACCENT_DEEP}; border-radius: 2px; }}
QSlider::handle:horizontal {{
    background: {ACCENT};
    border: none;
    width: 9px;
    margin: -4px 0;
    border-radius: 2px;
}}

QScrollBar:vertical {{ background: transparent; width: 9px; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 9px; margin: 0; }}
QScrollBar::handle {{ background: {BORDER_HI}; border-radius: 4px; min-height: 24px; }}
QScrollBar::handle:hover {{ background: {ACCENT_DEEP}; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* ---- panes ------------------------------------------------------------------- */
QDockWidget {{
    color: {TEXT_DIM};
    font-size: 10px;
    font-weight: 700;
}}
QDockWidget::title {{
    background: {BG_DEEP};
    border-bottom: 1px solid {BORDER};
    color: {ACCENT};
    padding: 4px 8px 4px 10px;
    text-align: left;
}}
/* No close button rule: the panes cannot be closed, only moved and floated. */
QDockWidget::float-button {{
    background: transparent;
    border: 1px solid transparent;
    border-radius: 2px;
    subcontrol-origin: margin;
    subcontrol-position: top right;
    icon-size: 10px;
    width: 14px;
    height: 14px;
    top: 3px;
    right: 4px;
}}
QDockWidget::float-button:hover {{
    background: {SURFACE_HI};
    border-color: {ACCENT_DEEP};
}}
/* The grab strip between two panes. Wider than a splitter handle: it is the only way to
   resize, and a one-pixel target is not one. */
QMainWindow::separator {{
    background: {BORDER};
    width: 4px;
    height: 4px;
}}
QMainWindow::separator:hover {{ background: {ACCENT_DEEP}; }}

/* ---- chrome ------------------------------------------------------------------ */
QToolBar {{
    background: transparent;
    border: none;
    padding: 0;
    spacing: 6px;
}}
QStatusBar QPushButton {{ padding: 2px 10px; font-size: 11px; }}
QStatusBar {{
    background: {BG_DEEP};
    border-top: 1px solid {BORDER};
    color: {TEXT_DIM};
}}
QStatusBar::item {{ border: none; }}
QStatusBar QLabel {{ font-size: 11px; }}

QToolTip {{
    background: {SURFACE};
    color: {TEXT};
    border: 1px solid {ACCENT_DEEP};
    border-radius: 3px;
    padding: 5px 8px;
}}

QMenu {{
    background: {SURFACE};
    border: 1px solid {BORDER_HI};
    border-radius: 3px;
    padding: 4px;
}}
QMenu::item {{ padding: 5px 22px 5px 14px; border-radius: 2px; }}
QMenu::item:selected {{ background: {_rgba(ACCENT, 0.18)}; color: {ACCENT}; }}
QMenu::separator {{ height: 1px; background: {BORDER}; margin: 4px 8px; }}
"""


class ElidedLabel(QLabel):
    """A label that shortens its text to fit instead of forcing the panel wider.

    Reference filenames are long and arbitrary; without this one of them sets the
    minimum width of the whole references column.
    """

    def __init__(self, text: str = "", parent=None):
        super().__init__(parent)
        self._full = ""
        self.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
        self.setMinimumWidth(0)
        self.setText(text)          # routed through the override, so it elides at once

    def setText(self, text: str) -> None:
        self._full = text or ""
        self.setToolTip(self._full)
        self._apply_elision()

    def fullText(self) -> str:
        return self._full

    def minimumSizeHint(self):
        # QLabel derives its minimum from the text it currently holds. Pinning the width
        # to zero is what guarantees a long filename can never widen the whole column.
        return QSize(0, super().minimumSizeHint().height())

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply_elision()

    def _apply_elision(self) -> None:
        width = max(0, self.width() - 2)
        elided = QFontMetrics(self.font()).elidedText(self._full, Qt.ElideMiddle, width)
        super().setText(elided)


def hint(text: str = "") -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "hint")
    return label


def metric(text: str = "") -> QLabel:
    """A monospaced, accented readout for a computed value."""
    label = QLabel(text)
    label.setProperty("role", "metric")
    mono(label, weight=QFont.DemiBold)
    return label


def restyle(widget) -> None:
    """Re-apply the stylesheet after a dynamic property change."""
    widget.style().unpolish(widget)
    widget.style().polish(widget)


def separator() -> QFrame:
    line = QFrame()
    line.setFrameShape(QFrame.HLine)
    line.setFrameShadow(QFrame.Plain)
    return line


def mono(widget, size: int = 9, weight: int = QFont.Normal) -> None:
    """Give one widget the technical typeface, for values rather than prose."""
    font = QFont()
    font.setFamilies(MONO_FAMILIES)
    font.setStyleHint(QFont.Monospace)
    font.setPointSize(size)
    font.setWeight(weight)
    widget.setFont(font)


def stylise(root) -> None:
    """Apply the touches Qt stylesheets cannot express.

    Qt has no text-transform, so the uppercase panel headings that carry most of the
    technical feel have to be set on the widgets themselves.
    """
    for group in root.findChildren(QGroupBox):
        title = group.title()
        if title and not title.isupper():
            group.setTitle(title.upper())


def apply_theme(app) -> None:
    ui_font = QFont()
    ui_font.setFamilies(UI_FAMILIES)
    ui_font.setPointSize(9)
    app.setFont(ui_font)

    palette = QPalette()
    palette.setColor(QPalette.Window, QColor(BG_DEEP))
    palette.setColor(QPalette.WindowText, QColor(TEXT))
    palette.setColor(QPalette.Base, QColor(FIELD))
    palette.setColor(QPalette.AlternateBase, QColor(SURFACE))
    palette.setColor(QPalette.ToolTipBase, QColor(SURFACE))
    palette.setColor(QPalette.ToolTipText, QColor(TEXT))
    palette.setColor(QPalette.Text, QColor(TEXT))
    palette.setColor(QPalette.Button, QColor(SURFACE))
    palette.setColor(QPalette.ButtonText, QColor(TEXT))
    palette.setColor(QPalette.BrightText, QColor(MAGENTA))
    palette.setColor(QPalette.Link, QColor(ACCENT))
    palette.setColor(QPalette.Highlight, QColor(ACCENT_DEEP))
    palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.Mid, QColor(BORDER))
    palette.setColor(QPalette.Dark, QColor(BG))
    palette.setColor(QPalette.PlaceholderText, QColor(TEXT_DIM))
    for role in (QPalette.Text, QPalette.ButtonText, QPalette.WindowText):
        palette.setColor(QPalette.Disabled, role, QColor(TEXT_DIM))

    app.setStyle("Fusion")     # a predictable base for the stylesheet to build on
    app.setPalette(palette)
    app.setStyleSheet(STYLESHEET)


#: Previous name, kept so existing callers keep working.
apply_dark_palette = apply_theme
