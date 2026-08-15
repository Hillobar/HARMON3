"""The theme applies cleanly and the widgets that carry it keep their roles."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

QtWidgets = pytest.importorskip("PySide6.QtWidgets")


@pytest.fixture(scope="module")
def qapp():
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    yield app


def test_theme_applies_without_error(qapp):
    from harmon3.ui import style
    style.apply_theme(qapp)
    assert qapp.styleSheet()
    # Setting a stylesheet wraps the base style, so the proxy is what shows up here.
    assert qapp.style().metaObject().className() == "QStyleSheetStyle"


def test_stylesheet_has_no_unresolved_placeholders(qapp):
    """An f-string typo would leave a literal brace behind and silently drop a rule."""
    from harmon3.ui import style
    sheet = style.STYLESHEET
    assert "{" not in sheet.replace("{{", "").replace("}}", "").split("QWidget")[0]
    for token in ("None", "{ACCENT}", "{TEXT}"):
        assert token not in sheet


def test_no_font_family_in_the_stylesheet(qapp):
    """Typography goes through QFont.setFamilies; Qt's stylesheet parser mishandles lists."""
    assert "font-family" not in style_sheet()


def style_sheet() -> str:
    from harmon3.ui import style
    return style.STYLESHEET


def test_palette_roles_are_all_dark(qapp):
    from PySide6.QtGui import QPalette
    from harmon3.ui import style

    style.apply_theme(qapp)
    palette = qapp.palette()
    for role in (QPalette.Window, QPalette.Base, QPalette.AlternateBase):
        colour = palette.color(role)
        assert colour.lightness() < 60, role


def test_text_contrasts_with_the_background(qapp):
    from PySide6.QtGui import QPalette
    from harmon3.ui import style

    style.apply_theme(qapp)
    palette = qapp.palette()
    background = palette.color(QPalette.Window).lightness()
    text = palette.color(QPalette.WindowText).lightness()
    assert text - background > 100


def test_metric_label_is_monospaced_and_tagged(qapp):
    from harmon3.ui import style

    label = style.metric("864 x 480")
    assert label.property("role") == "metric"
    assert label.font().families()[0] == style.MONO_FAMILIES[0]


def test_hint_label_is_tagged(qapp):
    from harmon3.ui import style
    assert style.hint("x").property("role") == "hint"


def _sized(label, width, qapp):
    """Give the label a real size.

    Qt only delivers a resize event once a widget has been shown, so the label has to be
    laid out for real -- WA_DontShowOnScreen does that without putting anything on screen.
    """
    from PySide6.QtCore import Qt

    label.setAttribute(Qt.WA_DontShowOnScreen, True)
    label.resize(width, 20)
    label.show()
    qapp.processEvents()
    return label


def test_elided_label_shortens_without_losing_the_full_text(qapp):
    from harmon3.ui import style

    full = "a_very_long_reference_filename_that_will_not_fit.png"
    label = _sized(style.ElidedLabel(full), 80, qapp)

    assert label.fullText() == full
    assert label.toolTip() == full          # the whole name stays reachable
    # QLabel.text() is what gets painted; it must have been shortened to fit.
    assert len(label.text()) < len(full)
    assert "…" in label.text()


def test_elided_label_leaves_short_text_alone(qapp):
    from harmon3.ui import style

    label = _sized(style.ElidedLabel("a.png"), 300, qapp)
    assert label.text() == "a.png"


def test_elided_label_re_elides_when_the_text_changes(qapp):
    from harmon3.ui import style

    label = _sized(style.ElidedLabel("a.png"), 80, qapp)
    label.setText("another_extremely_long_reference_filename_here.png")
    assert label.toolTip() == "another_extremely_long_reference_filename_here.png"
    assert len(label.text()) < len(label.fullText())


def test_elided_label_never_forces_a_minimum_width(qapp):
    """This is the point of it: one long filename must not widen the whole column."""
    from harmon3.ui import style

    label = style.ElidedLabel("x" * 400)
    assert label.minimumSizeHint().width() <= 40


def test_stylise_uppercases_group_headings(qapp):
    from harmon3.ui import style

    container = QtWidgets.QWidget()
    group = QtWidgets.QGroupBox("Resolution", container)
    style.stylise(container)
    assert group.title() == "RESOLUTION"
    container.deleteLater()


def test_stylise_is_idempotent(qapp):
    from harmon3.ui import style

    container = QtWidgets.QWidget()
    group = QtWidgets.QGroupBox("Duration", container)
    style.stylise(container)
    style.stylise(container)
    assert group.title() == "DURATION"
    container.deleteLater()


def test_tag_chips_are_monospaced(qapp):
    from harmon3.ui import style
    from harmon3.ui.ref_panel import TagChip

    chip = TagChip()
    assert chip.property("role") == "tag"
    assert chip.font().families()[0] == style.MONO_FAMILIES[0]
    chip.deleteLater()


def test_reference_section_headings_are_uppercase(qapp):
    from harmon3.refs import IMAGE
    from harmon3.ui.ref_panel import RefListWidget

    widget = RefListWidget(IMAGE)
    assert widget.title_label.text().startswith("IMAGES")
    assert widget.title_label.property("role") == "section"
    widget.deleteLater()


def test_the_legacy_theme_entry_point_still_works(qapp):
    from harmon3.ui import style
    assert style.apply_dark_palette is style.apply_theme
