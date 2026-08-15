"""QApplication bootstrap."""

from __future__ import annotations

import logging
import sys
import uuid

from PySide6.QtWidgets import QApplication, QMessageBox

from .. import config
from . import style

log = logging.getLogger(__name__)


def run_gui(server_url: str) -> int:
    app = QApplication(sys.argv[:1])
    app.setApplicationName("HARMON3")
    app.setOrganizationName("HARMON3")
    style.apply_theme(app)

    try:
        config.load_workflow()
    except (FileNotFoundError, ValueError) as exc:
        QMessageBox.critical(None, "HARMON3 cannot start", str(exc))
        return 2

    config.RUNS_DIR.mkdir(parents=True, exist_ok=True)
    config.VIDEO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    from .main_window import MainWindow

    window = MainWindow(server_url, uuid.uuid4().hex)
    window.show()
    return app.exec()
