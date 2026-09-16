"""Application entry point.

Run with::

    conda run -n lenstronomy_env python -m app.main
"""

from __future__ import annotations

import sys


def entry() -> int:
    from PyQt5.QtWidgets import QApplication

    from . import theme
    from .main_window import MainWindow

    app = QApplication(sys.argv)
    # One call styles every widget (theme.qss) *and* matches matplotlib to the
    # dark look.  Restyling later is a one-file change (app/theme.qss / palette).
    theme.apply_theme(app)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(entry())
