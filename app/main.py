"""Application entry point.

Run with::

    conda run -n lenstronomy_env python -m app.main
"""

from __future__ import annotations

import sys


def entry() -> int:
    from PyQt5.QtWidgets import QApplication

    from .main_window import MainWindow

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(entry())
