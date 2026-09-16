"""Render the themed LensMovie window to a PNG for visual checks.

Run from the project root (a display is needed, e.g. DISPLAY=:1):

    conda run -n lenstronomy_env python tools/render_screenshot.py img/lensmovie_dark.png

The screenshot shows the current skin (app/theme.qss) applied to the real
main window.  It is a visual-verification aid, not part of the app.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Make ``app`` importable no matter where this script is invoked from.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main(out_path: str) -> int:
    from PyQt5.QtWidgets import QApplication

    from app import theme
    from app.main_window import MainWindow

    app = QApplication(sys.argv)
    theme.apply_theme(app)
    win = MainWindow()
    win.resize(1500, 1050)
    win.show()
    for _ in range(40):
        app.processEvents()

    pix = win.grab()
    ok = pix.save(out_path)
    win.close()
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "img/lensmovie_dark.png"))
