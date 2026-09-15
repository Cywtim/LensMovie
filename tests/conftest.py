"""Pytest config for the test suite.

This module is imported once during test collection. It guarantees the project
root is on ``sys.path`` so ``import app`` works no matter where pytest is invoked
from or how the IDE is configured (a belt-and-braces companion to the
``pythonpath = ["."]`` setting in ``pyproject.toml``).
"""

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# UI tests default to the offscreen Qt platform so they run without a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
