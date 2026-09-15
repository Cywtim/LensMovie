"""Parameter control panel built from PyQt5 widgets.

Sliders update in real time; the panel emits a single ``parametersChanged`` signal
with the current parameter dict when any control changes.
"""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)


class _SliderRow(QWidget):
    """A labelled slider with a numeric value readout."""

    changed = pyqtSignal(str, float)

    def __init__(self, name: str, label: str, vmin: float, vmax: float, value: float, decimals: int = 2, parent=None):
        super().__init__(parent)
        self.name = name
        self.decimals = decimals
        self.vmin = vmin
        self.vmax = vmax

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel(label)
        self._label.setMinimumWidth(90)

        self._slider = QSlider(Qt.Horizontal)
        self._slider.setMinimum(0)
        self._slider.setMaximum(1000)
        self._slider.setValue(self._to_int(value))

        self._value_label = QLabel(f"{value:.{decimals}f}")
        self._value_label.setMinimumWidth(50)

        layout.addWidget(self._label)
        layout.addWidget(self._slider, 1)
        layout.addWidget(self._value_label)

        self._slider.valueChanged.connect(self._on_slider)

    def _to_int(self, value: float) -> int:
        t = (value - self.vmin) / (self.vmax - self.vmin)
        t = max(0.0, min(1.0, t))
        return int(round(t * 1000))

    def _from_int(self, v: int) -> float:
        t = v / 1000.0
        return self.vmin + t * (self.vmax - self.vmin)

    def _on_slider(self, v: int):
        value = self._from_int(v)
        self._value_label.setText(f"{value:.{self.decimals}f}")
        self.changed.emit(self.name, value)

    def set_value(self, value: float):
        self._slider.blockSignals(True)
        self._slider.setValue(self._to_int(value))
        self._value_label.setText(f"{value:.{self.decimals}f}")
        self._slider.blockSignals(False)


class ParameterPanel(QWidget):
    """Left-hand panel: lens params, source params, display settings."""

    parametersChanged = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.lens = {
            "theta_E": 1.0, "gamma1": 0.05, "gamma2": 0.0,
            "center_x": 0.0, "center_y": 0.0,
        }
        self.source = {
            "amp": 1.0, "R_sersic": 0.1, "e1": 0.1, "e2": -0.2,
            "center_x": 0.1, "center_y": -0.1,
        }
        self.display = {"num_pix": 150, "colormap": "viridis", "stretch": "log"}

        root = QVBoxLayout(self)

        # --- Lens group ---
        lens_box = QGroupBox("Lens (SIS + shear)")
        lens_form = QFormLayout(lens_box)
        lens_form.setContentsMargins(8, 12, 8, 8)
        self._sliders = {}
        lens_specs = [
            ("theta_E", "theta_E", 0.2, 3.0, self.lens["theta_E"], 2),
            ("gamma1", "gamma1", -0.3, 0.3, self.lens["gamma1"], 3),
            ("gamma2", "gamma2", -0.3, 0.3, self.lens["gamma2"], 3),
            ("center_x", "lens x", -2.0, 2.0, self.lens["center_x"], 2),
            ("center_y", "lens y", -2.0, 2.0, self.lens["center_y"], 2),
        ]
        for name, label, lo, hi, val, dec in lens_specs:
            row = _SliderRow(name, label, lo, hi, val, dec)
            self._sliders[name] = row
            lens_form.addRow(f"{label}:", row)
        root.addWidget(lens_box)

        # --- Source group ---
        src_box = QGroupBox("Source (Sersic)")
        src_form = QFormLayout(src_box)
        src_form.setContentsMargins(8, 12, 8, 8)
        src_specs = [
            ("amp", "amp", 0.1, 5.0, self.source["amp"], 2),
            ("R_sersic", "R_sersic", 0.02, 1.0, self.source["R_sersic"], 3),
            ("e1", "e1", -0.8, 0.8, self.source["e1"], 2),
            ("e2", "e2", -0.8, 0.8, self.source["e2"], 2),
            ("center_x", "src x", -2.0, 2.0, self.source["center_x"], 2),
            ("center_y", "src y", -2.0, 2.0, self.source["center_y"], 2),
        ]
        for name, label, lo, hi, val, dec in src_specs:
            row = _SliderRow(name, label, lo, hi, val, dec)
            self._sliders[name] = row
            src_form.addRow(f"{label}:", row)
        root.addWidget(src_box)

        # --- Display group ---
        disp_box = QGroupBox("Display")
        disp_form = QFormLayout(disp_box)
        disp_form.setContentsMargins(8, 12, 8, 8)

        self._num_pix_spin = QSpinBox()
        # Configure range/value BEFORE connecting signals, so the constructor's
        # clamping/value setting cannot mutate our parameter state.
        self._num_pix_spin.setRange(60, 400)
        self._num_pix_spin.setValue(self.display["num_pix"])
        disp_form.addRow("numPix:", self._num_pix_spin)

        self._cmap_combo = QComboBox()
        self._cmap_combo.addItems(["viridis", "plasma", "inferno", "magma", "gray", "turbo"])
        disp_form.addRow("colormap:", self._cmap_combo)

        self._stretch_combo = QComboBox()
        self._stretch_combo.addItems(["log", "linear"])
        disp_form.addRow("stretch:", self._stretch_combo)
        root.addWidget(disp_box)

        root.addStretch(1)

        # Wire signals
        for name, row in self._sliders.items():
            row.changed.connect(self._on_param_changed)
        self._num_pix_spin.valueChanged.connect(self._on_num_pix)
        self._cmap_combo.currentTextChanged.connect(self._on_cmap)
        self._stretch_combo.currentTextChanged.connect(self._on_stretch)

    # --- param bookkeeping: track which group a slider belongs to ---
    def _slider_owner(self, name: str) -> dict:
        if name in self.lens:
            return self.lens
        if name in self.source:
            return self.source
        return None

    def _on_param_changed(self, name: str, value: float):
        owner = self._slider_owner(name)
        if owner is not None:
            owner[name] = value
            self._emit()

    def _on_num_pix(self, v: int):
        self.display["num_pix"] = int(v)
        self._emit()

    def _on_cmap(self, v: str):
        self.display["colormap"] = v
        self._emit()

    def _on_stretch(self, v: str):
        self.display["stretch"] = v
        self._emit()

    def _emit(self):
        self.parametersChanged.emit(self.as_dict())

    def as_dict(self) -> dict:
        """Return a combined snapshot of current parameters."""
        return {
            "lens": dict(self.lens),
            "source": dict(self.source),
            "display": dict(self.display),
        }
