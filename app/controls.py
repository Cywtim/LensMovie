"""Configurable control panel: multiple lenses + multiple sources.

The panel manages a list of :class:`lensing_calc.LensParams` and
:class:`lensing_calc.SourceParams` entries. Each entry shows its own widgets
(model type, redshift, sliders) with a "remove" button; buttons add new entries.
The panel exposes :meth:`as_config`, which emits a ``configChanged`` signal and
returns a :class:`lensing_calc.Config` snapshot consumed by the physics core.
"""

from __future__ import annotations

from PyQt5.QtCore import pyqtSignal, Qt
from PyQt5.QtWidgets import (
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import lensing_calc as lc


class _Slider(QWidget):
    """A labelled slider with a live numeric readout."""

    changed = pyqtSignal()

    def __init__(self, label, vmin, vmax, value, decimals=2, parent=None):
        super().__init__(parent)
        self.decimals = decimals
        self.vmin, self.vmax = vmin, vmax
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel(label)
        self._label.setMinimumWidth(64)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 1000)
        self._value = QLabel()
        self._value.setMinimumWidth(52)
        lay.addWidget(self._label)
        lay.addWidget(self._slider, 1)
        lay.addWidget(self._value)
        self._slider.valueChanged.connect(self._on_change)
        self.set_value(value)

    def _to_int(self, v):
        t = max(0.0, min(1.0, (v - self.vmin) / (self.vmax - self.vmin)))
        return int(round(t * 1000))

    def _from_int(self, n):
        return self.vmin + n / 1000.0 * (self.vmax - self.vmin)

    def _on_change(self, n):
        self._value.setText(f"{self._from_int(n):.{self.decimals}f}")
        self.changed.emit()

    def set_value(self, v):
        self._slider.blockSignals(True)
        self._slider.setValue(self._to_int(v))
        self._value.setText(f"{v:.{self.decimals}f}")
        self._slider.blockSignals(False)

    def value(self):
        return self._from_int(self._slider.value())


class _EntryCard(QGroupBox):
    """One lens or source entry; emits ``changed`` on any widget edit and
    ``remove`` when its delete button is pressed."""

    changed = pyqtSignal()
    remove_requested = pyqtSignal(object)

    def __init__(self, title, editable_model=True, parent=None):
        super().__init__(parent)
        self.setTitle(title)
        self._editable_model = editable_model

        outer = QVBoxLayout(self)
        head = QHBoxLayout()
        self._model_combo = QComboBox()
        if editable_model:
            self._model_combo.addItems(["SIS", "SIE", "PEMD"])
            self._model_combo.currentTextChanged.connect(self._on_edit)
        self._z_spin = QDoubleSpinBox()
        self._z_spin.setRange(0.05, 5.0)
        self._z_spin.setDecimals(2)
        self._z_spin.setSingleStep(0.05)
        self._z_spin.valueChanged.connect(self._on_edit)
        rm_btn = QPushButton("✕")
        rm_btn.setMaximumWidth(28)
        rm_btn.clicked.connect(lambda: self.remove_requested.emit(self))
        head.addWidget(QLabel("model:"))
        head.addWidget(self._model_combo, 1)
        head.addWidget(QLabel("z:"))
        head.addWidget(self._z_spin)
        head.addWidget(rm_btn)
        outer.addLayout(head)

        self.form = QFormLayout()
        outer.addLayout(self.form)

        # Sliders are stored in self.sliders: name -> _Slider
        self.sliders: dict[str, _Slider] = {}

    def add_slider(self, name, label, vmin, vmax, value, decimals=2):
        s = _Slider(label, vmin, vmax, value, decimals)
        s.changed.connect(self._on_edit)
        self.sliders[name] = s
        self.form.addRow(f"{label}:", s)

    def _on_edit(self, *a):
        self.changed.emit()

    def redshift(self):
        return self._z_spin.value()

    def model(self):
        return self._model_combo.currentText() if self._editable_model else "SERSIC_ELLIPSE"


class _LensCard(_EntryCard):
    def __init__(self, lens: lc.LensParams | None = None, parent=None):
        lens = lens or lc.LensParams()
        super().__init__("Lens", editable_model=True, parent=parent)
        self._lens = lens
        self._model_combo.setCurrentText(lens.model if lens.model in ("SIS", "SIE", "PEMD") else "SIS")
        self._z_spin.setValue(lens.redshift)
        self.add_slider("theta_E", "theta_E", 0.2, 3.0, lens.theta_E, 2)
        self.add_slider("gamma1", "g1", -0.3, 0.3, lens.gamma1, 3)
        self.add_slider("gamma2", "g2", -0.3, 0.3, lens.gamma2, 3)
        self.add_slider("e1", "e1", -0.8, 0.8, lens.e1, 2)
        self.add_slider("e2", "e2", -0.8, 0.8, lens.e2, 2)
        self.add_slider("gamma", "γ", 1.0, 3.0, lens.gamma, 2)
        self.add_slider("center_x", "x", -2.0, 2.0, lens.center_x, 2)
        self.add_slider("center_y", "y", -2.0, 2.0, lens.center_y, 2)

    def to_params(self) -> lc.LensParams:
        s = self.sliders
        return lc.LensParams(
            model=self.model(),
            theta_E=s["theta_E"].value(),
            gamma1=s["gamma1"].value(),
            gamma2=s["gamma2"].value(),
            center_x=s["center_x"].value(),
            center_y=s["center_y"].value(),
            e1=s["e1"].value(),
            e2=s["e2"].value(),
            gamma=s["gamma"].value(),
            redshift=self.redshift(),
        )


class _SourceCard(_EntryCard):
    def __init__(self, source: lc.SourceParams | None = None, parent=None):
        source = source or lc.SourceParams()
        super().__init__("Source", editable_model=False, parent=parent)
        self._z_spin.setValue(source.redshift)
        self.add_slider("amp", "amp", 0.05, 5.0, source.amp, 2)
        self.add_slider("R_sersic", "R_s", 0.01, 1.0, source.R_sersic, 3)
        self.add_slider("e1", "e1", -0.8, 0.8, source.e1, 2)
        self.add_slider("e2", "e2", -0.8, 0.8, source.e2, 2)
        self.add_slider("center_x", "x", -2.0, 2.0, source.center_x, 2)
        self.add_slider("center_y", "y", -2.0, 2.0, source.center_y, 2)

    def to_params(self) -> lc.SourceParams:
        s = self.sliders
        return lc.SourceParams(
            amp=s["amp"].value(),
            R_sersic=s["R_sersic"].value(),
            n_sersic=4.0,
            e1=s["e1"].value(),
            e2=s["e2"].value(),
            center_x=s["center_x"].value(),
            center_y=s["center_y"].value(),
            redshift=self.redshift(),
        )


class ConfigPanel(QScrollArea):
    """Scrollable panel managing all lens + source entries and display settings."""

    configChanged = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._lenses: list[_LensCard] = []
        self._sources: list[_SourceCard] = []

        body = QWidget()
        self._root = QVBoxLayout(body)
        self._root.setContentsMargins(6, 6, 6, 6)
        self.setWidget(body)

        # Display settings group
        disp = QGroupBox("Display")
        dform = QFormLayout(disp)
        self._numpix = QSpinBox()
        self._numpix.setRange(60, 400)
        self._numpix.setValue(150)
        self._numpix.valueChanged.connect(self._emit)
        self._cmap = QComboBox()
        self._cmap.addItems(["magma", "viridis", "plasma", "inferno", "gray", "turbo"])
        self._cmap.currentTextChanged.connect(self._emit)
        self._stretch = QComboBox()
        self._stretch.addItems(["log", "linear"])
        self._stretch.currentTextChanged.connect(self._emit)
        dform.addRow("numPix:", self._numpix)
        dform.addRow("colormap:", self._cmap)
        dform.addRow("stretch:", self._stretch)
        self._root.addWidget(disp)

        self._lens_title = QLabel("<b>Lenses</b>")
        self._root.addWidget(self._lens_title)
        self._add_lens_btn = QPushButton("+ Add lens")
        self._add_lens_btn.clicked.connect(lambda: self._add_lens())
        self._root.addWidget(self._add_lens_btn)

        self._source_title = QLabel("<b>Sources</b>")
        self._root.addWidget(self._source_title)
        self._add_src_btn = QPushButton("+ Add source")
        self._add_src_btn.clicked.connect(lambda: self._add_source())
        self._root.addWidget(self._add_src_btn)

        self._root.addStretch(1)

        # Start with one of each.
        self._add_lens()
        self._add_source()

    # ------------------------------------------------------------- management
    def _insert_lens(self, card):
        """Insert a lens card just before the Sources section."""
        idx = self._root.indexOf(self._source_title)
        self._root.insertWidget(idx, card)

    def _insert_source(self, card):
        """Insert a source card just before the stretch (end of the list)."""
        stretch = self._root.itemAt(self._root.count() - 1)
        idx = self._root.count() - 1 if stretch.spacerItem() is not None else self._root.count()
        self._root.insertWidget(idx, card)

    def _add_lens(self):
        card = _LensCard()
        self._lenses.append(card)
        self._insert_lens(card)
        card.changed.connect(self._emit)
        card.remove_requested.connect(self._remove_entry)

    def _add_source(self):
        card = _SourceCard()
        self._sources.append(card)
        self._insert_source(card)
        card.changed.connect(self._emit)
        card.remove_requested.connect(self._remove_entry)

    def _remove_entry(self, card):
        if isinstance(card, _LensCard):
            if len(self._lenses) <= 1:
                return  # keep at least one lens
            self._lenses.remove(card)
        else:
            if len(self._sources) <= 1:
                return
            self._sources.remove(card)
        card.setParent(None)
        card.deleteLater()
        self._emit()

    # ------------------------------------------------------------- data model
    def _emit(self, *a):
        self.configChanged.emit()

    def display(self) -> dict:
        return {
            "num_pix": self._numpix.value(),
            "colormap": self._cmap.currentText(),
            "stretch": self._stretch.currentText(),
        }

    def as_config(self) -> lc.Config:
        lenses = [c.to_params() for c in self._lenses]
        sources = [c.to_params() for c in self._sources]
        d = self.display()
        return lc.Config(
            lenses=lenses,
            sources=sources,
            num_pix=d["num_pix"],
            delta_pix=0.05,
        )
