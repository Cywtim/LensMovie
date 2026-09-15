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
    QCheckBox,
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
    """A labelled slider with a live numeric readout and a "fix" (lock) button.

    Fixing a parameter locks its value: the slider is disabled and programmatic
    :meth:`set_value` calls are ignored. A fixed parameter can only be released
    by the user (see :meth:`set_fixed`).
    """

    changed = pyqtSignal()
    fixedChanged = pyqtSignal(bool)

    def __init__(self, label, vmin, vmax, value, decimals=2, parent=None):
        super().__init__(parent)
        self.decimals = decimals
        self.vmin, self.vmax = vmin, vmax
        self._fixed = False
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        self._label = QLabel(label)
        self._label.setMinimumWidth(64)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 1000)
        self._frozen_int = 0  # last accepted position (used to revert when fixed)
        self._value = QLabel()
        self._value.setMinimumWidth(52)

        # "Fix" toggle: locked parameters cannot be changed by anything except
        # the user unlocking them again.
        self._lock_btn = QPushButton("🔓")
        self._lock_btn.setCheckable(True)
        self._lock_btn.setFixedWidth(30)
        self._lock_btn.setToolTip("Fix this parameter (locked until you unfix it)")
        self._lock_btn.toggled.connect(self._on_lock_toggled)

        lay.addWidget(self._label)
        lay.addWidget(self._slider, 1)
        lay.addWidget(self._value)
        lay.addWidget(self._lock_btn)
        self._slider.valueChanged.connect(self._on_change)
        self.set_value(value)

    def _to_int(self, v):
        t = max(0.0, min(1.0, (v - self.vmin) / (self.vmax - self.vmin)))
        return int(round(t * 1000))

    def _from_int(self, n):
        return self.vmin + n / 1000.0 * (self.vmax - self.vmin)

    def _on_change(self, n):
        if self._fixed:
            # A fixed parameter must not change through *any* path, including a
            # programmatic setValue on the underlying widget: snap it back.
            self._slider.blockSignals(True)
            self._slider.setValue(self._frozen_int)
            self._slider.blockSignals(False)
            return
        self._frozen_int = n
        self._value.setText(f"{self._from_int(n):.{self.decimals}f}")
        self.changed.emit()

    def _on_lock_toggled(self, checked):
        # This handler is the *only* place that unfixes a parameter, and it runs
        # only in response to a user click on the lock button.
        self.set_fixed(bool(checked), user=True)

    # ------------------------------------------------------------------ fixing
    def is_fixed(self) -> bool:
        return self._fixed

    def set_fixed(self, fixed: bool, *, user: bool = False):
        """Fix/unfix the parameter.

        Fixing (``fixed=True``) may be done programmatically. Unfixing a fixed
        parameter is reserved for the user, so it requires ``user=True``;
        otherwise a PermissionError is raised.
        """
        fixed = bool(fixed)
        if not fixed and self._fixed and not user:
            raise PermissionError(
                "a fixed parameter can only be unfixed by the user"
            )
        if fixed == self._fixed:
            return
        self._fixed = fixed
        # Keep the button state in sync without re-entering the handler.
        self._lock_btn.blockSignals(True)
        self._lock_btn.setChecked(fixed)
        self._lock_btn.blockSignals(False)
        self._lock_btn.setText("🔒" if fixed else "🔓")
        self._lock_btn.setToolTip(
            "Fixed — click to unfix" if fixed
            else "Fix this parameter (locked until you unfix it)"
        )
        self._slider.setEnabled(not fixed)
        self._label.setEnabled(not fixed)
        self._value.setEnabled(not fixed)
        if fixed:
            # Remember the frozen position so any later change attempt reverts.
            self._frozen_int = self._slider.value()
        self.fixedChanged.emit(fixed)

    def set_value(self, v):
        # A fixed parameter keeps its value; nothing may change it silently.
        if self._fixed:
            return
        self._slider.blockSignals(True)
        self._slider.setValue(self._to_int(v))
        self._slider.blockSignals(False)
        self._frozen_int = self._slider.value()
        self._value.setText(f"{v:.{self.decimals}f}")

    def value(self):
        return self._from_int(self._slider.value())


class _EntryCard(QGroupBox):
    """One lens or source entry; emits ``changed`` on any widget edit and
    ``remove`` when its delete button is pressed."""

    changed = pyqtSignal()
    remove_requested = pyqtSignal(object)

    def __init__(self, title, models=None, parent=None):
        super().__init__(parent)
        self.setTitle(title)

        outer = QVBoxLayout(self)
        head = QHBoxLayout()
        self._model_combo = QComboBox()
        if models:
            self._model_combo.addItems(list(models))
            self._model_combo.currentTextChanged.connect(self._on_edit)
        self._model_combo.setToolTip("profile type")
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

    # ------------------------------------------------------------- fixing API
    def fixed_params(self) -> set:
        """Names of this entry's parameters that are currently fixed."""
        return {name for name, s in self.sliders.items() if s.is_fixed()}

    def fix_param(self, name: str, fixed: bool = True, *, user: bool = False):
        """Fix (or, with ``user=True``, unfix) one of this entry's parameters."""
        if name in self.sliders:
            self.sliders[name].set_fixed(fixed, user=user)

    def unfix_all(self, *, user: bool = False):
        """Unfix every parameter of this entry (user action required)."""
        for s in self.sliders.values():
            if s.is_fixed():
                s.set_fixed(False, user=user)

    def _on_edit(self, *a):
        self.changed.emit()

    def redshift(self):
        return self._z_spin.value()

    def model(self):
        return self._model_combo.currentText()


class _LensCard(_EntryCard):
    def __init__(self, lens: lc.LensParams | None = None, parent=None):
        lens = lens or lc.LensParams()
        super().__init__("Lens", models=["SIS", "SIE", "PEMD"], parent=parent)
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
        super().__init__("Source", models=lc.SOURCE_MODELS, parent=parent)
        self._model_combo.setCurrentText(
            source.model if source.model in lc.SOURCE_MODELS else "SERSIC_ELLIPSE")
        self._z_spin.setValue(source.redshift)
        self.add_slider("amp", "amp", 0.05, 5.0, source.amp, 2)
        self.add_slider("R_sersic", "R_sersic", 0.01, 1.0, source.R_sersic, 3)
        self.add_slider("sigma", "sigma", 0.01, 1.0, source.sigma, 3)
        self.add_slider("n_sersic", "n_sersic", 0.5, 8.0, source.n_sersic, 2)
        self.add_slider("e1", "e1", -0.8, 0.8, source.e1, 2)
        self.add_slider("e2", "e2", -0.8, 0.8, source.e2, 2)
        self.add_slider("center_x", "x", -2.0, 2.0, source.center_x, 2)
        self.add_slider("center_y", "y", -2.0, 2.0, source.center_y, 2)

    def to_params(self) -> lc.SourceParams:
        s = self.sliders
        return lc.SourceParams(
            model=self.model(),
            amp=s["amp"].value(),
            R_sersic=s["R_sersic"].value(),
            n_sersic=s["n_sersic"].value(),
            sigma=s["sigma"].value(),
            e1=s["e1"].value(),
            e2=s["e2"].value(),
            center_x=s["center_x"].value(),
            center_y=s["center_y"].value(),
            redshift=self.redshift(),
        )


class _CardListPanel(QScrollArea):
    """Base scrollable panel managing one list of entry cards (lenses or sources)."""

    changed = pyqtSignal()

    def __init__(self, title: str, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._cards: list[_EntryCard] = []

        body = QWidget()
        self._root = QVBoxLayout(body)
        self._root.setContentsMargins(6, 6, 6, 6)
        self.setWidget(body)

        self._title = QLabel(f"<b>{title}</b>")
        self._root.addWidget(self._title)
        self._root.addStretch(1)

    def _insert_card(self, card):
        # Insert just before the trailing stretch (always last item).
        stretch = self._root.itemAt(self._root.count() - 1)
        idx = self._root.count() - 1 if stretch.spacerItem() is not None else self._root.count()
        self._root.insertWidget(idx, card)

    def _add_card(self, card):
        self._cards.append(card)
        self._insert_card(card)
        card.changed.connect(self._emit)
        card.remove_requested.connect(self._remove_card)

    def _remove_card(self, card):
        if len(self._cards) <= 1:
            return  # keep at least one entry
        self._cards.remove(card)
        card.setParent(None)
        card.deleteLater()
        self._emit()

    def _emit(self, *a):
        self.changed.emit()

    def fixed_params(self) -> list:
        """Per-entry set of fixed parameter names, in card order."""
        return [c.fixed_params() for c in self._cards]

    def unfix_all(self, *, user: bool = False):
        """Unfix every parameter of every entry (user action required)."""
        for c in self._cards:
            c.unfix_all(user=user)


class LensesPanel(_CardListPanel):
    """Right hand column (top): the list of lens planes."""

    def __init__(self, parent=None):
        super().__init__("Lenses", parent=parent)
        btn = QPushButton("+ Add lens")
        btn.clicked.connect(self._add_lens)
        self._root.insertWidget(self._root.count() - 1, btn)
        self._add_lens()

    def _add_lens(self):
        self._add_card(_LensCard())

    def lens_list(self):
        return [c.to_params() for c in self._cards]


class SourcesPanel(_CardListPanel):
    """Right hand column (bottom): the list of sources."""

    def __init__(self, parent=None):
        super().__init__("Sources", parent=parent)
        btn = QPushButton("+ Add source")
        btn.clicked.connect(self._add_source)
        self._root.insertWidget(self._root.count() - 1, btn)
        self._add_source()

    def _add_source(self):
        self._add_card(_SourceCard())

    def source_list(self):
        return [c.to_params() for c in self._cards]


class DisplayBar(QWidget):
    """A compact horizontal strip of display settings (grid, colormap, stretch)."""

    changed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)

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

        lay.addWidget(QLabel("numPix:"))
        lay.addWidget(self._numpix)
        lay.addSpacing(12)
        lay.addWidget(QLabel("colormap:"))
        lay.addWidget(self._cmap)
        lay.addSpacing(12)
        lay.addWidget(QLabel("stretch:"))
        lay.addWidget(self._stretch)
        lay.addSpacing(18)
        # 3D toggle: turning it off hides the 3D scene and skips rebuilding it,
        # which saves the per-update mesh construction / GL upload cost.
        self._three_d = QCheckBox("3D scene")
        self._three_d.setChecked(True)
        self._three_d.setToolTip("Show the 3D scene (uncheck to save resources)")
        self._three_d.toggled.connect(self._emit)
        lay.addWidget(self._three_d)
        lay.addStretch(1)

    def _emit(self, *a):
        self.changed.emit()

    def three_d_enabled(self) -> bool:
        return self._three_d.isChecked()

    def display(self) -> dict:
        return {
            "num_pix": self._numpix.value(),
            "colormap": self._cmap.currentText(),
            "stretch": self._stretch.currentText(),
        }

