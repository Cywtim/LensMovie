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
    """A labelled slider **plus an editable number box**, and a "fix" (lock) button.

    Dragging a 1000-step slider cannot hit an exact value, so the value is also
    editable: either drag, or type the precise number. The spin box is the
    authoritative value (:meth:`value` reads it), so a typed value keeps its full
    precision even though the slider can only snap to its nearest step.

    Fixing a parameter locks its value: both the slider and the box are disabled
    and programmatic :meth:`set_value` calls are ignored. A fixed parameter can
    only be released by the user (see :meth:`set_fixed`).
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
        lay.setSpacing(4)
        self._label = QLabel(label)
        self._label.setMinimumWidth(60)
        self._slider = QSlider(Qt.Horizontal)
        self._slider.setRange(0, 1000)
        self._frozen_int = 0      # last accepted slider position (revert when fixed)
        self._frozen_value = 0.0  # last accepted exact value (revert when fixed)

        # Editable value. It is given one more decimal than the readout used to
        # show, i.e. finer than the slider's own step, so typing is worthwhile.
        self._value = QDoubleSpinBox()
        self._value.setRange(float(vmin), float(vmax))
        self._value.setDecimals(max(int(decimals), 3))
        self._value.setSingleStep(max((vmax - vmin) / 200.0, 10 ** -max(int(decimals), 3)))
        self._value.setKeyboardTracking(False)   # act on commit, not per keystroke
        self._value.setMinimumWidth(84)
        self._value.setToolTip("type an exact value, or drag the slider")

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
        self._slider.valueChanged.connect(self._on_slider)
        self._value.valueChanged.connect(self._on_spin)
        self.set_value(value)

    def _to_int(self, v):
        t = max(0.0, min(1.0, (v - self.vmin) / (self.vmax - self.vmin)))
        return int(round(t * 1000))

    def _from_int(self, n):
        return self.vmin + n / 1000.0 * (self.vmax - self.vmin)

    def _on_slider(self, n):
        """The user dragged the slider: mirror it into the exact-value box."""
        if self._fixed:
            # A fixed parameter must not change through *any* path, including a
            # programmatic setValue on the underlying widget: snap it back.
            self._slider.blockSignals(True)
            self._slider.setValue(self._frozen_int)
            self._slider.blockSignals(False)
            return
        v = self._from_int(n)
        self._frozen_int = n
        self._frozen_value = v
        self._value.blockSignals(True)
        self._value.setValue(v)
        self._value.blockSignals(False)
        self.changed.emit()

    def _on_spin(self, v):
        """The user typed a value: move the slider to its nearest step.

        The slider's signals are blocked so this cannot bounce the typed value
        back to the (coarser) slider position.
        """
        if self._fixed:
            self._value.blockSignals(True)
            self._value.setValue(self._frozen_value)
            self._value.blockSignals(False)
            return
        self._frozen_value = float(v)
        self._frozen_int = self._to_int(v)
        self._slider.blockSignals(True)
        self._slider.setValue(self._frozen_int)
        self._slider.blockSignals(False)
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
        self._value.setEnabled(not fixed)      # the number box is locked too
        if fixed:
            # Remember the frozen state so any later change attempt reverts.
            self._frozen_int = self._slider.value()
            self._frozen_value = self._value.value()
        self.fixedChanged.emit(fixed)

    def set_value(self, v):
        # A fixed parameter keeps its value; nothing may change it silently.
        if self._fixed:
            return
        self._frozen_int = self._to_int(v)
        self._frozen_value = float(v)
        self._slider.blockSignals(True)
        self._slider.setValue(self._frozen_int)
        self._slider.blockSignals(False)
        self._value.blockSignals(True)
        self._value.setValue(float(v))
        self._value.blockSignals(False)

    def value(self):
        """The exact value: the number box wins over the quantised slider."""
        return float(self._value.value())

    def slider_value(self):
        """The value implied by the slider's (quantised) position."""
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

    def param_spec(self) -> dict:
        """Return {name: (value, lower, upper, fixed)} for this entry.

        Used by the fitting layer to build free/fixed parameter lists and bounds
        without depending on Qt.
        """
        return {
            name: (s.value(), s.vmin, s.vmax, s.is_fixed())
            for name, s in self.sliders.items()
        }

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
        super().__init__("Lens", models=lc.available_lens_models(), parent=parent)
        self._lens = lens
        self._model_combo.setCurrentText(
            lens.model if lens.model in lc.available_lens_models() else "SIS")
        self._z_spin.setValue(lens.redshift)
        self.add_slider("theta_E", "theta_E", 0.2, 3.0, lens.theta_E, 2)
        self.add_slider("gamma1", "g1", -0.3, 0.3, lens.gamma1, 3)
        self.add_slider("gamma2", "g2", -0.3, 0.3, lens.gamma2, 3)
        self.add_slider("e1", "e1", -0.8, 0.8, lens.e1, 2)
        self.add_slider("e2", "e2", -0.8, 0.8, lens.e2, 2)
        self.add_slider("gamma", "γ", 1.0, 3.0, lens.gamma, 2)
        self.add_slider("Rs", "Rs", 0.1, 5.0, lens.Rs, 2)
        self.add_slider("alpha_Rs", "α_Rs", 0.05, 2.0, lens.alpha_Rs, 3)
        self.add_slider("r_trunc", "r_trunc", 0.5, 10.0, lens.r_trunc, 2)
        self.add_slider("center_x", "x", -2.0, 2.0, lens.center_x, 2)
        self.add_slider("center_y", "y", -2.0, 2.0, lens.center_y, 2)

        # --- deflector (lens galaxy) light ---------------------------------
        # Unlensed light from the lens galaxy itself; needed so its light is not
        # absorbed into the source when fitting real data.
        self.form.addRow(QLabel("<i>— lens light —</i>"))
        self._light_combo = QComboBox()
        self._light_combo.addItems(lc.LENS_LIGHT_MODELS)
        self._light_combo.setCurrentText(
            lens.light_model if lens.light_model in lc.LENS_LIGHT_MODELS else "NONE")
        self._light_combo.currentTextChanged.connect(self._on_edit)
        self._light_combo.setToolTip("light profile of the deflector galaxy")
        self.form.addRow("light:", self._light_combo)
        self.add_slider("light_amp", "L amp", 0.0, 5.0, lens.light_amp, 3)
        self.add_slider("light_R_sersic", "L R_s", 0.05, 3.0, lens.light_R_sersic, 2)
        self.add_slider("light_n_sersic", "L n", 0.5, 8.0, lens.light_n_sersic, 2)
        self.add_slider("light_sigma", "L σ", 0.05, 3.0, lens.light_sigma, 2)
        self.add_slider("light_e1", "L e1", -0.8, 0.8, lens.light_e1, 2)
        self.add_slider("light_e2", "L e2", -0.8, 0.8, lens.light_e2, 2)

        self._light_combo.currentTextChanged.connect(self._sync_light_enabled)
        self._sync_light_enabled()

    def _sync_light_enabled(self, *a):
        """Grey out the light sliders when the lens emits no light."""
        on = self._light_combo.currentText() != "NONE"
        for name in ("light_amp", "light_R_sersic", "light_n_sersic",
                     "light_sigma", "light_e1", "light_e2"):
            self.sliders[name].setEnabled(on)

    def light_model(self) -> str:
        return self._light_combo.currentText()

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
            Rs=s["Rs"].value(),
            alpha_Rs=s["alpha_Rs"].value(),
            r_trunc=s["r_trunc"].value(),
            redshift=self.redshift(),
            light_model=self.light_model(),
            light_amp=s["light_amp"].value(),
            light_R_sersic=s["light_R_sersic"].value(),
            light_n_sersic=s["light_n_sersic"].value(),
            light_sigma=s["light_sigma"].value(),
            light_e1=s["light_e1"].value(),
            light_e2=s["light_e2"].value(),
        )


class _SourceCard(_EntryCard):
    point_toggled = pyqtSignal(bool)

    def __init__(self, source: lc.SourceParams | None = None, parent=None):
        source = source or lc.SourceParams()
        super().__init__("Source", models=lc.SOURCE_MODELS, parent=parent)
        self._model_combo.setCurrentText(
            source.model if source.model in lc.SOURCE_MODELS else "SERSIC_ELLIPSE")
        self._z_spin.setValue(source.redshift)
        self._pt_check = QCheckBox("point source (AGN / lensed point)")
        self._pt_check.setToolTip(
            "Give this source a lensed point source at its centre (an AGN).\n"
            "It shows as a PSF spike at the multiple image positions.")
        self._pt_check.toggled.connect(self.point_toggled)
        self.form.addRow("point:", self._pt_check)
        self.add_slider("amp", "amp", 0.05, 5.0, source.amp, 2)
        self.add_slider("R_sersic", "R_sersic", 0.01, 1.0, source.R_sersic, 3)
        self.add_slider("sigma", "sigma", 0.01, 1.0, source.sigma, 3)
        self.add_slider("n_sersic", "n_sersic", 0.5, 8.0, source.n_sersic, 2)
        self.add_slider("e1", "e1", -0.8, 0.8, source.e1, 2)
        self.add_slider("e2", "e2", -0.8, 0.8, source.e2, 2)
        self.add_slider("Rs", "Rs", 0.02, 1.5, source.Rs, 3)
        self.add_slider("Rb", "Rb", 0.02, 1.5, source.Rb, 3)
        self.add_slider("gamma", "γ", 0.1, 4.0, source.gamma, 2)
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
            Rs=s["Rs"].value(),
            Rb=s["Rb"].value(),
            gamma=s["gamma"].value(),
            center_x=s["center_x"].value(),
            center_y=s["center_y"].value(),
            redshift=self.redshift(),
        )

    def set_point_checked(self, on: bool, *, quiet: bool = False):
        """Set the point-source checkbox (from the point-source panel), without
        re-triggering ``point_toggled`` when ``quiet`` (the other panel drove it)."""
        if quiet:
            self._pt_check.blockSignals(True)
        self._pt_check.setChecked(on)
        if quiet:
            self._pt_check.blockSignals(False)


class _PointSourceCard(_EntryCard):
    """One point source.

    ``LENSED``   — a source-plane point that is lensed by the same lens as the
                   extended sources (its images + magnification are solved);
                   amplitude is a *source-plane* flux (source_amp).
    ``UNLENSED`` — an image-plane star fixed at an on-sky position (point_amp).
    An entry can either carry its own position/redshift or *attach to a source*
    (e.g. an AGN on top of a host galaxy), sharing its centre and redshift.
    """

    ref_changed = pyqtSignal(int)      # this entry's (new) ref_source index

    def __init__(self, point: lc.PointSourceParams | None = None,
                 sources: list | None = None, parent=None):
        point = point or lc.PointSourceParams()
        super().__init__("Point source", models=["LENSED", "UNLENSED"],
                         parent=parent)
        self._model_combo.setCurrentText(
            point.model if point.model in ("LENSED", "UNLENSED") else "LENSED")
        self._z_spin.setValue(point.redshift)
        self._ref_source = point.ref_source

        # anchor: own position, or reuse a source's centre + redshift
        self._anchor_combo = QComboBox()
        self._anchor_combo.setToolTip("own position, or ride on a source's centre")
        self._anchor_combo.currentIndexChanged.connect(self._on_anchor)
        self.form.addRow("anchor:", self._anchor_combo)

        self.add_slider("source_amp", "Flux", 0.02, 5.0, point.source_amp, 3)
        self.add_slider("point_amp", "On-sky", 0.02, 5.0, point.point_amp, 3)
        self.add_slider("center_x", "x", -2.0, 2.0, point.center_x, 2)
        self.add_slider("center_y", "y", -2.0, 2.0, point.center_y, 2)

        self.set_sources(sources or [], point.ref_source)
        self._sync_model()
        self._model_combo.currentTextChanged.connect(self._sync_model)

    # ------------------------------------------------------------- anchoring
    def set_sources(self, sources: list, ref_source: int = -1):
        """Rebuild the anchor options from the current source list and re-apply
        ``ref_source`` (out-of-range values fall back to 'own')."""
        self._ref_source = ref_source if 0 <= ref_source < len(sources) else -1
        self._anchor_combo.blockSignals(True)
        self._anchor_combo.clear()
        self._anchor_combo.addItem("own position")
        for i in range(len(sources)):
            self._anchor_combo.addItem(f"attach to Source {i + 1}")
        self._anchor_combo.blockSignals(False)
        self._anchor_combo.setCurrentIndex(self._ref_source + 1)
        self._apply_anchor()

    def _on_anchor(self, index):
        old = self._ref_source
        self._ref_source = index - 1 if index >= 1 else -1
        self._apply_anchor()
        self.changed.emit()
        if old != self._ref_source:
            self.ref_changed.emit(self._ref_source)

    def _apply_anchor(self):
        attached = self._ref_source >= 0
        for name in ("center_x", "center_y"):
            self.sliders[name].setEnabled(not attached)
        self._z_spin.setEnabled(not attached)

    def _sync_model(self, *a):
        lensed = self._model_combo.currentText() == "LENSED"
        self.sliders["source_amp"].setEnabled(lensed)
        self.sliders["point_amp"].setEnabled(not lensed)

    def to_params(self) -> lc.PointSourceParams:
        s = self.sliders
        return lc.PointSourceParams(
            model=self.model(),
            source_amp=s["source_amp"].value(),
            point_amp=s["point_amp"].value(),
            center_x=s["center_x"].value(),
            center_y=s["center_y"].value(),
            redshift=self.redshift(),
            ref_source=self._ref_source,
        )


class _CardListPanel(QScrollArea):
    """Base scrollable panel managing one list of entry cards (lenses or sources)."""

    changed = pyqtSignal()

    def __init__(self, title: str, entry_label: str, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self._cards: list[_EntryCard] = []
        self._entry_label = entry_label   # "Lens" / "Source" / "Point source"

        body = QWidget()
        self._root = QVBoxLayout(body)
        self._root.setContentsMargins(6, 6, 6, 6)
        self.setWidget(body)

        self._title = QLabel(f"<b>{title}</b>")
        self._title.setObjectName("panelTitle")
        self._root.addWidget(self._title)
        self._root.addStretch(1)

    def _insert_card(self, card):
        # Insert just before the trailing stretch (always last item).
        stretch = self._root.itemAt(self._root.count() - 1)
        idx = self._root.count() - 1 if stretch.spacerItem() is not None else self._root.count()
        self._root.insertWidget(idx, card)

    def _relabel(self):
        """Renumber the card titles (Lens 1, Lens 2, …) after add/remove."""
        for i, card in enumerate(self._cards):
            card.setTitle(f"{self._entry_label} {i + 1}")

    def _add_card(self, card):
        self._cards.append(card)
        self._insert_card(card)
        card.changed.connect(self._emit)
        card.remove_requested.connect(self._remove_card)
        self._relabel()

    def _remove_card(self, card):
        if len(self._cards) <= 1:
            return  # keep at least one entry
        self._cards.remove(card)
        card.setParent(None)
        card.deleteLater()
        self._relabel()
        self._emit()

    def _emit(self, *a):
        self.changed.emit()

    def fixed_params(self) -> list:
        """Per-entry set of fixed parameter names, in card order."""
        return [c.fixed_params() for c in self._cards]

    def param_specs(self) -> list:
        """Per-entry {name: (value, lower, upper, fixed)}, in card order."""
        return [c.param_spec() for c in self._cards]

    def unfix_all(self, *, user: bool = False):
        """Unfix every parameter of every entry (user action required)."""
        for c in self._cards:
            c.unfix_all(user=user)


class LensesPanel(_CardListPanel):
    """Right hand column (top): the list of lens planes."""

    def __init__(self, parent=None):
        super().__init__("Lenses", "Lens", parent=parent)
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

    point_source_toggled = pyqtSignal(int, bool)   # (source index, now attached?)

    def __init__(self, parent=None):
        super().__init__("Sources", "Source", parent=parent)
        btn = QPushButton("+ Add source")
        btn.clicked.connect(self._add_source)
        self._root.insertWidget(self._root.count() - 1, btn)
        self._add_source()

    def _add_source(self):
        card = _SourceCard()
        self._add_card(card)
        card.point_toggled.connect(
            lambda on, c=card: self.point_source_toggled.emit(
                self._cards.index(c), on))

    def source_list(self):
        return [c.to_params() for c in self._cards]

    def set_point_checked(self, source_index: int, on: bool):
        """Mirror the point-source panel's attachment state onto the checkbox."""
        if 0 <= source_index < len(self._cards):
            self._cards[source_index].set_point_checked(on, quiet=True)


class PointSourcesPanel(_CardListPanel):
    """Point sources: lensed (source-plane) points and unlensed image-plane stars.

    Each entry can attach to a source (AGN on a host galaxy) or carry its own
    position.  ``attached_changed`` lets the Sources panel mirror which source
    has a point source.
    """

    attached_changed = pyqtSignal(int, bool)   # (source index, now attached?)

    def __init__(self, parent=None):
        super().__init__("Point sources", "Point source", parent=parent)
        btn = QPushButton("+ Add point source")
        btn.clicked.connect(self._add_point)
        self._root.insertWidget(self._root.count() - 1, btn)
        self._sources: list = []
        self._attached_flags: list[bool] = []
        # starts EMPTY: no point source is a valid (and default) config.

    def _add_point(self, point: lc.PointSourceParams | None = None,
                   ref_source: int = -1):
        card = _PointSourceCard(point=point, sources=self._sources)
        if ref_source >= 0:
            card.set_sources(self._sources, ref_source)
        card.ref_changed.connect(self._on_ref_changed)
        self._add_card(card)
        self._sync_attached_flags()
        return card

    def _on_ref_changed(self, ref):
        self.changed.emit()
        self._sync_attached_flags()

    def _remove_card(self, card):
        # A point source is optional — unlike lenses/sources, allow an empty list.
        if card in self._cards:
            self._cards.remove(card)
            card.setParent(None)
            card.deleteLater()
            self._relabel()
            self.changed.emit()
            self._sync_attached_flags()

    def point_source_list(self):
        return [c.to_params() for c in self._cards]

    def update_sources(self, sources: list):
        """Refresh the anchor options when the source list changes."""
        self._sources = list(sources)
        for card in self._cards:
            card.set_sources(self._sources, card._ref_source)
        self._attached_flags = [False] * len(self._sources)
        for c in self._cards:
            r = c._ref_source
            if 0 <= r < len(self._attached_flags):
                self._attached_flags[r] = True

    def set_attached(self, source_index: int, enabled: bool):
        """Attach a point source to (or detach it from) a source index.

        Driven by the Sources panel's "point" checkbox.
        """
        if not (0 <= source_index < len(self._sources)):
            return
        if enabled:
            if not any(c._ref_source == source_index for c in self._cards):
                self._add_point(ref_source=source_index)
        else:
            for c in list(self._cards):
                if c._ref_source == source_index:
                    c.set_sources(self._sources, -1)      # free, keep the card
            self._sync_attached_flags()

    def _sync_attached_flags(self):
        """Recompute per-source attachment and emit changes (also mirrors onto
        the Sources panel's checkboxes)."""
        n = len(self._sources)
        desired = [False] * n
        for c in self._cards:
            r = c._ref_source
            if 0 <= r < n:
                desired[r] = True
        for i in range(n):
            if desired[i] != self._attached_flags[i]:
                self._attached_flags[i] = desired[i]
                self.attached_changed.emit(i, desired[i])


class DataBar(QWidget):
    """The external-image file buttons, on their own (separated) section of the
    display row so they do not consume vertical space inside the image panel."""

    loadImageRequested = pyqtSignal()
    clearRequested = pyqtSignal()
    loadAuxRequested = pyqtSignal(str)      # "noise" | "mask" | "psf"

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(4)

        self._load_btn = QPushButton("Load…")
        self._load_btn.setToolTip("Load a lensed image / matrix\n(npy, fits, mat, text, image)")
        self._load_btn.clicked.connect(self.loadImageRequested)
        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setToolTip("Forget the loaded data")
        self._clear_btn.clicked.connect(self.clearRequested)

        self._noise_btn = QPushButton("Noise…")
        self._mask_btn = QPushButton("Mask…")
        self._psf_btn = QPushButton("PSF…")
        for kind, btn in (("noise", self._noise_btn), ("mask", self._mask_btn),
                          ("psf", self._psf_btn)):
            btn.setToolTip(
                ("Load the PSF kernel" if kind == "psf" else f"Load the {kind} map")
                + " (same shape as the image; npy/fits/…)"
            )
            btn.clicked.connect(lambda _=False, k=kind: self.loadAuxRequested.emit(k))

        for b in (self._load_btn, self._clear_btn, self._noise_btn,
                  self._mask_btn, self._psf_btn):
            lay.addWidget(b)


class FitBar(QWidget):
    """Compact strip controlling the PSO fit against the loaded data."""

    fitRequested = pyqtSignal()
    cancelRequested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        lay = QHBoxLayout(self)
        lay.setContentsMargins(4, 2, 4, 2)

        self._fit_btn = QPushButton("Fit (PSO)")
        self._fit_btn.setToolTip(
            "Fit the model to the loaded data, varying only the unlocked (🔓) "
            "parameters"
        )
        self._fit_btn.clicked.connect(self.fitRequested)
        self._cancel_btn = QPushButton("Cancel")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self.cancelRequested)

        def spin(value, lo, hi, tip):
            s = QSpinBox()
            s.setRange(lo, hi)
            s.setValue(value)
            s.setToolTip(tip)
            return s

        self._particles = spin(30, 5, 500, "PSO particles per restart")
        self._iterations = spin(100, 10, 5000, "PSO iterations per restart")
        self._restarts = spin(2, 1, 20, "number of PSO restarts (best kept)")

        # Live preview while fitting. Rendering the intermediate model costs time,
        # so it is optional and rate-limited: a larger interval means fewer
        # redraws and a faster fit; unchecking it removes the cost entirely.
        self._preview_chk = QCheckBox("preview")
        self._preview_chk.setChecked(True)
        self._preview_chk.setToolTip(
            "Draw the swarm's current model while fitting.\n"
            "Uncheck to skip preview rendering entirely (fastest)."
        )
        self._preview_interval = QDoubleSpinBox()
        self._preview_interval.setRange(0.05, 30.0)
        self._preview_interval.setDecimals(2)
        self._preview_interval.setSingleStep(0.25)
        self._preview_interval.setValue(0.5)
        self._preview_interval.setSuffix(" s")
        self._preview_interval.setToolTip(
            "Minimum time between preview redraws. Larger = fewer redraws and a "
            "faster fit."
        )
        self._preview_chk.toggled.connect(self._preview_interval.setEnabled)

        lay.addWidget(self._fit_btn)
        lay.addWidget(self._cancel_btn)
        lay.addSpacing(8)
        lay.addWidget(QLabel("particles:"))
        lay.addWidget(self._particles)
        lay.addWidget(QLabel("iterations:"))
        lay.addWidget(self._iterations)
        lay.addWidget(QLabel("restarts:"))
        lay.addWidget(self._restarts)
        lay.addSpacing(10)
        lay.addWidget(self._preview_chk)
        lay.addWidget(self._preview_interval)
        lay.addSpacing(12)
        self._status = QLabel("no fit run yet")
        self._status.setObjectName("fitStatus")
        lay.addWidget(self._status, 1)

    # ------------------------------------------------------------------ state
    def set_running(self, running: bool):
        self._fit_btn.setEnabled(not running)
        self._cancel_btn.setEnabled(running)

    def set_status(self, text: str):
        self._status.setText(text)

    def preview_enabled(self) -> bool:
        return self._preview_chk.isChecked()

    def settings(self) -> dict:
        return {
            "n_particles": self._particles.value(),
            "n_iterations": self._iterations.value(),
            "n_restarts": self._restarts.value(),
            "preview_enabled": self._preview_chk.isChecked(),
            "preview_interval": self._preview_interval.value(),
        }


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
        # Pixel scale of the model grid. This must match the external data for a
        # meaningful model-vs-data comparison (fitting).
        self._delta_pix = QDoubleSpinBox()
        self._delta_pix.setRange(0.005, 0.5)
        self._delta_pix.setDecimals(3)
        self._delta_pix.setSingleStep(0.005)
        self._delta_pix.setValue(0.05)
        self._delta_pix.setToolTip("arcsec per pixel of the model grid")
        self._delta_pix.valueChanged.connect(self._emit)
        # PSF: Gaussian FWHM in arcsec (0 = no blurring, i.e. a delta PSF).
        self._psf_fwhm = QDoubleSpinBox()
        self._psf_fwhm.setRange(0.0, 3.0)
        self._psf_fwhm.setDecimals(3)
        self._psf_fwhm.setSingleStep(0.05)
        self._psf_fwhm.setValue(0.0)
        self._psf_fwhm.setToolTip("PSF Gaussian FWHM in arcsec (0 = delta PSF)")
        self._psf_fwhm.valueChanged.connect(self._emit)
        self._cmap = QComboBox()
        self._cmap.addItems(["magma", "viridis", "plasma", "inferno", "gray", "turbo"])
        self._cmap.currentTextChanged.connect(self._emit)
        self._stretch = QComboBox()
        self._stretch.addItems(["log", "linear"])
        self._stretch.currentTextChanged.connect(self._emit)

        lay.addWidget(QLabel("numPix:"))
        lay.addWidget(self._numpix)
        lay.addSpacing(10)
        lay.addWidget(QLabel("pixel scale:"))
        lay.addWidget(self._delta_pix)
        lay.addWidget(QLabel("″/px"))
        lay.addSpacing(10)
        lay.addWidget(QLabel("PSF FWHM:"))
        lay.addWidget(self._psf_fwhm)
        lay.addWidget(QLabel("″"))
        lay.addSpacing(10)
        lay.addWidget(QLabel("colormap:"))
        lay.addWidget(self._cmap)
        lay.addSpacing(10)
        lay.addWidget(QLabel("stretch:"))
        lay.addWidget(self._stretch)
        lay.addSpacing(10)
        # Constant sky background pedestal added to the model image.
        self._sky = QDoubleSpinBox()
        self._sky.setRange(0.0, 1.0)
        self._sky.setDecimals(4)
        self._sky.setSingleStep(0.001)
        self._sky.setValue(0.0)
        self._sky.setToolTip("constant sky background level added to the model")
        self._sky.valueChanged.connect(self._emit)
        lay.addWidget(QLabel("sky:"))
        lay.addWidget(self._sky)
        lay.addSpacing(14)
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
            "delta_pix": self._delta_pix.value(),
            "psf_fwhm": self._psf_fwhm.value(),
            "sky_amp": self._sky.value(),
            "colormap": self._cmap.currentText(),
            "stretch": self._stretch.currentText(),
        }

