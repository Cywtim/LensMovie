# LensMovie ⇄ lenstronomy: fitting-pipeline comparison

Status / scope: this compares LensMovie's fit pipeline with **lenstronomy** as
installed (`1.14.2`) and with its public API/tutorial conventions.  Source
quotes below were read from the installed package:
`Data/imaging_data.py`, `Data/image_noise.py`, `ImSim/image_model.py`,
`Sampling/Likelihoods/image_likelihood.py`, `Sampling/sampler.py`,
`Sampling/Samplers/pso.py`.

## 0. The short version

- **Same ruler.** LensMovie's reported chi² (`fd.chi2`) and the swarm's
  objective are already the *same* number: with a `noise_map`, lenstronomy's
  `Data.C_D_model()` returns `noise_map**2`, and `log_likelihood` is
  `-0.5 * sum((model-data)**2 / C_D)` over (masked) pixels — i.e. `-chi2/2`.
- **What we aligned (commit 719dd00):** the one missing variance term,
  *PSF-model error*, added in quadrature (`C_D + |additional_error_map|`
  semantics).
- **What we intentionally keep different:** see §4.

## 1. The likelihood / covariance (both sides)

lenstronomy (`Data/image_noise.py`):
```python
def C_D_model(self, model):
    if self._noise_map is not None:
        return self._noise_map ** 2
    else:
        return covariance_matrix(model, self._background_rms,
                                 self._exp_map, self._gradient_boost_factor)
```
and (`Data/imaging_data.py`):
```python
chi2 = (model - self._data)**2 / (c_d + np.abs(additional_error_map)) * mask
log_likelihood = -np.sum(chi2) / 2
```
Key consequences:
- `noise_map` present ⇒ `background_rms` / `exposure_time` / Poisson model terms
  are **ignored**; the covariance is just the given per-pixel σ².  This is
  exactly what LensMovie does — LensMovie always passes a `noise_map` (the
  loaded σ file), so its reported `fd.chi2 = Σ((model−data)/σ)²·mask` and the
  swarm's `−2·logL` are the same ruler by construction (`app/fitting.py
  _build_data_joint`).
- `additional_error_map` (currently produced from point-source **PSF-variance**,
  `_error_map_psf`) is added to the variance **in quadrature** and is
  `np.abs()`-guarded.  LensMovie now mirrors this with an optional per-pixel
  **PSF-error σ map**: `effective_noise = sqrt(noise² + psf_error²)`, used in
  `fd.chi2` **and** fed back as `noise_map` to lenstronomy so the swarm keeps
  optimizing the reported number.

## 2. Optimizer start / seeding

- lenstronomy `_init_swarm` (`Samplers/pso.py`) is **uniform random only**:
```python
swarm.append(Particle(np.random.uniform(self.low, self.high, size=self.param_count), ...))
```
- `Sampler.pso` (`Sampling/sampler.py`) injects the initial guess as the swarm's
  **global best**, not as a particle:
```python
if init_pos is None:
    init_pos = (upper_start - lower_start) / 2 + lower_start
pso.set_global_best(init_pos, [0]*len(init_pos), self.chain.logL(init_pos))
```
- LensMovie drives the raw `ParticleSwarmOptimizer` directly (to stream previews
  and the parameter chain), so it must supply the seeding itself.  Commit
  `921d1ad` added the canonical `set_global_best(init_pos, …, logL(init_pos))`
  *plus* keeps one real particle pinned at `init_pos` so the swarm also explores
  outward from it.  Net guarantee: **the result is never worse than the current
  slider values** — identical in spirit to lenstronomy's `global_best` seeding,
  stronger in practice (before this, an all-random swarm could regress from a
  sharp start; this was the "example fits badly" root cause).

## 3. Convergence / termination

- lenstronomy `ParticleSwarmOptimizer.sample()` has built-in early stop:
  `_converged_fit` (fitness spread over fraction `p` of particles) +
  `_converged_space` (position spread < `n`), plus an optional
  `early_stop_tolerance` on `chi_square = -2·logL` (`_acceptable_convergence`).
- LensMovie (commit `d395326`) exposes an opt-in **"stop χ²ν ≤"** control: the
  target reduced chi² is converted to lenstronomy's `-2·logL` scale (through the
  same offset that maps logL onto chi²) and passed as `early_stop_tolerance`;
  when a restart hits it, the remaining restarts are skipped too.  Default off —
  without it every iteration/restart runs as before.

## 4. Intentional differences (kept, documented)

| Area | lenstronomy | LensMovie | Why kept |
|---|---|---|---|
| Source model | parametric **and** pixelized / linear inversion (`LinearFit`) | parametric only | scope; parametric matches the GUI sliders 1:1 |
| Ray tracing | single **and** multi-plane | single-plane | the GUI models one z_lens → one z_source |
| Covariance model | Poisson (`exposure_time`) + background² when no `noise_map` | always an explicit total-σ `noise_map` | real reduced data carries a σ map; keeps one ruler |
| PSF uncertainty | `psf_error_map` auto-built from PSF-variance around **point sources** | user-supplied per-pixel **PSF-error σ map** | no automated PSF-variance machinery; explicit map is transparent and matches `C_D + |error_map|` |
| Convergence defaults | auto (fitness+space) | fixed iterations/restarts unless "stop χ²ν" on | predictability of run time |
| Source-plane ambiguity / lens-mass degeneracies | discussed in tutorials (MSD etc.) | noted, not modelled | out of scope for a viewer/demo |

## 5. What was verified against the example

With σ = 0.004 (SNR ~ 12):
- fit `θ_E = 1.05` recovered from a perturbed start, χ² 25181 → 3550
  (reduced χ² ≈ 0.99), i.e. the swarm+simplex reaches the noise floor — the
  renderer/likelihood/noise triples genuinely agree (a mismatch would stall
  "above the noise floor", see `fitting.py` "must match the renderer" note).
- `best-fit model` = clean signal; `residual` ≈ noise (mean ≈ 0, σ ≈ 0.004):
  the fit separates signal from noise as expected.
