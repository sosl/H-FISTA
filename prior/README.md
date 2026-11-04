# Realistic (thin-screen) wavefields and a learned prior

Builds on `claude/dense-wavefields`.

## A realistic benchmark (`screen.py`, `bench_screen.py`)

The earlier benchmarks used wavefields made of on-grid point components with random
phases. Here the wavefield comes from a physical simulation:

* a 2-D power-law (Kolmogorov) phase screen, anisotropy 4;
* Fresnel propagation per frequency channel (screen phase ∝ 1/ν, r_F² ∝ 1/ν, 20% bandwidth);
* the field sampled along the screen's motion over a window, so it isn't periodic in time.

D = |E|² + 1% noise, and the true wavefield is h = fft2(E): dense, off-grid, physically
structured. The scattering strength σ_φ (rms screen phase) runs from sparse to dense:

| regime | σ_φ | pixels holding 90% of the power | origin's share of the power |
|---|---|---|---|
| S3 | 3 rad | ~0.1% | ~30% |
| S10 | 10 rad | ~3% | ~2% |
| S30 | 30 rad | ~12% | ~0.2% |

Each method is followed by the paper's densification. Similarity to the true wavefield is
invariant to global phase, translation and the twin image. `twin_split` measures, with the
origin pixel excluded, the separate match of a model to the truth and to its twin.

### H-FISTA and greedy pursuit

| regime | H-FISTA 256² | greedy 256² | H-FISTA 128² | greedy 128² |
|---|---|---|---|---|
| S3 | 0.61 | 0.59 | 0.69 | 0.70 |
| S10 | 0.25 | 0.23 | 0.30 | 0.29 |
| S30 | 0.10 | 0.09 | 0.12 | 0.11 |

(mean of 4 seeds, dense models). **Both fit the data to the noise level in every case**
(dense relative residual 4e-5 to 7e-4), yet recover only part of the wavefield, and very
little of it in the dense regimes.

**Twin images explain much of it.** At S3 the H-FISTA models match the twin of the truth
as strongly as the truth itself (0.50 vs 0.51 at 128², origin excluded). The models are
mixtures of the wavefield and its twin, which fit |E|² equally well. With dense
wavefields the half-plane constraint does not separate them: the twin of a component at
positive delay lies at negative delay, but cross terms between components do not.

### Locally dense fields: does H-FISTA concentrate power? (`coarse_power.py`)

Hypothesis: in locally dense regions, H-FISTA puts the power of a cluster into a single
component, and a coarser (block) model should help. Diagnostics on the same 128² cases:

* **Block concentration.** Within the 4×4 blocks that hold 90% of the true (non-origin)
  power, take the share of each block's power held by its brightest pixel
  (power-weighted mean):

  | regime | truth | H-FISTA sparse | H-FISTA dense | group 2×2 sparse | greedy sparse |
  |---|---|---|---|---|---|
  | S3 | 0.53 | 0.51 | 0.51 | 0.48 | 0.54 |
  | S10 | 0.26 | 0.42 | 0.41 | 0.33 | 0.42 |
  | S30 | 0.23 | 0.60 | 0.43 | 0.54 | 0.53 |

  **Confirmed:** in the dense regimes H-FISTA's models are 1.6–2.6× too concentrated,
  and densification only partly spreads the power. The 2×2 group penalty brings the
  concentration closer to the truth.
* **Coarse power is largely right.** The normalised correlation of the dense model's
  block-power map with the truth's, best over shifts, for H-FISTA:

  | regime | pixel | 2×2 | 4×4 | 8×8 |
  |---|---|---|---|---|
  | S10 | 0.49 | 0.75 | 0.86 | 0.92 |
  | S30 | 0.30 | 0.51 | 0.72 | 0.84 |

  The twin's block-power map correlates almost as well (e.g. 0.81 at S10, 4×4), so this
  measure says little about the twin.
* **Block penalties don't fix the field.** Field similarity (dense, mean of 4 seeds;
  `results/screen128_group.jsonl`):

  | regime | l1 | group 2×2 | fixed 2→1 | adaptive 4→2→1 | greedy |
  |---|---|---|---|---|---|
  | S3 | 0.69 | 0.69 | 0.68 | 0.68 | 0.70 |
  | S10 | 0.30 | 0.34 | 0.33 | 0.28 | 0.29 |
  | S30 | 0.12 | 0.11 | 0.11 | 0.10 | 0.11 |

  The S10 gain for the 2×2 groups comes mostly from one seed (0.40 → 0.54).

**Conclusion.** The coarse power distribution is recovered reasonably well, and its
over-concentration can be reduced with group penalties. The phases inside the dense
regions (and the twin mixture) are what is lost, and a better power distribution alone
doesn't recover them.

**Extending H-FISTA's support with the coarse solution, then refitting with no l1 penalty**
(`coarse_support.py`, `results/coarse_support128.jsonl`).
* **Setup:** start from the H-FISTA model and refit with 1000 FISTA iterations on its
  support, extended by 4×4 blocks. Fits are unpenalised or ridge-penalised: μ‖h‖² with
  μ = c·L₀, c = 0.01 or 0.1.
* **Block sources:**
  * dense90: the blocks holding 90% of the densified model's power;
  * group: blocks touched by the 2×2 group model;
  * oracle: the blocks holding 90% of the true power.
* **Baseline:** the paper's densification, i.e. the whole half plane, unpenalised.

Mean similarity (4 seeds) and median relative residual:

| regime | l1 support only | + dense90 | + group | + **oracle** | full half plane |
|---|---|---|---|---|---|
| S3 | 0.69 (6e-2) | 0.69 (6e-2) | 0.66 (4e-2) | 0.70 (6e-2) | 0.69 (6e-4) |
| S10 | 0.30 (0.21) | 0.29 (0.15) | 0.31 (0.12) | 0.30 (0.16) | 0.30 (4e-4) |
| S30 | 0.11 (0.54) | 0.10 (0.14) | 0.10 (0.45) | 0.13 (0.18) | 0.12 (4e-4) |

(c = 0; ridge changes the similarity by ≤ 0.03, sometimes up and sometimes down.)

* **Even the true coarse support barely helps** (S30: 0.12 → 0.13).
* **Restricted supports don't fit the data:** relative residuals of 0.15–0.5 at S10/S30,
  even for the oracle support covering a third of the plane at S30. The 10% of the power
  outside the main blocks is spread thinly over the whole plane. Through cross terms with
  the bright components it still carries much of the dynamic spectrum's structure.
* Only the full half plane fits, and it still lands in a twin mixture. Starting from the
  l1 model keeps what H-FISTA got right. But with a fixed support and no penalty, nothing
  picks between the true and twin phases inside a dense region.

**Chromatic smearing is not the cause.** Rescaling time per channel (t' = t (ν/ν₀)^p,
p = ±1, 2, with the uncovered samples masked) does not make the simulated wavefield
sparser: the share of pixels holding 90% of the power changes by ≤ 15%. The paper's
J0837 data have 0.5% fractional bandwidth in any case.

## A learned prior: plug-and-play (`denoiser.py`, `pnp.py`)

* **Denoiser:** an 8-layer CNN (32 channels) trained on 1500 thin-screen wavefields at
  128², with σ_φ log-uniform in [2, 40] rad and seeds disjoint from the test seeds.
  * It works on the fftshifted grid, with the origin removed, normalised by the rms of
    the remaining pixels.
  * It sees a compressed input x/(1+|x|) and predicts the noise in units of σ.
  * Loss: log(output MSE / noise power), averaged over samples.
  * Three earlier versions failed to learn (`results/denoiser128_v*.log`): the origin
    swamped the input, a 1/σ²-weighted loss let tiny-noise samples dominate, and the
    ~4-decade value range blocked optimisation.
  * After 15 epochs it gains 4.8 dB (median; 3.4 dB at σ < 0.05, 7.4 dB at σ ~ 0.1–1).
    That's somewhat better than a complex soft threshold (≈4 dB at σ = 0.3).
* **Plug-and-play FISTA:** H-FISTA's demerit with the prox replaced by the denoiser,
  forbidden delays zeroed, B3 backtracking. The denoising strength decreases
  geometrically from 1 to 0.01 (relative rms) over 20 stages of 20 iterations, in the
  role of H-FISTA's λ schedule.

128², 4 seeds, dense models (`results/pnp128.jsonl`, `results/screen128.jsonl`):

| regime | H-FISTA | greedy | **plug-and-play** | match to truth / twin (H-FISTA → PnP) |
|---|---|---|---|---|
| S3 | 0.69 | 0.70 | **0.76** | 0.50 / 0.51 → 0.58 / 0.47 |
| S10 | 0.30 | 0.29 | **0.37** (one seed 0.40 → 0.67) | 0.26 / 0.17 → 0.32 / 0.17 |
| S30 | 0.12 | 0.11 | 0.11 | 0.10 / 0.08 → 0.09 / 0.07 |

* **The first method on any branch to do measurably better than H-FISTA on realistic
  wavefields**, in the sparse and intermediate regimes. It shifts models towards the truth
  and away from the twin.
* **It doesn't help in the dense regime (S30)**, where every method recovers ~10%.
* **Caveat:** the denoiser was trained on the same simulator it's tested on (different
  seeds), so this is a best case. On real data the gain depends on how well the
  simulator's statistics match the real screen. Validation on J1939 / J0837 has no truth;
  only the truth-free diagnostics would be available.
* It's slower: ~20 s against 2–7 s per 128² case (CNN on CPU).

**Out of distribution** (`results/pnp128_ood.jsonl`). The same denoiser (trained on
anisotropy 4, r_F = 15) is used on screens with other anisotropies and a smaller
Fresnel scale. Mean dense similarity over 4 seeds, H-FISTA → PnP:

| screen | S3 | S10 |
|---|---|---|
| anisotropy 1.5 | 0.64 → 0.65 | 0.25 → 0.30 |
| anisotropy 12 | 0.78 → **0.87** | 0.53 → **0.65** |
| r_F = 8 | 0.74 → 0.75 | 0.33 → 0.37 |

* **The gain survives** outside the training family: PnP is better on average in all six
  settings, and better per seed in 18 of 24 cases.
* **The gain is largest for very anisotropic screens:** sharp arcs, more like many real
  pulsars.
* It is smallest for nearly isotropic screens and at weak scattering.
* The other simulator assumptions still apply: a single thin screen, Kolmogorov
  spectrum, this noise model.

**A broader denoiser** (`results/denoiser128_broad.pt`):
* **Training:** 3000 screens with anisotropy log-uniform in [1.5, 12] and r_F in [6, 20];
  10 layers of 48 channels; 25 epochs, ~2 h on 3 CPU threads.
* **Denoising:** median validation gain ~7 dB, against 4.8 dB for the first model.
* **Evaluation:** PnP on the main benchmark, the settings above, and screens outside
  the new family (anisotropy 1 and 25, r_F 4 and 30). Mean dense similarity, 4 seeds
  (`results/pnp128_broad*.jsonl`, `results/pnp128_narrow_far.jsonl`):

| screen | regime | H-FISTA | PnP, first denoiser | PnP, broad denoiser |
|---|---|---|---|---|
| anisotropy 4, r_F 15 (main) | S3 | 0.69 | 0.76 | 0.74 |
| | S10 | 0.30 | 0.37 | 0.37 |
| | S30 | 0.12 | 0.11 | **0.14** |
| anisotropy 1 (outside both) | S3 | **0.68** | 0.63 | 0.59 |
| | S10 | 0.29 | 0.37 | 0.36 |
| anisotropy 1.5 | S3 | 0.64 | 0.65 | 0.68 |
| | S10 | 0.25 | 0.30 | 0.34 |
| anisotropy 12 | S3 | 0.78 | 0.87 | 0.87 |
| | S10 | 0.53 | 0.65 | 0.66 |
| anisotropy 25 (outside both) | S3 | 0.85 | 0.95 | 0.94 |
| | S10 | 0.76 | 0.83 | 0.82 |
| r_F 4 (outside both) | S3 | 0.72 | 0.77 | 0.78 |
| | S10 | 0.44 | 0.52 | 0.53 |
| r_F 8 | S3 | 0.74 | 0.75 | 0.77 |
| | S10 | 0.33 | 0.37 | 0.37 |
| r_F 30 (outside both) | S3 | 0.57 | 0.59 | 0.58 |
| | S10 | 0.21 | 0.22 | 0.25 |

* **PnP with the broad denoiser beats H-FISTA in 55 of 68 cases**, including screens
  outside its training family.
* **One exception: isotropic screens at weak scattering**, where H-FISTA is better
  (0.68 against 0.59–0.63).
* **The broad denoiser is only marginally better than the first** (better in 42 of 68
  cases), despite denoising ~2 dB better. Its one notable gain is the first improvement
  in the dense regime, S30 0.12 → 0.14.
* **Denoising quality is not the bottleneck.** The remaining error is the twin ambiguity,
  which a better denoiser of single fields does not break.

## Next steps if pursued

* Train at 256²/512² and with several screens or other inner scales.
* Use a denoiser trained on |h| to penalise twin-like structure explicitly. The twin's
  arc curvature is reversed, which a prior over physical screens could learn to reject.
* Use diffusion-model posterior sampling (DDRM-PR, Kaya & Öktem 2025, Applied Optics) to
  get uncertainties and multiple solutions in place of a single PnP estimate.

## Files

* `screen.py`: simulator, `twin_split`; `bench_screen.py`: H-FISTA / greedy benchmark
* `denoiser.py`: training; `pnp.py`: plug-and-play solver and benchmark
* `greedy_local.py`: copy of `greedy/pursuit.py` from `claude/greedy-pursuit`
* `results/`: benchmark records, training logs, `denoiser128.pt`
