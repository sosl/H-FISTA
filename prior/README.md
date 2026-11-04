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

## Next steps if pursued

* Train at 256²/512² and on a broader family of screens (anisotropy, inner scale, several
  screens); test on screens outside the training family.
* Use a denoiser trained on |h| to penalise twin-like structure explicitly. The twin's
  arc curvature is reversed, which a prior over physical screens could learn to reject.
* Use diffusion-model posterior sampling (DDRM-PR, Kaya & Öktem 2025, Applied Optics) to
  get uncertainties and multiple solutions in place of a single PnP estimate.

## Files

* `screen.py`: simulator, `twin_split`; `bench_screen.py`: H-FISTA / greedy benchmark
* `denoiser.py`: training; `pnp.py`: plug-and-play solver and benchmark
* `greedy_local.py`: copy of `greedy/pursuit.py` from `claude/greedy-pursuit`
* `results/`: benchmark records, training logs, `denoiser128.pt`
