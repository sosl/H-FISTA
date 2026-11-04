# Greedy sparse phase retrieval: a fast CLEAN

The CLEAN-style method of Walker & Stinebring (2005) and Walker et al. (2008) works on
dynamic spectra, as H-FISTA does. It builds the wavefield one component at a time,
which makes it slow. This branch tests whether a *batched* greedy method can match
H-FISTA at a fraction of the cost. It borrows from two places:

* the major/minor cycles of Clark and Cotton–Schwab CLEAN: pick many components per
  cycle, then refit everything against the data;
* the sparse phase-retrieval literature: GESPAR (Shechtman, Beck & Eldar 2014), and
  the CoSaMP / subspace-pursuit idea of adding more candidates than needed and pruning
  back.

Builds on `claude/dense-wavefields`. The benchmark is the same: 256² noise-free
simulations, the same regimes and seeds, and the same invariant similarity, with
solved = > 0.99.

## Method (`pursuit.py`)

One round:

1. compute the gradient g of H-FISTA's demerit at the current model;
2. add the 2k allowed pixels with the largest |g| outside the support, where
   k = max(4, 0.15·|S|), each starting at the gradient step −g/L;
3. refit all values on the enlarged support: 30 iterations of unpenalised FISTA on
   the support, which is H-FISTA's debias step;
4. **prune**: keep the |S| + k largest |h| (the origin pixel is always kept) and refit.

It starts from the same single origin component as H-FISTA and has no λ and no
threshold schedule. Unlike H-FISTA's approval, step 4 can remove a component found
earlier.

It stops when the noise-free data are fitted (relative residual < 1e-6), or otherwise
at the paper's 3% sparsity cap. A residual-plateau rule doesn't work, because the
residual barely moves in the first rounds. Noisy data will need a noise-level rule
(not done yet).

## Results (`results/summary256.md`)

| method | uniform 12.5% | uniform 25% | log-uniform 25% | log-uniform 50% | FFTs per run |
|---|---|---|---|---|---|
| H-FISTA (l1) | 0.49 (2/6) | 0.19 (0/6) | **1.00 (6/6)** | 0.64 (2/6) | 32–43k |
| **greedy** | **0.64 (2/6)** | **0.20 (0/6)** | 0.99 (5/6) | **0.75 (3/6)** | **5–7k** |
| greedy, no pruning | 0.49 (2/6) | 0.19 (0/6) | 0.99 (5/6) | 0.75 (3/6) | 2.5–3.5k |
| greedy, growth 0.4 | 0.49 (1/6) | 0.19 (0/6) | 0.99 (5/6) | 0.64 (2/6) | 2.7–3.3k |
| greedy, 8 restarts (jitter 0.3), lowest residual | 0.61 (2/6) | 0.20 (0/6) | **1.00 (6/6)** | **0.76 (4/6)** | 55–93k |

Mean similarity of the stopped model (solved / runs). The l1 FFT counts are measured on
2 seeds per regime. Wall time per run is about 10 s for greedy and 45–65 s for H-FISTA
on this machine.

* **About 6× cheaper than H-FISTA, with equal or better results** in three of four
  regimes. The one loss is a single easy seed (log-uniform 25%, 0.92).
* **Pruning is what makes it better than l1**: without step 4 it's no better than H-FISTA
  on uniform 12.5%. Growing faster (0.4) loses accuracy.
* **Restarts barely help.** Eight randomised runs, keeping the lowest data residual,
  fix the easy-case miss and add one log-uniform 50% case, at 8× the cost.
* **Locally dense uniform-modulus wavefields (uniform 25%) stay unsolved** by every
  method, as with H-FISTA.

## Open next steps

* a noise-aware stopping rule (residual reaching the noise level), then the paper's
  noisy simulation and J0837/J1939;
* pruning with the firm threshold that worked on `claude/dense-wavefields` (MCP);
* greedy as a cheap first pass, followed by H-FISTA or MCP from its support.

## Files

* `pursuit.py`: `greedy_pursuit` (with `h_init` to start from a given model, `jitter`/`seed`),
  `restarted_pursuit`, `FFTCounter`
* `benchmark.py`, `summarize.py`; `results/bench256*.jsonl`, `results/l1_ffts.jsonl`, `results/summary256.md`
