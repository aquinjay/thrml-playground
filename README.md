# thrml-playground

A structured sandbox for learning **THRML** (Extropic's JAX/Equinox library for probabilistic graphical models, block Gibbs sampling, and energy-based models), building intuition from small exact systems up to custom extensions, and preparing to contribute upstream.

## Setup

Requires [`uv`](https://docs.astral.sh/uv/) and Python `>=3.14`.

```bash
uv sync
uv run jupyter lab
```

## Repository Structure

```text
.
├── notebooks/
│   ├── 00_quick_test.ipynb             # Environment & logging smoke test
│   └── 01_two_interacting_spins.ipynb  # Exact 2-spin Ising EBM (E, Z, p(s), E[s1*s2])
├── src/thrml_playground/               # Reusable helpers, custom factors/observers/samplers
├── pyproject.toml
└── README.md
```

---

## Learning & Contribution Roadmap

### Phase 0 — Foundations (Done)
- [x] Verify `thrml`, `jax`, and `loguru` environment (`notebooks/00_quick_test.ipynb`)
- [x] Build a 2-spin `IsingEBM`, compute exact energies, partition function $Z$, Boltzmann probabilities, and spin correlation $\mathbb{E}[s_1 s_2]$ across $J \in \{-0.8, 0.0, +0.8\}$ (`notebooks/01_two_interacting_spins.ipynb`)

### Phase 1 — Block Gibbs Sampling & Convergence
*Target `thrml` modules: `block_management.py`, `block_sampling.py`, `models/ising.py`*

- [ ] `notebooks/02_block_gibbs_two_spins.ipynb`
  - Sample from the exact 2-spin `IsingEBM` using `IsingSamplingProgram`, `SamplingSchedule`, and `sample_states`.
  - Explore why connected spins must live in separate `free_blocks` (graph coloring / conditional independence) and verify what breaks if they are put in the same block.
  - Plot sampled $\hat{\mathbb{E}}[s_1 s_2]$ against exact $\tanh(J)$ while sweeping `n_warmup`, `n_samples`, and `steps_per_sample`.
  - Practice JAX PRNG key splitting (`jax.random.split`) and `jax.vmap` over multiple independent chains.

### Phase 2 — Scaling Up: 2D Ising Lattice & Observers
*Target `thrml` modules: `observers.py`, `block_sampling.py`*

- [ ] `notebooks/03_2d_ising_lattice.ipynb`
  - Construct an $N \times N$ 2D Ising grid (periodic or open boundaries) with checkerboard 2-coloring (`free_blocks = [even_block, odd_block]`).
  - Use `sample_with_observation` with `MomentAccumulatorObserver` instead of storing raw states.
  - Sweep inverse temperature $\beta$ across the Onsager critical point ($\beta_c = \frac{1}{2}\ln(1+\sqrt{2}) \approx 0.4407$) and plot magnetization $\langle |M| \rangle$, energy per spin, and spin-grid snapshots.
  - Benchmark `jax.jit` compile time vs. sampling throughput across lattice sizes and batch sizes.

### Phase 3 — Clamped Sampling & Training an Ising EBM / RBM
*Target `thrml` modules: `models/ising.py` (`IsingTrainingSpec`, `hinton_init`, `estimate_moments`, `estimate_kl_grad`)*

- [ ] `notebooks/04_training_ising_ebm.ipynb`
  - Experiment with `clamped_blocks` to sample conditional distributions $P(h \mid v)$.
  - Build a small Restricted Boltzmann Machine (bipartite visible/hidden spin graph) initialized with `hinton_init`.
  - Estimate positive- and negative-phase moments with `estimate_moments` and train weights/biases with `estimate_kl_grad` on synthetic binary patterns (e.g., bars-and-stripes).
  - Verify learned samples reproduce the target data distribution and track KL divergence / log-likelihood on a small enough model where $Z$ is exact.

### Phase 4 — General Discrete EBMs (Categorical & Mixed Graphs)
*Target `thrml` modules: `pgm.py`, `factor.py`, `interaction.py`, `conditional_samplers.py`, `models/ebm.py`, `models/discrete_ebm.py`*

- [ ] `notebooks/05_categorical_and_mixed_ebms.ipynb`
  - Build a Potts model using `CategoricalNode`, `CategoricalEBMFactor` / `SquareCategoricalEBMFactor`, and `CategoricalGibbsConditional`.
  - Build a mixed graph combining `SpinNode` and `CategoricalNode` with `FactorizedEBM` and `FactorSamplingProgram`.
  - Trace how `AbstractFactor.to_interaction_groups()` compiles high-level factors into directional `InteractionGroup`s consumed by `BlockSamplingProgram`.

### Phase 5 — Custom Extensions & Upstream Contributions
*Move reusable code into `src/thrml_playground/` with tests before opening PRs upstream.*

- [ ] **Read all 12 source files of `thrml`** end-to-end and note edge cases, missing docstrings/examples, type annotations (`jaxtyping`), or performance bottlenecks.
- [ ] **Build custom primitives** in `src/thrml_playground/`:
  - A custom `AbstractObserver` (e.g., online energy tracker or lag-$k$ autocorrelation / effective sample size estimator).
  - A custom `AbstractConditionalSampler` or `EBMFactor` (e.g., higher-order / 3-body interaction or continuous Gaussian node).
  - A graph-coloring helper for arbitrary `edges` → valid `free_blocks`.
- [ ] **Contribution readiness**:
  - Clone Extropic's `thrml` repo locally, run its test suite, and review open issues / PR conventions.
  - Start with a high-signal contribution: a well-tested bug fix, benchmark/example notebook, missing utility (like graph coloring or diagnostic observer), or JAX performance improvement.

---

## Core Engineering Stack to Sharpen Alongside

- **JAX**: `jit`, `vmap`, `lax.scan`, functional PRNGs, static vs. traced values, PyTree manipulation.
- **Equinox & `jaxtyping`**: `eqx.Module` immutability, `eqx.filter_jit`, strict shape/dtype annotations.
- **PGMs & Statistical Physics**: Factor graphs, chromatic number / block Gibbs parallelism, mixing time & autocorrelation, Boltzmann distributions, contrastive divergence / KL gradient estimation.
