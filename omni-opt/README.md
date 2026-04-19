# omni-opt

A pure-Rust, high-performance mathematical optimization framework built under
a strict **zero-allocation invariant**: every heap allocation happens during
the builder / constructor phase, and the per-iteration hot loop of every
solver is allocation-free.


## Features

### High-level: unconstrained & box-constrained optimization

- **`Solver` + `SolverBuilder`** — fluent builder, one call to `.build(n)`
  pre-allocates everything; `.run(&mut oracle, &mut x)` then drives the
  full descent loop with zero heap activity.
- **`Method` selection** — `SteepestDescent`, `BFGS` (dense rank-2 with
  curvature safeguard), `LBFGS` (two-loop recursion over the Phase-1
  ring), and `Newton` with Levenberg–Marquardt damping fallback
  (`faer` feature).
- **Line searches** — `LineSearch::StrongWolfe` (Nocedal & Wright
  Alg. 3.5/3.6, safeguarded quadratic zoom) or
  `LineSearch::BacktrackingArmijo`, each with a pre-allocated
  `LineSearchWorkspace`.
- **`BoxConstraints` + `.bounds(...)` setter** — componentwise
  `lᵢ ≤ xᵢ ≤ uᵢ` enforced by **projected clipping** after every
  accepted step. Reduced-gradient stopping criterion under bounds;
  infeasible `x₀` is silently projected. Deliberately *not* full
  L-BFGS-B — see limitations note in the
  [constrained example](#constrained-optimization-projected-clipping).
- **Structured exits** — `SolverReport { status, iters, f_evals,
  g_evals, f_final, grad_inf_final }`; `status` is a non-allocating
  `TerminationStatus` enum (Converged / MaxIterationsReached /
  LineSearchFailed / StagnationDetected / NumericalFailure).

### Linear least squares (`faer` feature)

- **`linear::solve`** over dense overdetermined systems `A x ≈ b`
  (m ≥ n), backed by faer's orthogonal factorizations.
- **Three backends**: `HouseholderQr` (fastest, full-rank only),
  `ColPivQr` (rank-revealing, **default**), `Svd` (most robust,
  rank-deficiency tolerant). **No normal-equations path** — conditioning
  stays at κ(A), never κ(A)².

### Streaming & row-action

- **`kaczmarz::run`** drives randomized Kaczmarz using only the Phase-1
  `RowAccess` trait — `A` is never materialized, so dense / CSR / CSC /
  file-backed / generator-style matrices plug in unchanged.
- **Row-sampling strategies**: `Uniform`, `SquaredRowNorm`
  (Strohmer–Vershynin, **default**), `Cyclic`. A deterministic
  internal xorshift64 RNG makes runs reproducible from `cfg.seed`.

### Oracle & gradient utilities

- **`Objective` / `Oracle`** — minimal evaluation trait, plus a fused
  `value_grad` entry point that avoids redundant forward passes.
  `HessianOracle` (`faer` feature) adds `∇²f(x)` for Newton.
- **`CentralDifferenceOracle<O>`** — one-allocation adapter that
  promotes any `Objective` to a full `Oracle` via central differences,
  keeping the solver hot loop allocation-free.

### Low-level primitives (Phase 1)

- **`OmniVec` / `OmniVecMut`** — storage-generic traits over contiguous
  `f64` buffers. Blanket impls for `[f64]`, `&[f64]`, `&mut [f64]`,
  `Vec<f64>`, and (behind `faer`) `faer::Col<f64>`.
- **`RowAccess`** — streaming matrix trait (`nrows`, `ncols`,
  `row_dot`, `row_sq_norm`, `axpy_row`).
- **`LBFGSWorkspace`** — flat `m × n` ring buffer with `n·m`
  overflow-checked allocation and a write/commit split that lets the
  solver silently discard a curvature-violating slot without tearing
  ring state.
- **`StoppingCriteria` / `TerminationStatus`** — user-configured
  tolerances and a `Copy`, heap-free exit enum.

## Installation

```toml
[dependencies]
omni-opt = "0.1"
```

Opt in to `faer::Col<f64>` interoperation, Newton/LM, and dense
orthogonal least squares:

```toml
[dependencies]
omni-opt = { version = "0.1", features = ["faer"] }
```

**MSRV:** Rust 1.75.

## Quickstart — unconstrained minimization

The high-level path. Implement `Oracle`, pick a `Method`, let the
builder do the allocation:

```rust
use omni_opt::{
    LineSearch, Method, Objective, Oracle, Solver, TerminationStatus,
};

/// f(x) = ½ ‖x − c‖² ;  ∇f(x) = x − c.
struct ShiftedQuadratic { c: Vec<f64> }

impl Objective for ShiftedQuadratic {
    fn n(&self) -> usize { self.c.len() }
    fn value(&mut self, x: &[f64]) -> f64 {
        0.5 * x.iter().zip(&self.c).map(|(xi, ci)| (xi - ci).powi(2)).sum::<f64>()
    }
}

impl Oracle for ShiftedQuadratic {
    fn value_grad(&mut self, x: &[f64], g: &mut [f64]) -> f64 {
        let mut s = 0.0;
        for i in 0..x.len() {
            g[i] = x[i] - self.c[i];
            s   += g[i] * g[i];
        }
        0.5 * s
    }
}

let mut oracle = ShiftedQuadratic { c: vec![1.0, -2.0, 3.0] };

let mut solver = Solver::builder()
    .method(Method::lbfgs())                  // L-BFGS (m = 10)
    .line_search(LineSearch::StrongWolfe)
    .grad_inf_tol(1e-8)
    .max_iter(200)
    .build(3);                                // ←— all heap allocs here

let mut x = vec![0.0; 3];
let report = solver.run(&mut oracle, &mut x); // ←— zero heap allocs

assert_eq!(report.status, TerminationStatus::Converged);
// x ≈ [1.0, -2.0, 3.0]
```

Swap `Method::lbfgs()` for `Method::bfgs()`, `Method::steepest_descent()`,
or (under `--features faer`) `Method::newton()` — the builder rebalances
the workspace; everything else stays identical.

## Constrained optimization (projected clipping)

Attach componentwise box constraints with a single builder call; the
solver clips every accepted step onto the feasible region and uses the
reduced gradient for the convergence test. `x₀` is silently projected,
so a slightly-infeasible initial guess is fine:

```rust
use omni_opt::{BoxConstraints, Method, Solver, TerminationStatus};
# use omni_opt::{Objective, Oracle};
# struct Q { c: Vec<f64> }
# impl Objective for Q {
#     fn n(&self) -> usize { self.c.len() }
#     fn value(&mut self, x: &[f64]) -> f64 {
#         0.5 * x.iter().zip(&self.c).map(|(xi,ci)|(xi-ci).powi(2)).sum::<f64>()
#     }
# }
# impl Oracle for Q {
#     fn value_grad(&mut self, x: &[f64], g: &mut [f64]) -> f64 {
#         let mut s = 0.0;
#         for i in 0..x.len() { g[i] = x[i] - self.c[i]; s += g[i] * g[i]; }
#         0.5 * s
#     }
# }

let mut oracle = Q { c: vec![5.0, 5.0] };           // unconstrained optimum
                                                     //   sits at (5, 5)…
let bounds = BoxConstraints::uniform(2, -1.0, 1.0); // …but we clamp to the
                                                     //   box [-1, 1]²
let mut solver = Solver::builder()
    .method(Method::lbfgs())
    .bounds(Some(bounds))
    .grad_inf_tol(1e-8)
    .max_iter(50)
    .build(2);

let mut x = vec![0.0, 0.0];
let report = solver.run(&mut oracle, &mut x);

assert_eq!(report.status, TerminationStatus::Converged);
// x == [1.0, 1.0] — pinned at the upper corner of the feasible box.
```

**Scope & limitations.** This is **projected clipping**, not full
L-BFGS-B. The iterate is kept feasible at every step, but the solver
does *not* perform generalized Cauchy-point search, active-set
identification, or reduced-space curvature updates. Quasi-Newton
curvature (`BFGS` / `LBFGS`) can therefore degrade mildly when many
bounds are active; convergence near the boundary may slow accordingly.
Callers requiring certified first-order optimality under active bounds
should wait for a future L-BFGS-B milestone.

Unbounded sides are expressed with `f64::INFINITY` /
`f64::NEG_INFINITY`; `BoxConstraints::upper_only` and
`BoxConstraints::lower_only` are convenience constructors for the
common one-sided cases.

## Dense linear least squares (`faer`)

Solve `A x ≈ b` with `m ≥ n` through column-pivoted QR by default.
`A`ᵀ`A` is never formed:

```rust,ignore
use faer::Mat;
use omni_opt::{linear, LinearSolver, LinearSolverWorkspace};

let a: Mat<f64> = /* m × n, m ≥ n */;
let b: Vec<f64> = /* m        */;
let mut x       = vec![0.0; a.ncols()];

let mut ws = LinearSolverWorkspace::new(a.nrows(), a.ncols(), LinearSolver::default());
let report = linear::solve(a.as_ref(), &b, &mut x, &mut ws)?;
// report.rank, report.residual_norm, report.rank_deficient
```

Explicit-opt-in alternatives: `LinearSolver::qr()` (full-rank only,
fastest), `LinearSolver::svd()` (most robust, tolerates near-singular
matrices).

## Streaming Kaczmarz

For massive or out-of-core systems — implement `RowAccess` on your
storage, and the solver never asks for a full row:

```rust
use omni_opt::{kaczmarz, KaczmarzConfig, KaczmarzSampling, KaczmarzWorkspace, RowAccess};

struct DenseRowMajor { nrows: usize, ncols: usize, data: Vec<f64> }

impl RowAccess for DenseRowMajor {
    fn nrows(&self) -> usize { self.nrows }
    fn ncols(&self) -> usize { self.ncols }

    fn row_dot(&self, i: usize, x: &[f64]) -> f64 {
        let row = &self.data[i * self.ncols .. (i + 1) * self.ncols];
        row.iter().zip(x).map(|(a, b)| a * b).sum()
    }

    fn row_sq_norm(&self, i: usize) -> f64 {
        let row = &self.data[i * self.ncols .. (i + 1) * self.ncols];
        row.iter().map(|a| a * a).sum()
    }

    fn axpy_row(&self, i: usize, alpha: f64, y: &mut [f64]) {
        let row = &self.data[i * self.ncols .. (i + 1) * self.ncols];
        for (yj, &a) in y.iter_mut().zip(row) { *yj += alpha * a; }
    }
}

let a = DenseRowMajor { /* … */ nrows: 3, ncols: 2, data: vec![1.0,0.0, 0.0,1.0, 1.0,1.0] };
let b = vec![1.0, 2.0, 3.0];
let mut x = vec![0.0; 2];

let mut ws = KaczmarzWorkspace::new(a.nrows, a.ncols, KaczmarzSampling::default());
let report = kaczmarz::run(&KaczmarzConfig::default(), &a, &b, &mut x, &mut ws);
// report.status, report.epochs, report.iters, report.residual_norm_final
```

`KaczmarzSampling::Uniform` allocates **zero** per-row scratch and is
the right choice when even `O(m)` row-norm caching is intractable.
`Cyclic` is deterministic for reproducibility tests.

## Derivative-free objectives

Wrap an `Objective` in `CentralDifferenceOracle` to get a full
`Oracle` via `2n + 1` value evaluations per gradient, all through a
single pre-allocated scratch buffer:

```rust
use omni_opt::{CentralDifferenceOracle, Objective};
// `MyLoss` implements only `Objective`.
# struct MyLoss; impl Objective for MyLoss { fn n(&self) -> usize { 0 } fn value(&mut self, _: &[f64]) -> f64 { 0.0 } }
let adapter = CentralDifferenceOracle::with_default_step(MyLoss);
// `adapter` now implements `Oracle` — feed it straight into `Solver::run`.
```

## Cargo features

| Feature | Default | Effect                                                                                 |
|---------|---------|----------------------------------------------------------------------------------------|
| `faer`  | off     | Adds `faer::Col<f64>` `OmniVec` impls, `HessianOracle`, `Method::Newton`, `linear::*`. |

## Design guarantees

- Zero heap allocation in the per-iteration hot loop of every solver
  (descent, line search, linear, Kaczmarz).
- Heap allocation confined to builder / `new` constructors.
- Structured, non-panicking termination via `Copy` enum discriminants.
- Generic over user-supplied storage — no forced copies.
- Overflow-checked buffer sizing; `debug_assert!` preconditions keep
  release builds branchless.
- No normal-equations path anywhere — orthogonal factorizations only,
  so conditioning is never squared.

## Testing & benchmarking

```sh
# Unit + integration tests (incl. the pathology suite).
cargo test
cargo test --features faer

# Lint: all targets, all features, denied warnings.
cargo clippy --all-features --all-targets -- -D warnings

# Dual-axis criterion benches.
cargo bench --bench solver_bench -- algorithmic   # f-eval counts
cargo bench --bench solver_bench -- systems       # wall-clock
```

The `tests/pathology.rs` suite pins correctness on classical edge
cases — Rosenbrock (narrow curved valley), badly-scaled anisotropic
quadratics (κ up to 10⁶), saddle points (descent-direction
correctness + faer-gated Newton/LM), and noisy central-difference
gradients (truncation vs. round-off tripwires).

The `benches/solver_bench.rs` binary wires a custom criterion
`Measurement` that reports **function / gradient evaluation counts**
with full statistical machinery (mean, stddev, outliers, change
detection) alongside the standard wall-clock group — so algorithmic
efficiency and systems performance are isolated on independent axes
of the same run.

## License

Dual-licensed under either of

- Apache License, Version 2.0
- MIT License

at your option.
