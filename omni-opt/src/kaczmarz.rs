//! Randomized Kaczmarz least-squares solver operating entirely
//! through the Phase-1 [`crate::RowAccess`] trait.
//!
//! # Zero-materialization contract
//!
//! The algorithm touches `A` only via the four primitives
//! `nrows`, `ncols`, `row_dot`, `row_sq_norm`, and `axpy_row`.
//! No `A[i, :]` is ever materialized as an owned `Vec<f64>`, so
//! dense / CSR / CSC / file-backed / generator-style matrices
//! plug into the same [`run`] entry point without change.
//!
//! # Per-iteration memory footprint
//!
//! * `x: &mut [f64]` (length `n`) — the solution, caller-owned.
//! * `b: &[f64]` (length `m`) — the RHS, caller-owned.
//! * Nothing else is written in the inner loop. One `row_dot` +
//!   one `row_sq_norm` lookup + one `axpy_row` per iteration,
//!   `O(n)` arithmetic and `O(n)` memory traffic regardless of
//!   `m`. Workspace scratch (`row_norms_sq`, cumulative weights)
//!   is populated once per run and then read-only.
//!
//! `x` and `b` are always caller-owned.

// Clippy escape: numerical kernels index multiple slices in
// lockstep; rewriting via `enumerate()` still requires indexing
// the remaining slices by `i` and obscures the math.
#![allow(clippy::needless_range_loop)]

use crate::traits::RowAccess;
use crate::state::TerminationStatus;

// =============================================================
// Sampling / config / report types.
// =============================================================

/// Row sampling strategy.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum KaczmarzSampling {
    /// Uniform i.i.d. row sampling. Cheapest — no per-row
    /// scratch. Use when `m` is so large that even caching
    /// `‖Aᵢ‖²` is prohibitive.
    Uniform,
    /// Strohmer–Vershynin (2009) importance sampling: row `i`
    /// drawn with probability proportional to `‖Aᵢ‖²`. Requires
    /// the `O(m)` row-norm cache; gives the best theoretical
    /// convergence rate for poorly-scaled matrices.
    #[default]
    SquaredRowNorm,
    /// Deterministic cyclic sweep `0, 1, …, m−1`, wrap. No
    /// randomization; included for reproducibility / debugging.
    Cyclic,
}

/// User-configurable parameters.
#[derive(Clone, Copy, Debug)]
pub struct KaczmarzConfig {
    /// Maximum number of epochs (1 epoch = `m` iterations).
    pub max_epochs: u32,

    /// Relaxation parameter `ω`. Must satisfy `0 < ω < 2` for
    /// the iteration to stay non-expansive.
    pub relaxation: f64,

    /// Row-sampling strategy.
    pub sampling: KaczmarzSampling,

    /// Terminate when `‖A x − b‖₂ / max(‖b‖₂, 1) ≤ residual_tol`.
    pub residual_tol: f64,

    /// Terminate when the epoch-cumulative step norm
    /// `√(Σ ‖Δxₖ‖²) / (1 + ‖x‖) ≤ step_tol`.
    pub step_tol: f64,

    /// How often to evaluate the expensive `O(mn)` residual
    /// check: every `check_residual_every_epochs` epochs. `1`
    /// = every epoch. Default `1`.
    pub check_residual_every_epochs: u32,

    /// Deterministic RNG seed for `Uniform` / `SquaredRowNorm`.
    /// Two runs with the same seed and matrix produce identical
    /// iteration sequences. **Must be non-zero** (xorshift64
    /// degenerates on seed 0); enforced by `run`.
    pub seed: u64,
}

impl Default for KaczmarzConfig {
    fn default() -> Self {
        Self {
            max_epochs: 100,
            relaxation: 1.0,
            sampling: KaczmarzSampling::default(),
            residual_tol: 1e-8,
            step_tol: 1e-12,
            check_residual_every_epochs: 1,
            seed: 0xDEAD_BEEF_CAFE_BABE,
        }
    }
}

/// Solver outcome. `Copy`, heap-free.
#[derive(Clone, Copy, Debug)]
pub struct KaczmarzReport {
    /// Structured exit state.
    pub status: TerminationStatus,
    /// Number of epochs executed.
    pub epochs: u32,
    /// Total inner iterations (≤ `epochs · m`).
    pub iters: u64,
    /// `‖A x_final − b‖₂`. `f64::NAN` if no residual was ever
    /// computed (e.g., the run terminated before the first
    /// scheduled check).
    pub residual_norm_final: f64,
}

// =============================================================
// Workspace.
// =============================================================

/// Pre-allocated scratch for the solver.
///
/// Construction allocates up to two `Vec<f64>` of length `m`
/// (only for [`KaczmarzSampling::SquaredRowNorm`]); everything
/// else is inline scalars. `x` and `b` are caller-owned — the
/// workspace never allocates them.
pub struct KaczmarzWorkspace {
    m: usize,
    n: usize,
    sampling: KaczmarzSampling,

    // For `SquaredRowNorm`: per-row squared norms cached on
    // first `run` and a prefix-sum table for O(log m) sampling.
    // Empty `Vec`s otherwise.
    row_norms_sq: Vec<f64>,
    cumulative_weights: Vec<f64>,

    // `true` once `row_norms_sq` / `cumulative_weights` reflect
    // the current matrix. Cleared by `invalidate_row_norms`.
    row_norms_ready: bool,

    // xorshift64 generator. Reseeded at the start of every
    // `run` call from `cfg.seed` (so a run is fully
    // deterministic given its config, independent of any
    // previous run against the same workspace).
    rng_state: u64,
}

impl KaczmarzWorkspace {
    /// Allocate scratch. `m` and `n` are locked at construction
    /// for dimension-check purposes.
    ///
    /// # Panics
    ///
    /// Panics on `m == 0`, `n == 0`, or `m < n`. Kaczmarz as
    /// deployed here targets **overdetermined** least-squares
    /// problems; underdetermined systems require a distinct
    /// (minimum-norm) formulation that is out of scope for this
    /// module.
    pub fn new(m: usize, n: usize, sampling: KaczmarzSampling) -> Self {
        assert!(m > 0, "KaczmarzWorkspace::new: m must be > 0");
        assert!(n > 0, "KaczmarzWorkspace::new: n must be > 0");
        assert!(
            m >= n,
            "KaczmarzWorkspace::new: require m ({}) >= n ({}) for an overdetermined system",
            m,
            n
        );
        let (row_norms_sq, cumulative_weights) = match sampling {
            KaczmarzSampling::SquaredRowNorm => (vec![0.0; m], vec![0.0; m]),
            KaczmarzSampling::Uniform | KaczmarzSampling::Cyclic => (Vec::new(), Vec::new()),
        };
        Self {
            m,
            n,
            sampling,
            row_norms_sq,
            cumulative_weights,
            row_norms_ready: false,
            rng_state: 1, // re-seeded on every `run`
        }
    }

    /// Number of rows this workspace was built for.
    pub fn m(&self) -> usize {
        self.m
    }

    /// Number of unknowns this workspace was built for.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Row-sampling strategy fixed at construction.
    pub fn sampling(&self) -> KaczmarzSampling {
        self.sampling
    }

    /// Invalidate the cached row norms. Call when the
    /// underlying matrix has been mutated between runs;
    /// the next `run` will recompute the cache.
    pub fn invalidate_row_norms(&mut self) {
        self.row_norms_ready = false;
    }
}

// =============================================================
// Driver.
// =============================================================

/// Run randomized Kaczmarz on `matrix · x = b`.
///
/// # Preconditions
///
/// * `matrix.nrows() == workspace.m()` and
///   `matrix.ncols() == workspace.n()`.
/// * `b.len() == workspace.m()`, `x.len() == workspace.n()`.
/// * `0.0 < cfg.relaxation < 2.0`.
/// * `cfg.seed != 0` (xorshift64 degenerates otherwise).
///
/// All enforced with `debug_assert_eq!` / `debug_assert!` so
/// release builds stay branchless.
///
/// # Behaviour
///
/// Every iteration executes the classical projection
///
/// ```text
///   x  ←  x  +  ω · (bᵢ − Aᵢ·x) / ‖Aᵢ‖² · Aᵢᵀ.
/// ```
///
/// Termination status mapping:
///
/// * [`TerminationStatus::Converged`] — residual or step test met.
/// * [`TerminationStatus::MaxIterationsReached`] — ran out of epochs.
/// * [`TerminationStatus::NumericalFailure`] — NaN / Inf observed.
pub fn run<R: RowAccess>(
    cfg: &KaczmarzConfig,
    matrix: &R,
    b: &[f64],
    x: &mut [f64],
    workspace: &mut KaczmarzWorkspace,
) -> KaczmarzReport {
    debug_assert_eq!(matrix.nrows(), workspace.m);
    debug_assert_eq!(matrix.ncols(), workspace.n);
    debug_assert_eq!(b.len(), workspace.m);
    debug_assert_eq!(x.len(), workspace.n);
    // Overdetermined systems only — redundant with the same check
    // in `KaczmarzWorkspace::new` but re-asserted here as a
    // defence against an ill-constructed workspace sneaking
    // through unit tests.
    debug_assert!(
        workspace.m >= workspace.n,
        "kaczmarz::run: require m >= n for an overdetermined system"
    );
    debug_assert!(
        cfg.relaxation > 0.0 && cfg.relaxation < 2.0,
        "kaczmarz::run: relaxation ω must satisfy 0 < ω < 2"
    );
    debug_assert!(cfg.seed != 0, "kaczmarz::run: seed must be non-zero");
    // Numeric-threshold validation: each tolerance must be a
    // finite, non-negative f64; the iteration budgets must be
    // strictly positive. Caught in debug so release builds stay
    // branchless.
    debug_assert!(
        cfg.residual_tol.is_finite() && cfg.residual_tol >= 0.0,
        "kaczmarz::run: residual_tol must be finite and non-negative (got {})",
        cfg.residual_tol
    );
    debug_assert!(
        cfg.step_tol.is_finite() && cfg.step_tol >= 0.0,
        "kaczmarz::run: step_tol must be finite and non-negative (got {})",
        cfg.step_tol
    );
    debug_assert!(
        cfg.max_epochs > 0,
        "kaczmarz::run: max_epochs must be > 0"
    );
    debug_assert!(
        cfg.check_residual_every_epochs > 0,
        "kaczmarz::run: check_residual_every_epochs must be > 0"
    );

    let m = workspace.m;

    // Seed the RNG for this run. We deliberately re-seed on
    // every call so reruns against the same workspace are
    // reproducible.
    workspace.rng_state = cfg.seed;

    // Populate the row-norm cache the first time we see a given
    // matrix (or after an explicit invalidation).
    if workspace.sampling == KaczmarzSampling::SquaredRowNorm && !workspace.row_norms_ready {
        let mut total = 0.0;
        for i in 0..m {
            let w = matrix.row_sq_norm(i);
            workspace.row_norms_sq[i] = w;
            total += w;
            workspace.cumulative_weights[i] = total;
        }
        if !total.is_finite() || total == 0.0 {
            // Degenerate matrix — every row is zero. No projection
            // will ever update `x`. Report NumericalFailure
            // rather than loop forever.
            return KaczmarzReport {
                status: TerminationStatus::NumericalFailure,
                epochs: 0,
                iters: 0,
                residual_norm_final: f64::NAN,
            };
        }
        workspace.row_norms_ready = true;
    }

    let b_norm = l2_norm(b).max(1.0);
    let mut status = TerminationStatus::MaxIterationsReached;
    let mut epochs_done: u32 = 0;
    let mut iters_done: u64 = 0;
    let mut residual_norm_final = f64::NAN;

    'outer: while epochs_done < cfg.max_epochs {
        // Per-epoch step-norm accumulator.
        let mut step_norm_sq_epoch = 0.0;

        for _ in 0..m {
            let i = match workspace.sampling {
                KaczmarzSampling::Uniform => {
                    uniform_row(&mut workspace.rng_state, m)
                }
                KaczmarzSampling::SquaredRowNorm => weighted_row(
                    &mut workspace.rng_state,
                    &workspace.cumulative_weights,
                ),
                KaczmarzSampling::Cyclic => (iters_done as usize) % m,
            };

            // Compute residual along row i, skip zero-norm rows.
            let row_norm_sq = match workspace.sampling {
                KaczmarzSampling::SquaredRowNorm => workspace.row_norms_sq[i],
                _ => matrix.row_sq_norm(i),
            };
            if row_norm_sq == 0.0 || !row_norm_sq.is_finite() {
                iters_done = iters_done.saturating_add(1);
                continue;
            }

            let ax_i = matrix.row_dot(i, x);
            let residual_i = b[i] - ax_i;
            if !residual_i.is_finite() {
                status = TerminationStatus::NumericalFailure;
                break 'outer;
            }

            let alpha = cfg.relaxation * residual_i / row_norm_sq;
            if !alpha.is_finite() {
                status = TerminationStatus::NumericalFailure;
                break 'outer;
            }

            // x ← x + α · Aᵢᵀ.
            matrix.axpy_row(i, alpha, x);

            // Step norm: ‖Δx‖² = α² · ‖Aᵢ‖².
            step_norm_sq_epoch += alpha * alpha * row_norm_sq;
            iters_done = iters_done.saturating_add(1);
        }

        epochs_done = epochs_done.saturating_add(1);

        // Step-norm is a *heuristic*, not a convergence proof:
        // under randomized sampling `step_norm_sq_epoch` can hit
        // zero simply because the epoch happened to re-project
        // onto rows already at their targets, without having
        // visited every row. We therefore use it only to trigger
        // an *early* residual check; the residual test is the
        // authoritative convergence signal.
        let step_norm_epoch = step_norm_sq_epoch.sqrt();
        let x_norm = l2_norm(x);
        let step_heuristic_tripped =
            step_norm_epoch <= cfg.step_tol * (1.0 + x_norm);

        let check_every = cfg.check_residual_every_epochs.max(1);
        let scheduled_check = epochs_done % check_every == 0;

        if step_heuristic_tripped || scheduled_check {
            let r_norm = residual_l2_norm(matrix, x, b);
            if !r_norm.is_finite() {
                status = TerminationStatus::NumericalFailure;
                residual_norm_final = r_norm;
                break 'outer;
            }
            residual_norm_final = r_norm;
            if r_norm <= cfg.residual_tol * b_norm {
                status = TerminationStatus::Converged;
                break 'outer;
            }
        }
    }

    // Ensure the report always carries a residual if one is
    // available for free (cheap when the status is already
    // decided).
    if residual_norm_final.is_nan() && status != TerminationStatus::NumericalFailure {
        residual_norm_final = residual_l2_norm(matrix, x, b);
    }

    KaczmarzReport {
        status,
        epochs: epochs_done,
        iters: iters_done,
        residual_norm_final,
    }
}

// =============================================================
// Internal helpers.
// =============================================================

/// xorshift64 — 64-bit internal state, period `2⁶⁴ − 1`. Used
/// for both uniform and weighted sampling. Zero-state is illegal
/// (the generator degenerates to 0), enforced by the
/// `cfg.seed != 0` precondition.
#[inline]
fn xorshift64_next(state: &mut u64) -> u64 {
    let mut x = *state;
    x ^= x << 13;
    x ^= x >> 7;
    x ^= x << 17;
    *state = x;
    x
}

#[inline]
fn uniform_row(state: &mut u64, m: usize) -> usize {
    // Modulo bias is negligible for `m` small relative to
    // `u64::MAX`, which is the overwhelmingly common case.
    (xorshift64_next(state) % m as u64) as usize
}

/// Binary search over a non-decreasing cumulative-weight array.
/// Returns the smallest `i` such that `cum[i] ≥ target`.
#[inline]
fn weighted_row(state: &mut u64, cum: &[f64]) -> usize {
    let m = cum.len();
    debug_assert!(m > 0);
    let total = cum[m - 1];
    // Uniform in [0, total) by casting the upper 53 bits of the
    // xorshift output into [0, 1).
    let u01 = (xorshift64_next(state) >> 11) as f64 / ((1u64 << 53) as f64);
    let target = u01 * total;

    let (mut lo, mut hi) = (0usize, m - 1);
    while lo < hi {
        let mid = (lo + hi) / 2;
        if cum[mid] < target {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    lo
}

#[inline]
fn l2_norm(v: &[f64]) -> f64 {
    let mut s = 0.0;
    for &x in v {
        s += x * x;
    }
    s.sqrt()
}

/// `‖A x − b‖₂` via a reduce loop over rows. No scratch vector;
/// re-issues `row_dot` per row (the Director-approved trade-off).
#[inline]
fn residual_l2_norm<R: RowAccess>(matrix: &R, x: &[f64], b: &[f64]) -> f64 {
    let m = matrix.nrows();
    let mut s = 0.0;
    for i in 0..m {
        let r = matrix.row_dot(i, x) - b[i];
        s += r * r;
    }
    s.sqrt()
}

// =============================================================
// Tests.
// =============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::traits::RowAccess;

    /// Tiny dense row-major matrix implementation for tests.
    /// Also exercises the zero-materialization contract: the
    /// solver only sees `DenseRowMajor` via `RowAccess`.
    struct DenseRowMajor {
        nrows: usize,
        ncols: usize,
        data: Vec<f64>,
    }

    impl RowAccess for DenseRowMajor {
        fn nrows(&self) -> usize {
            self.nrows
        }
        fn ncols(&self) -> usize {
            self.ncols
        }
        fn row_dot(&self, i: usize, x: &[f64]) -> f64 {
            let row = &self.data[i * self.ncols..(i + 1) * self.ncols];
            let mut s = 0.0;
            for j in 0..self.ncols {
                s += row[j] * x[j];
            }
            s
        }
        fn row_sq_norm(&self, i: usize) -> f64 {
            let row = &self.data[i * self.ncols..(i + 1) * self.ncols];
            let mut s = 0.0;
            for j in 0..self.ncols {
                s += row[j] * row[j];
            }
            s
        }
        fn axpy_row(&self, i: usize, alpha: f64, y: &mut [f64]) {
            let row = &self.data[i * self.ncols..(i + 1) * self.ncols];
            for j in 0..self.ncols {
                y[j] += alpha * row[j];
            }
        }
    }

    fn identity_matrix(n: usize) -> DenseRowMajor {
        let mut data = vec![0.0; n * n];
        for i in 0..n {
            data[i * n + i] = 1.0;
        }
        DenseRowMajor {
            nrows: n,
            ncols: n,
            data,
        }
    }

    #[test]
    fn converges_on_identity_system() {
        let a = identity_matrix(4);
        let b = vec![1.0, -2.0, 3.0, 0.5];
        let mut x = vec![0.0; 4];
        let mut ws = KaczmarzWorkspace::new(4, 4, KaczmarzSampling::SquaredRowNorm);
        let cfg = KaczmarzConfig {
            max_epochs: 50,
            ..KaczmarzConfig::default()
        };
        let r = run(&cfg, &a, &b, &mut x, &mut ws);
        assert_eq!(r.status, TerminationStatus::Converged);
        for (got, want) in x.iter().zip(&b) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
        assert!(r.residual_norm_final < 1e-6);
    }

    #[test]
    fn converges_on_full_rank_overdetermined_system() {
        // 3 equations, 2 unknowns; exact solution x = [1, 2].
        let a = DenseRowMajor {
            nrows: 3,
            ncols: 2,
            data: vec![
                1.0, 0.0, //
                0.0, 1.0, //
                1.0, 1.0,
            ],
        };
        let b = vec![1.0, 2.0, 3.0];
        let mut x = vec![0.0, 0.0];
        let mut ws = KaczmarzWorkspace::new(3, 2, KaczmarzSampling::SquaredRowNorm);
        let cfg = KaczmarzConfig {
            max_epochs: 200,
            residual_tol: 1e-8,
            ..KaczmarzConfig::default()
        };
        let r = run(&cfg, &a, &b, &mut x, &mut ws);
        assert_eq!(r.status, TerminationStatus::Converged);
        assert!((x[0] - 1.0).abs() < 1e-5);
        assert!((x[1] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn cyclic_sampling_is_deterministic() {
        let a = identity_matrix(3);
        let b = vec![1.0, 2.0, 3.0];
        let mut x1 = vec![0.0; 3];
        let mut x2 = vec![0.0; 3];
        let mut ws1 = KaczmarzWorkspace::new(3, 3, KaczmarzSampling::Cyclic);
        let mut ws2 = KaczmarzWorkspace::new(3, 3, KaczmarzSampling::Cyclic);
        let cfg = KaczmarzConfig {
            sampling: KaczmarzSampling::Cyclic,
            max_epochs: 5,
            residual_tol: 1e-20,     // force full max_epochs
            step_tol: 0.0,            // disable step-norm exit
            ..KaczmarzConfig::default()
        };
        let r1 = run(&cfg, &a, &b, &mut x1, &mut ws1);
        let r2 = run(&cfg, &a, &b, &mut x2, &mut ws2);
        assert_eq!(r1.iters, r2.iters);
        assert_eq!(x1, x2);
    }

    #[test]
    fn uniform_sampling_is_deterministic_given_seed() {
        let a = identity_matrix(4);
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut x1 = vec![0.0; 4];
        let mut x2 = vec![0.0; 4];
        let mut ws1 = KaczmarzWorkspace::new(4, 4, KaczmarzSampling::Uniform);
        let mut ws2 = KaczmarzWorkspace::new(4, 4, KaczmarzSampling::Uniform);
        let cfg = KaczmarzConfig {
            sampling: KaczmarzSampling::Uniform,
            max_epochs: 3,
            residual_tol: 1e-20,
            step_tol: 0.0,
            seed: 0x1234_5678_9ABC_DEF0,
            ..KaczmarzConfig::default()
        };
        let _ = run(&cfg, &a, &b, &mut x1, &mut ws1);
        let _ = run(&cfg, &a, &b, &mut x2, &mut ws2);
        assert_eq!(x1, x2);
    }

    #[test]
    fn invalidate_row_norms_forces_recache() {
        let a = identity_matrix(3);
        let b = vec![1.0, 2.0, 3.0];
        let mut x = vec![0.0; 3];
        let mut ws = KaczmarzWorkspace::new(3, 3, KaczmarzSampling::SquaredRowNorm);
        let cfg = KaczmarzConfig::default();
        let _ = run(&cfg, &a, &b, &mut x, &mut ws);
        assert!(ws.row_norms_ready);
        ws.invalidate_row_norms();
        assert!(!ws.row_norms_ready);
    }

    #[test]
    fn uniform_strategy_allocates_no_row_norm_cache() {
        let ws = KaczmarzWorkspace::new(1_000_000, 5, KaczmarzSampling::Uniform);
        // Critical for streaming scale: no O(m) scratch.
        assert!(ws.row_norms_sq.is_empty());
        assert!(ws.cumulative_weights.is_empty());
    }

    #[test]
    fn rejects_all_zero_matrix_with_numerical_failure() {
        let a = DenseRowMajor {
            nrows: 3,
            ncols: 2,
            data: vec![0.0; 6],
        };
        let b = vec![1.0, 1.0, 1.0];
        let mut x = vec![0.0, 0.0];
        let mut ws = KaczmarzWorkspace::new(3, 2, KaczmarzSampling::SquaredRowNorm);
        let r = run(&KaczmarzConfig::default(), &a, &b, &mut x, &mut ws);
        assert_eq!(r.status, TerminationStatus::NumericalFailure);
    }

    #[test]
    fn default_sampling_is_squared_row_norm() {
        assert_eq!(KaczmarzSampling::default(), KaczmarzSampling::SquaredRowNorm);
    }

    #[test]
    fn default_config_uses_sensible_values() {
        let c = KaczmarzConfig::default();
        assert!(c.relaxation > 0.0 && c.relaxation < 2.0);
        assert!(c.residual_tol > 0.0);
        assert!(c.seed != 0);
        assert!(c.max_epochs > 0);
        assert!(c.check_residual_every_epochs >= 1);
    }

    #[test]
    #[should_panic]
    fn workspace_rejects_zero_m() {
        let _ = KaczmarzWorkspace::new(0, 3, KaczmarzSampling::Uniform);
    }

    #[test]
    #[should_panic]
    fn workspace_rejects_zero_n() {
        let _ = KaczmarzWorkspace::new(3, 0, KaczmarzSampling::Uniform);
    }

    #[test]
    #[should_panic(expected = "require m")]
    fn workspace_rejects_m_less_than_n() {
        let _ = KaczmarzWorkspace::new(2, 3, KaczmarzSampling::Uniform);
    }

    #[test]
    #[should_panic(expected = "residual_tol")]
    fn run_rejects_negative_residual_tol() {
        let a = identity_matrix(3);
        let b = vec![1.0, 2.0, 3.0];
        let mut x = vec![0.0; 3];
        let mut ws = KaczmarzWorkspace::new(3, 3, KaczmarzSampling::Uniform);
        let cfg = KaczmarzConfig {
            residual_tol: -1e-6,
            ..KaczmarzConfig::default()
        };
        let _ = run(&cfg, &a, &b, &mut x, &mut ws);
    }

    #[test]
    #[should_panic(expected = "step_tol")]
    fn run_rejects_non_finite_step_tol() {
        let a = identity_matrix(3);
        let b = vec![1.0, 2.0, 3.0];
        let mut x = vec![0.0; 3];
        let mut ws = KaczmarzWorkspace::new(3, 3, KaczmarzSampling::Uniform);
        let cfg = KaczmarzConfig {
            step_tol: f64::NAN,
            ..KaczmarzConfig::default()
        };
        let _ = run(&cfg, &a, &b, &mut x, &mut ws);
    }

    #[test]
    #[should_panic(expected = "max_epochs")]
    fn run_rejects_zero_max_epochs() {
        let a = identity_matrix(3);
        let b = vec![1.0, 2.0, 3.0];
        let mut x = vec![0.0; 3];
        let mut ws = KaczmarzWorkspace::new(3, 3, KaczmarzSampling::Uniform);
        let cfg = KaczmarzConfig {
            max_epochs: 0,
            ..KaczmarzConfig::default()
        };
        let _ = run(&cfg, &a, &b, &mut x, &mut ws);
    }

    #[test]
    #[should_panic(expected = "check_residual_every_epochs")]
    fn run_rejects_zero_check_cadence() {
        let a = identity_matrix(3);
        let b = vec![1.0, 2.0, 3.0];
        let mut x = vec![0.0; 3];
        let mut ws = KaczmarzWorkspace::new(3, 3, KaczmarzSampling::Uniform);
        let cfg = KaczmarzConfig {
            check_residual_every_epochs: 0,
            ..KaczmarzConfig::default()
        };
        let _ = run(&cfg, &a, &b, &mut x, &mut ws);
    }
}
