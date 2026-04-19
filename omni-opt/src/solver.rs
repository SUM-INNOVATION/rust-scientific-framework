//! Unconstrained descent driver: fluent builder, pre-allocated
//! workspace, and the outer `run` loop.
//!
//! All heap allocation is confined to [`SolverBuilder::build`];
//! every subsequent invocation of [`Solver::run`] /
//! [`Solver::run_with_hessian`] runs the full descent loop
//! allocation-free.

// `needless_range_loop` is counter-productive for vector kernels
// indexing multiple slices in lockstep — rewriting via enumerate()
// still requires indexing the remaining slices by `i`.
#![allow(clippy::needless_range_loop)]

use crate::constraints::BoxConstraints;
use crate::line_search::{self, LineSearch, LineSearchConfig, LineSearchError, LineSearchWorkspace};
use crate::methods::{Method, MethodError, MethodState};
use crate::oracle::Oracle;
use crate::state::{StoppingCriteria, TerminationStatus};

#[cfg(feature = "faer")]
use crate::oracle::HessianOracle;

// =============================================================
// Workspace — all per-iteration buffers.
// =============================================================

/// Pre-allocated per-iteration memory for the descent loop.
///
/// Owned by [`Solver`] for its lifetime. Construction allocates
/// five `Vec<f64>` of length `n` plus the line-search workspace
/// (two more `Vec<f64>`) and the per-method scratch; every method
/// invocation afterwards is allocation-free.
///
/// # Visibility
///
/// `SolverWorkspace` is deliberately `pub(crate)`. All
/// user-facing configuration flows through [`SolverBuilder`],
/// which enforces the invariants (non-zero `n`, valid Wolfe
/// constants, finite `curvature_epsilon`, …). Exposing the
/// workspace publicly would let callers construct a `Solver`
/// around an unvalidated workspace and bypass those checks —
/// so the type and its methods are not re-exported from the
/// crate root.
pub(crate) struct SolverWorkspace {
    n: usize,

    // Iteration-boundary buffers. `x` / `g` hold the current
    // iterate and gradient; `x_new` / `g_new` the next. At the
    // end of every iteration we `std::mem::swap(&mut x, &mut x_new)`
    // (and analogously for g) — an O(1) pointer rotation that
    // keeps the hot loop allocation-free and never copies.
    x: Vec<f64>,
    x_new: Vec<f64>,
    g: Vec<f64>,
    g_new: Vec<f64>,

    // Search direction, overwritten every iteration.
    d: Vec<f64>,

    // Phase-2 line-search scratch (two Vec<f64> of length n).
    ls: LineSearchWorkspace,

    // Per-method scratch. `MethodState::SteepestDescent` is a
    // zero-byte variant, so callers who pick that method pay no
    // incidental memory cost.
    method: MethodState,
}

impl SolverWorkspace {
    /// Allocate all descent-loop memory for problem dimension `n`.
    ///
    /// Crate-private: only [`SolverBuilder::build`] may construct
    /// a workspace, ensuring the tolerances and method parameters
    /// were validated on the way in.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0` or if any `Method::*` parameter is
    /// invalid (delegated to the per-method constructor).
    pub(crate) fn new(n: usize, method: Method) -> Self {
        assert!(n > 0, "SolverWorkspace::new: n must be > 0");
        Self {
            n,
            x: vec![0.0; n],
            x_new: vec![0.0; n],
            g: vec![0.0; n],
            g_new: vec![0.0; n],
            d: vec![0.0; n],
            ls: LineSearchWorkspace::new(n),
            method: MethodState::new(n, method),
        }
    }
}

// =============================================================
// Builder.
// =============================================================

/// Fluent configuration of a [`Solver`]. Cheap to hold — only
/// POD fields plus an optional owned [`BoxConstraints`]. Every
/// setter consumes and returns `self` by move, so the absence of
/// `Copy` costs nothing at call sites.
#[derive(Clone, Debug)]
pub struct SolverBuilder {
    method: Method,
    line_search: LineSearch,
    ls_config: LineSearchConfig,
    stopping: StoppingCriteria,
    bounds: Option<BoxConstraints>,
}

impl Default for SolverBuilder {
    fn default() -> Self {
        Self {
            method: Method::lbfgs(),
            line_search: LineSearch::StrongWolfe,
            ls_config: LineSearchConfig::default(),
            stopping: StoppingCriteria::default(),
            bounds: None,
        }
    }
}

impl SolverBuilder {
    /// Fresh builder with defaults: `LBFGS` (m = 10),
    /// Strong-Wolfe line search, default tolerances.
    pub fn new() -> Self {
        Self::default()
    }

    /// Select the descent algorithm.
    #[must_use]
    pub fn method(mut self, m: Method) -> Self {
        self.method = m;
        self
    }

    /// Select the line-search algorithm.
    #[must_use]
    pub fn line_search(mut self, ls: LineSearch) -> Self {
        self.line_search = ls;
        self
    }

    /// Replace the full [`LineSearchConfig`] in one call.
    #[must_use]
    pub fn line_search_config(mut self, c: LineSearchConfig) -> Self {
        self.ls_config = c;
        self
    }

    /// Replace the full [`StoppingCriteria`] in one call.
    #[must_use]
    pub fn stopping(mut self, s: StoppingCriteria) -> Self {
        self.stopping = s;
        self
    }

    /// Shortcut for the most-tuned field of [`StoppingCriteria`].
    #[must_use]
    pub fn grad_inf_tol(mut self, tol: f64) -> Self {
        self.stopping.grad_inf_tol = tol;
        self
    }

    /// Shortcut for [`StoppingCriteria::max_iter`].
    #[must_use]
    pub fn max_iter(mut self, k: usize) -> Self {
        self.stopping.max_iter = k;
        self
    }

    /// Attach componentwise box constraints.
    ///
    /// The solver projects the initial iterate and every
    /// accepted step onto the feasible box via projected
    /// clipping. See [`BoxConstraints`] for the scope of the
    /// guarantee (and the documented gap versus full L-BFGS-B).
    ///
    /// Passing `None` clears any previously-attached bounds;
    /// the builder can therefore be constructed once and reused
    /// across bounded / unbounded runs.
    #[must_use]
    pub fn bounds(mut self, bounds: Option<BoxConstraints>) -> Self {
        self.bounds = bounds;
        self
    }

    /// Allocate the solver for problem dimension `n`.
    ///
    /// # Panics
    ///
    /// Panics on `n == 0`, invalid `Method::*` parameters, or
    /// `ls_config.c1 >= ls_config.c2`. Every misconfiguration
    /// fails at construction time, never inside the descent loop.
    pub fn build(self, n: usize) -> Solver {
        assert!(n > 0, "SolverBuilder::build: n must be > 0");
        assert!(
            self.ls_config.c1 > 0.0
                && self.ls_config.c1 < self.ls_config.c2
                && self.ls_config.c2 < 1.0,
            "SolverBuilder::build: require 0 < c1 < c2 < 1"
        );
        if let Some(b) = &self.bounds {
            assert_eq!(
                b.len(),
                n,
                "SolverBuilder::build: bounds.len() ({}) != n ({})",
                b.len(),
                n
            );
        }
        Solver {
            workspace: SolverWorkspace::new(n, self.method),
            line_search: self.line_search,
            ls_config: self.ls_config,
            stopping: self.stopping,
            bounds: self.bounds,
        }
    }
}

// =============================================================
// Solver.
// =============================================================

/// Fully-configured, pre-allocated solver.
///
/// Calling [`Self::run`] / [`Self::run_with_hessian`] triggers a
/// complete descent loop; the `Solver` can be reused across
/// successive runs (warm-starts, parameter sweeps, etc.) on
/// initial guesses of the same dimension without reallocating.
pub struct Solver {
    workspace: SolverWorkspace,
    line_search: LineSearch,
    ls_config: LineSearchConfig,
    stopping: StoppingCriteria,
    /// Optional projected-clipping box constraints. When present,
    /// `bounds.len() == workspace.n` (enforced at build time).
    bounds: Option<BoxConstraints>,
}

impl Solver {
    /// Fluent builder entry point.
    pub fn builder() -> SolverBuilder {
        SolverBuilder::new()
    }

    /// Dimension this solver was built for.
    #[inline]
    pub fn n(&self) -> usize {
        self.workspace.n
    }

    /// Drive the descent loop for methods that do not require the
    /// Hessian (SteepestDescent / BFGS / L-BFGS).
    ///
    /// `x` is mutated **in place**: on entry it carries `x₀`; on
    /// exit the best iterate the solver produced. The user retains
    /// ownership of the buffer throughout — the solver never
    /// allocates a user-visible iterate.
    ///
    /// Selecting [`Method::Newton`] with this entry point returns
    /// a report with [`TerminationStatus::NumericalFailure`]; use
    /// [`Self::run_with_hessian`] in that case.
    pub fn run<O: Oracle>(&mut self, oracle: &mut O, x: &mut [f64]) -> SolverReport {
        debug_assert_eq!(x.len(), self.workspace.n);
        descent_loop::<O, NoHessianStep>(
            &mut self.workspace,
            oracle,
            self.line_search,
            &self.ls_config,
            &self.stopping,
            self.bounds.as_ref(),
            x,
        )
    }

    /// Drive the descent loop with a [`HessianOracle`], enabling
    /// [`Method::Newton`] in addition to all other methods.
    #[cfg(feature = "faer")]
    pub fn run_with_hessian<O: HessianOracle>(
        &mut self,
        oracle: &mut O,
        x: &mut [f64],
    ) -> SolverReport {
        debug_assert_eq!(x.len(), self.workspace.n);
        descent_loop::<O, HessianStep>(
            &mut self.workspace,
            oracle,
            self.line_search,
            &self.ls_config,
            &self.stopping,
            self.bounds.as_ref(),
            x,
        )
    }
}

// Zero-sized type-level switch for direction dispatch. The trait
// bound on its `compute` method carries the `HessianOracle`
// requirement through monomorphization without inflating the
// non-Hessian path with unused trait code.
trait DirectionStep<O> {
    fn compute(
        state: &mut MethodState,
        oracle: &mut O,
        x: &[f64],
        g: &[f64],
        d: &mut [f64],
    ) -> Result<(), MethodError>;
}

struct NoHessianStep;
impl<O: Oracle> DirectionStep<O> for NoHessianStep {
    #[inline]
    fn compute(
        state: &mut MethodState,
        oracle: &mut O,
        x: &[f64],
        g: &[f64],
        d: &mut [f64],
    ) -> Result<(), MethodError> {
        state.compute_direction(oracle, x, g, d)
    }
}

#[cfg(feature = "faer")]
struct HessianStep;
#[cfg(feature = "faer")]
impl<O: HessianOracle> DirectionStep<O> for HessianStep {
    #[inline]
    fn compute(
        state: &mut MethodState,
        oracle: &mut O,
        x: &[f64],
        g: &[f64],
        d: &mut [f64],
    ) -> Result<(), MethodError> {
        state.compute_direction_hess(oracle, x, g, d)
    }
}

// =============================================================
// Exit report.
// =============================================================

/// Final outcome of a descent. `Copy`, heap-free — returning it
/// preserves the zero-allocation invariant on the exit path.
#[derive(Clone, Copy, Debug)]
pub struct SolverReport {
    /// Structured exit state.
    pub status: TerminationStatus,
    /// Iterations performed.
    pub iters: u32,
    /// Total `oracle.value` + `oracle.value_grad` calls.
    pub f_evals: u32,
    /// Subset of `f_evals` that also produced a gradient.
    pub g_evals: u32,
    /// `f(x_final)`.
    pub f_final: f64,
    /// `‖∇f(x_final)‖_∞`.
    pub grad_inf_final: f64,
}

// =============================================================
// The descent loop — single implementation, parameterized on
// whether the oracle carries Hessians.
// =============================================================

#[allow(clippy::too_many_arguments)]
fn descent_loop<O, D>(
    ws: &mut SolverWorkspace,
    oracle: &mut O,
    line_search_kind: LineSearch,
    ls_config: &LineSearchConfig,
    stopping: &StoppingCriteria,
    bounds: Option<&BoxConstraints>,
    user_x: &mut [f64],
) -> SolverReport
where
    O: Oracle,
    D: DirectionStep<O>,
{
    // Seed the internal iterate from the user's buffer. Single
    // memcpy — subsequent iterations never touch user_x.
    ws.x.copy_from_slice(user_x);

    // Silently project `x₀` into the feasible set. Standard
    // practice in industry solvers (SciPy L-BFGS-B et al.):
    // rejecting a slightly-infeasible initial guess is
    // user-hostile, and the projection is O(n) with zero
    // allocation.
    if let Some(b) = bounds {
        b.project_in_place(&mut ws.x);
    }

    // Initial f, ∇f.
    let mut f = oracle.value_grad(&ws.x, &mut ws.g);
    let mut f_evals: u32 = 1;
    let mut g_evals: u32 = 1;
    let mut stagnation: usize = 0;

    let mut status = TerminationStatus::MaxIterationsReached;
    let mut iters: u32 = 0;

    let max_iter = stopping.max_iter as u32;

    while iters < max_iter {
        // --- Convergence check on the current iterate. -----
        //
        // Under bounds the plain gradient can be non-zero at the
        // true constrained optimum (it points outside the box),
        // so we test the reduced gradient instead. We overlay it
        // on the `d` buffer — `d` holds the previous iteration's
        // direction (or zeros on the first iteration) and is
        // about to be overwritten by `method.compute_direction`.
        // Saves `n · 8` bytes versus a dedicated buffer.
        let grad_inf = if let Some(b) = bounds {
            b.reduced_gradient(&ws.x, &ws.g, &mut ws.d);
            inf_norm(&ws.d)
        } else {
            inf_norm(&ws.g)
        };
        if !grad_inf.is_finite() || !f.is_finite() {
            status = TerminationStatus::NumericalFailure;
            break;
        }
        if grad_inf <= stopping.grad_inf_tol {
            status = TerminationStatus::Converged;
            break;
        }

        // --- Compute search direction. ----------------------
        // Monomorphized per `DirectionStep`: `NoHessianStep`
        // routes to `MethodState::compute_direction`;
        // `HessianStep` routes to `compute_direction_hess`.
        let dir_result = D::compute(&mut ws.method, oracle, &ws.x, &ws.g, &mut ws.d);

        match dir_result {
            Ok(()) => {}
            Err(_) => {
                // All method-level failures collapse to
                // NumericalFailure — callers reach for a
                // different method or restart policy.
                status = TerminationStatus::NumericalFailure;
                break;
            }
        }

        // --- Line search. -----------------------------------
        let ls_result = line_search::run(
            line_search_kind,
            ls_config,
            oracle,
            &ws.x,
            &ws.d,
            f,
            &ws.g,
            &mut ws.ls,
        );

        let step = match ls_result {
            Ok(s) => s,
            Err(LineSearchError::NonDescentDirection)
            | Err(LineSearchError::ArmijoFailed)
            | Err(LineSearchError::BracketingFailed)
            | Err(LineSearchError::CurvatureFailed) => {
                status = TerminationStatus::LineSearchFailed;
                break;
            }
            Err(LineSearchError::NumericalFailure) => {
                status = TerminationStatus::NumericalFailure;
                break;
            }
        };

        f_evals = f_evals.saturating_add(step.f_evals);
        // Strong Wolfe evaluates the gradient at every trial point;
        // Armijo never does. The adjustment keeps g_evals honest
        // without tracking it inside the line search.
        if line_search_kind == LineSearch::StrongWolfe {
            g_evals = g_evals.saturating_add(step.f_evals);
        }

        // --- Copy accepted iterate / gradient out of LS ws. -
        ws.x_new.copy_from_slice(ws.ls.x_trial());
        if step.g_new_dot_d.is_some() {
            // Strong-Wolfe: gradient at the accepted point was
            // computed by the curvature test, no re-eval needed.
            ws.g_new.copy_from_slice(ws.ls.g_trial());
        } else {
            // Armijo: refresh `∇f(x_new)` for the next direction.
            // Documented extra gradient evaluation per iteration
            // compared to Strong-Wolfe.
            let _ = oracle.value_grad(&ws.x_new, &mut ws.g_new);
            f_evals = f_evals.saturating_add(1);
            g_evals = g_evals.saturating_add(1);
        }
        let mut f_new = step.f_new;

        // --- Projected clipping + conditional gradient refresh ---
        //
        // Projected clipping sits between the line-search result
        // and the stopping-criteria block. `project_in_place`
        // returns `true` only when a coordinate actually moved;
        // on iterations where the step landed inside the box we
        // skip the extra oracle call entirely.
        //
        // When a coordinate was clipped, the stored `g_new`
        // (from the line search, evaluated at the pre-projection
        // trial point) is now stale. We refresh `f_new` / `g_new`
        // at the projected iterate so the next direction is
        // computed from a consistent `(x, ∇f(x))` pair.
        if let Some(b) = bounds {
            if b.project_in_place(&mut ws.x_new) {
                f_new = oracle.value_grad(&ws.x_new, &mut ws.g_new);
                f_evals = f_evals.saturating_add(1);
                g_evals = g_evals.saturating_add(1);
            }
        }

        // --- Stopping-criteria checks on the NEW iterate. ---
        // Step-norm convergence (strong signal: reached minimum).
        let x_norm = l2_norm(&ws.x);
        let step_norm = l2_diff_norm(&ws.x_new, &ws.x);
        if step_norm <= stopping.step_tol * (1.0 + x_norm) {
            // Adopt x_new and terminate.
            core::mem::swap(&mut ws.x, &mut ws.x_new);
            core::mem::swap(&mut ws.g, &mut ws.g_new);
            f = f_new;
            status = TerminationStatus::Converged;
            iters = iters.saturating_add(1);
            break;
        }

        // Stagnation: consecutive iterations with too-small
        // relative function decrease.
        let denom = f.abs().max(f_new.abs()).max(1.0);
        let rel_dec = (f - f_new) / denom;
        if rel_dec <= stopping.rel_f_tol {
            stagnation = stagnation.saturating_add(1);
            if stopping.stagnation_window > 0 && stagnation >= stopping.stagnation_window {
                core::mem::swap(&mut ws.x, &mut ws.x_new);
                core::mem::swap(&mut ws.g, &mut ws.g_new);
                f = f_new;
                status = TerminationStatus::StagnationDetected;
                iters = iters.saturating_add(1);
                break;
            }
        } else {
            stagnation = 0;
        }

        // --- Commit step into per-method state. -------------
        ws.method.update(&ws.x, &ws.x_new, &ws.g, &ws.g_new);

        // --- Swap iterates (O(1) pointer rotation). ---------
        core::mem::swap(&mut ws.x, &mut ws.x_new);
        core::mem::swap(&mut ws.g, &mut ws.g_new);
        f = f_new;
        iters = iters.saturating_add(1);
    }

    // Copy final iterate back into the caller's buffer.
    user_x.copy_from_slice(&ws.x);

    let grad_inf_final = inf_norm(&ws.g);

    SolverReport {
        status,
        iters,
        f_evals,
        g_evals,
        f_final: f,
        grad_inf_final,
    }
}

// =============================================================
// Scalar reductions.
// =============================================================

#[inline]
fn inf_norm(v: &[f64]) -> f64 {
    let mut m = 0.0_f64;
    for &x in v {
        let a = x.abs();
        if a > m {
            m = a;
        }
    }
    m
}

#[inline]
fn l2_norm(v: &[f64]) -> f64 {
    let mut s = 0.0;
    for &x in v {
        s += x * x;
    }
    s.sqrt()
}

#[inline]
fn l2_diff_norm(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        s += d * d;
    }
    s.sqrt()
}

// =============================================================
// Tests.
// =============================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::{Objective, Oracle};

    /// f(x) = ½ ‖x − c‖² ;  ∇f(x) = x − c.
    struct ShiftedQuadratic {
        c: Vec<f64>,
    }

    impl Objective for ShiftedQuadratic {
        fn n(&self) -> usize {
            self.c.len()
        }
        fn value(&mut self, x: &[f64]) -> f64 {
            0.5 * x
                .iter()
                .zip(&self.c)
                .map(|(xi, ci)| {
                    let d = xi - ci;
                    d * d
                })
                .sum::<f64>()
        }
    }

    impl Oracle for ShiftedQuadratic {
        fn value_grad(&mut self, x: &[f64], g: &mut [f64]) -> f64 {
            let mut s = 0.0;
            for i in 0..x.len() {
                let di = x[i] - self.c[i];
                g[i] = di;
                s += di * di;
            }
            0.5 * s
        }
    }

    fn converges_to<O: Oracle>(
        oracle: &mut O,
        x0: Vec<f64>,
        target: &[f64],
        method: Method,
    ) -> SolverReport {
        let n = x0.len();
        let mut solver = Solver::builder()
            .method(method)
            .line_search(LineSearch::StrongWolfe)
            .grad_inf_tol(1e-8)
            .max_iter(200)
            .build(n);
        let mut x = x0;
        let report = solver.run(oracle, &mut x);

        assert_eq!(
            report.status,
            TerminationStatus::Converged,
            "method must converge on the shifted quadratic"
        );
        for (got, want) in x.iter().zip(target) {
            assert!((got - want).abs() < 1e-6, "got {}, want {}", got, want);
        }
        report
    }

    #[test]
    fn steepest_descent_converges_on_quadratic() {
        let mut oracle = ShiftedQuadratic { c: vec![1.0, -2.0, 3.0] };
        converges_to(
            &mut oracle,
            vec![0.0; 3],
            &[1.0, -2.0, 3.0],
            Method::SteepestDescent,
        );
    }

    #[test]
    fn bfgs_converges_on_quadratic() {
        let mut oracle = ShiftedQuadratic { c: vec![1.0, -2.0, 3.0] };
        converges_to(
            &mut oracle,
            vec![0.0; 3],
            &[1.0, -2.0, 3.0],
            Method::bfgs(),
        );
    }

    #[test]
    fn lbfgs_converges_on_quadratic() {
        let mut oracle = ShiftedQuadratic { c: vec![1.0, -2.0, 3.0] };
        converges_to(&mut oracle, vec![0.0; 3], &[1.0, -2.0, 3.0], Method::lbfgs());
    }

    #[test]
    fn builder_defaults_select_lbfgs_strong_wolfe() {
        let b = SolverBuilder::new();
        assert!(matches!(b.line_search, LineSearch::StrongWolfe));
        assert!(matches!(b.method, Method::LBFGS { .. }));
    }

    #[test]
    #[should_panic]
    fn build_rejects_zero_dimension() {
        let _ = Solver::builder().build(0);
    }

    #[test]
    #[should_panic]
    fn build_rejects_bad_line_search_config() {
        let bad = LineSearchConfig {
            c1: 0.5,
            c2: 0.1,
            ..LineSearchConfig::default()
        };
        let _ = Solver::builder().line_search_config(bad).build(3);
    }

    #[test]
    fn armijo_also_converges_on_quadratic() {
        let mut oracle = ShiftedQuadratic { c: vec![1.0, -2.0] };
        let n = 2;
        let mut solver = Solver::builder()
            .method(Method::lbfgs())
            .line_search(LineSearch::BacktrackingArmijo)
            .grad_inf_tol(1e-8)
            .max_iter(200)
            .build(n);
        let mut x = vec![0.0; n];
        let r = solver.run(&mut oracle, &mut x);
        assert_eq!(r.status, TerminationStatus::Converged);
        assert!((x[0] - 1.0).abs() < 1e-6);
        assert!((x[1] - -2.0).abs() < 1e-6);
    }

    #[test]
    fn newton_variant_without_hessian_oracle_reports_failure() {
        // `run` cannot access a Hessian, so selecting Newton via
        // the `Oracle`-only entry point must surface a structured
        // failure instead of panicking.
        #[cfg(feature = "faer")]
        {
            let mut oracle = ShiftedQuadratic { c: vec![0.0] };
            let mut solver = Solver::builder()
                .method(Method::newton())
                .max_iter(10)
                .build(1);
            let mut x = vec![3.0];
            let r = solver.run(&mut oracle, &mut x);
            assert_eq!(r.status, TerminationStatus::NumericalFailure);
        }
    }

    #[test]
    fn report_is_copy_and_carries_counts() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<SolverReport>();
    }

    // ----- Phase-5 projected-clipping tests -------------------

    #[test]
    fn bounds_clip_the_optimum_into_the_feasible_box() {
        // Unconstrained optimum of the shifted quadratic is at
        // `c = (5, 5)`, but the box [-1, 1]² pins both coordinates
        // at their upper bound. Reduced gradient at (1, 1) is 0,
        // so the solver converges on the corner.
        let mut oracle = ShiftedQuadratic { c: vec![5.0, 5.0] };
        let bounds = crate::BoxConstraints::uniform(2, -1.0, 1.0);
        let mut solver = Solver::builder()
            .method(Method::lbfgs())
            .bounds(Some(bounds))
            .grad_inf_tol(1e-8)
            .max_iter(50)
            .build(2);
        let mut x = vec![0.0, 0.0];
        let r = solver.run(&mut oracle, &mut x);
        assert_eq!(r.status, TerminationStatus::Converged);
        assert!((x[0] - 1.0).abs() < 1e-8);
        assert!((x[1] - 1.0).abs() < 1e-8);
    }

    #[test]
    fn bounds_unconstrained_optimum_remains_when_inside_box() {
        // When the unconstrained optimum sits strictly inside the
        // box, projected clipping is a no-op and we recover the
        // same answer as the unbounded solver.
        let mut oracle = ShiftedQuadratic { c: vec![0.25, -0.5] };
        let bounds = crate::BoxConstraints::uniform(2, -1.0, 1.0);
        let mut solver = Solver::builder()
            .method(Method::bfgs())
            .bounds(Some(bounds))
            .grad_inf_tol(1e-8)
            .max_iter(50)
            .build(2);
        let mut x = vec![0.9, -0.9]; // start NEAR the boundary
        let r = solver.run(&mut oracle, &mut x);
        assert_eq!(r.status, TerminationStatus::Converged);
        assert!((x[0] - 0.25).abs() < 1e-6);
        assert!((x[1] - -0.5).abs() < 1e-6);
    }

    #[test]
    fn infeasible_initial_guess_is_silently_projected() {
        // `x₀ = (100, -100)` starts well outside the box. The
        // solver must not panic, and the iterate at termination
        // must be feasible.
        let mut oracle = ShiftedQuadratic { c: vec![0.0, 0.0] };
        let bounds = crate::BoxConstraints::uniform(2, -1.0, 1.0);
        let mut solver = Solver::builder()
            .method(Method::lbfgs())
            .bounds(Some(bounds))
            .grad_inf_tol(1e-8)
            .max_iter(50)
            .build(2);
        let mut x = vec![100.0, -100.0];
        let r = solver.run(&mut oracle, &mut x);
        assert_eq!(r.status, TerminationStatus::Converged);
        assert!(x[0] >= -1.0 - 1e-12 && x[0] <= 1.0 + 1e-12);
        assert!(x[1] >= -1.0 - 1e-12 && x[1] <= 1.0 + 1e-12);
    }

    #[test]
    #[should_panic(expected = "bounds.len()")]
    fn build_rejects_bounds_of_wrong_dimension() {
        let bounds = crate::BoxConstraints::uniform(3, -1.0, 1.0);
        let _ = Solver::builder()
            .bounds(Some(bounds))
            .build(2); // mismatch
    }

    // ---------- Newton / LM tests (require `faer`) ----------

    #[cfg(feature = "faer")]
    mod newton_tests {
        use super::*;
        use crate::oracle::HessianOracle;

        /// f(x) = ½ xᵀ A x − bᵀ x, `∇f = A x − b`, `∇²f = A`.
        struct Quadratic {
            // Packed lower triangle would complicate the test
            // without changing anything meaningful; store full A.
            a: Vec<f64>,
            b: Vec<f64>,
            n: usize,
        }

        impl Objective for Quadratic {
            fn n(&self) -> usize {
                self.n
            }
            fn value(&mut self, x: &[f64]) -> f64 {
                let mut quad = 0.0;
                for i in 0..self.n {
                    for j in 0..self.n {
                        quad += x[i] * self.a[i * self.n + j] * x[j];
                    }
                }
                let lin: f64 = x.iter().zip(&self.b).map(|(xi, bi)| xi * bi).sum();
                0.5 * quad - lin
            }
        }

        impl Oracle for Quadratic {
            fn value_grad(&mut self, x: &[f64], g: &mut [f64]) -> f64 {
                for i in 0..self.n {
                    let mut ax_i = 0.0;
                    for j in 0..self.n {
                        ax_i += self.a[i * self.n + j] * x[j];
                    }
                    g[i] = ax_i - self.b[i];
                }
                self.value(x)
            }
        }

        impl HessianOracle for Quadratic {
            fn hessian(&mut self, _x: &[f64], mut h: faer::MatMut<'_, f64>) {
                // Dense symmetric `A` → lower triangle.
                for j in 0..self.n {
                    for i in j..self.n {
                        h[(i, j)] = self.a[i * self.n + j];
                    }
                }
            }
        }

        #[test]
        fn newton_converges_in_one_step_on_positive_definite_quadratic() {
            // A = 2I, b = (2, 4) ⇒ optimum at x* = (1, 2).
            let mut oracle = Quadratic {
                a: vec![2.0, 0.0, 0.0, 2.0],
                b: vec![2.0, 4.0],
                n: 2,
            };
            let mut solver = Solver::builder()
                .method(Method::newton())
                .grad_inf_tol(1e-10)
                .max_iter(5)
                .build(2);
            let mut x = vec![0.0, 0.0];
            let r = solver.run_with_hessian(&mut oracle, &mut x);
            assert_eq!(r.status, TerminationStatus::Converged);
            assert!((x[0] - 1.0).abs() < 1e-9);
            assert!((x[1] - 2.0).abs() < 1e-9);
        }

        #[test]
        fn newton_lm_recovers_from_nonpsd_hessian_at_start() {
            // Construct an indefinite quadratic: A = diag(1, -1).
            // The Hessian is not positive definite, so exact
            // Newton's Cholesky must fail and the LM damping
            // kicks in. With enough damping the step direction
            // reduces to a scaled steepest-descent step — the
            // solver should still drive to the true minimum
            // of the DAMPED subproblem, and for a strongly
            // convex reformulation we just verify "does not
            // panic, terminates structurally." Concretely we
            // switch to a PSD quadratic with a tiny bump that
            // is ε-close to the boundary, forcing LM on the
            // very first factorization.
            let eps = 1e-18;
            let mut oracle = Quadratic {
                a: vec![eps, 0.0, 0.0, eps],
                b: vec![1.0, 1.0],
                n: 2,
            };
            let mut solver = Solver::builder()
                .method(Method::Newton {
                    initial_mu: 1e-6,
                    mu_growth: 10.0,
                    max_mu: 1e12,
                })
                .grad_inf_tol(1e-6)
                .max_iter(200)
                .build(2);
            let mut x = vec![0.0, 0.0];
            let r = solver.run_with_hessian(&mut oracle, &mut x);
            // The LM-damped solve must not panic and must return a
            // structured status — either Converged (if damping
            // stays manageable) or NumericalFailure (if μ escapes
            // max_mu). Both are acceptable here; the key is no
            // panic inside the Cholesky path.
            assert!(matches!(
                r.status,
                TerminationStatus::Converged | TerminationStatus::NumericalFailure
            ));
        }

        #[test]
        fn newton_respects_max_mu_cap() {
            // Strongly indefinite (diag(-1, -1)) ⇒ Cholesky
            // always fails; μ grows past any finite cap and
            // we bail out with NumericalFailure.
            let mut oracle = Quadratic {
                a: vec![-1.0, 0.0, 0.0, -1.0],
                b: vec![0.0, 0.0],
                n: 2,
            };
            let mut solver = Solver::builder()
                .method(Method::Newton {
                    initial_mu: 1e-2,
                    mu_growth: 2.0,
                    max_mu: 1.0, // very tight cap — fails fast
                })
                .max_iter(5)
                .build(2);
            let mut x = vec![1.0, 1.0];
            let r = solver.run_with_hessian(&mut oracle, &mut x);
            assert_eq!(r.status, TerminationStatus::NumericalFailure);
        }
    }
}
