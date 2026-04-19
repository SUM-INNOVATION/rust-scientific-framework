//! Line-search algorithms and scratch workspace.
//!
//! Every heap allocation occurs inside [`LineSearchWorkspace::new`]
//! (two `Vec<f64>` of length `n`). The hot path — [`run`] — performs
//! zero heap allocations: the trial iterate `x + α d` is written
//! into `workspace.x_trial`, and (for [`LineSearch::StrongWolfe`])
//! the trial gradient into `workspace.g_trial`; both are
//! overwritten in place at every candidate `α`.
//!
//! The Strong-Wolfe path follows Nocedal & Wright, *Numerical
//! Optimization* (2nd ed.), Algorithms 3.5 (bracketing) and 3.6
//! (zoom), with a safeguarded quadratic-interpolation step
//! selector. On exit, the returned step satisfies the Strong
//! Wolfe conditions (sufficient decrease and curvature) to the
//! tolerances configured in [`LineSearchConfig`].

use crate::oracle::Oracle;

/// Choice of line-search algorithm.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineSearch {
    /// Bracket + zoom Strong-Wolfe. Guarantees
    /// `f(x + α d) ≤ f(x) + c₁ α ∇f·d` and
    /// `|∇f(x + α d) · d| ≤ −c₂ ∇f(x)·d`.
    StrongWolfe,
    /// Armijo (sufficient-decrease) backtracking with geometric
    /// shrink. Cheaper — one oracle-value call per iteration, no
    /// gradient evaluations on trial points — but produces steps
    /// that satisfy only Armijo, not curvature.
    BacktrackingArmijo,
}

/// Tunables for the line search. All fields `Copy`.
///
/// Defaults are set for L-BFGS-style quasi-Newton use; nonlinear
/// CG callers should lower `c2` to `~0.1`.
#[derive(Clone, Copy, Debug)]
pub struct LineSearchConfig {
    /// Armijo sufficient-decrease constant. Must satisfy
    /// `0 < c1 < c2 < 1`. Typical: `1e-4`.
    pub c1: f64,

    /// Strong-Wolfe curvature constant. Typical: `0.9` for
    /// Newton / L-BFGS, `0.1` for nonlinear CG. Ignored by
    /// [`LineSearch::BacktrackingArmijo`].
    pub c2: f64,

    /// Initial trial step length `α₀`.
    pub initial_step: f64,

    /// Upper bound on `α` during bracketing. Caps runaway
    /// step-doubling when the objective is unbounded below
    /// along `d`.
    pub max_step: f64,

    /// Maximum number of bracket / zoom / backtrack iterations
    /// before returning the corresponding `*Failed` error.
    pub max_iters: u32,

    /// Geometric shrink factor for Armijo backtracking. Must
    /// satisfy `0 < shrink < 1`. Typical: `0.5`.
    pub shrink: f64,
}

impl Default for LineSearchConfig {
    fn default() -> Self {
        Self {
            c1: 1e-4,
            c2: 0.9,
            initial_step: 1.0,
            max_step: 1e20,
            max_iters: 20,
            shrink: 0.5,
        }
    }
}

/// Pre-allocated scratch for line searches.
///
/// Two `Vec<f64>` allocations happen once in [`new`]. The solver
/// holds a `LineSearchWorkspace` for the life of its run; every
/// call to [`run`] reuses the same buffers.
///
/// [`new`]: LineSearchWorkspace::new
pub struct LineSearchWorkspace {
    // Scratch for `x + α d` at every candidate step. Overwritten
    // in place per trial α, so no intermediate `Vec` is ever
    // materialized. On successful `run` exit, holds the accepted
    // iterate `x + α* d`.
    x_trial: Vec<f64>,

    // Scratch for ∇f(x_trial). Populated by Strong-Wolfe at every
    // trial α (the curvature test needs it). Left untouched by
    // Armijo, which never evaluates a gradient on trial points.
    // On successful StrongWolfe exit, holds ∇f(x_new) so the
    // solver can adopt it directly without re-invoking the oracle.
    g_trial: Vec<f64>,
}

impl LineSearchWorkspace {
    /// Allocate a fresh workspace.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0` — misconfiguration fails at build time,
    /// never inside the solver loop.
    pub fn new(n: usize) -> Self {
        assert!(n > 0, "LineSearchWorkspace::new: n must be > 0");
        Self {
            x_trial: vec![0.0; n],
            g_trial: vec![0.0; n],
        }
    }

    /// Dimensionality passed to [`Self::new`].
    #[inline]
    pub fn n(&self) -> usize {
        self.x_trial.len()
    }

    /// Accepted iterate `x + α* d` after a successful [`run`].
    /// Contents are undefined after a `Err(_)` return.
    pub fn x_trial(&self) -> &[f64] {
        &self.x_trial
    }

    /// `∇f(x + α* d)` after a successful [`LineSearch::StrongWolfe`]
    /// run. Undefined after [`LineSearch::BacktrackingArmijo`]
    /// (which does not evaluate the gradient at the accepted
    /// point) or after an `Err(_)` return.
    pub fn g_trial(&self) -> &[f64] {
        &self.g_trial
    }
}

/// Successful line-search outcome. `Copy`, no heap data.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LineSearchStep {
    /// Accepted step length `α*`.
    pub alpha: f64,

    /// `f(x + α* d)` — already computed, returned so the solver
    /// does not re-evaluate the oracle.
    pub f_new: f64,

    /// `∇f(x + α* d) · d`. `Some` after Strong-Wolfe (reused
    /// from the curvature test), `None` after Armijo.
    pub g_new_dot_d: Option<f64>,

    /// Number of inner iterations (bracket + zoom, or backtrack
    /// steps) the search consumed.
    pub iters: u32,

    /// Number of oracle evaluations consumed.
    pub f_evals: u32,
}

/// Structured line-search failure modes.
///
/// Deliberately separate from [`crate::TerminationStatus`] —
/// a line-search failure is an intermediate event the solver
/// may recover from (restart, shrink trust radius, fall back to
/// steepest descent) before escalating to
/// [`crate::TerminationStatus::LineSearchFailed`]. Keeping the
/// two enums distinct lets the solver own that decision.
///
/// `Copy`, non-allocating — the termination path stays heap-free.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LineSearchError {
    /// Armijo's sufficient-decrease condition could not be met
    /// within `cfg.max_iters` backtracks.
    ArmijoFailed,
    /// Strong-Wolfe bracketing phase exhausted `max_iters`
    /// without bracketing a valid step (or hit `max_step`).
    BracketingFailed,
    /// Zoom phase exhausted its iteration budget without
    /// satisfying the curvature condition.
    CurvatureFailed,
    /// `d` is not a descent direction: `∇f(x) · d ≥ 0`. Checked
    /// up-front before any oracle evaluation.
    NonDescentDirection,
    /// NaN / Inf observed in `f` or `∇f` during the search.
    NumericalFailure,
}

/// Execute a line search along direction `d` from iterate `x`.
///
/// # Preconditions
///
/// * `x.len() == d.len() == g_x.len() == workspace.n()`
/// * `f_x` is finite and equal to `oracle.value(x)`.
/// * `g_x` is `∇f(x)`.
///
/// Length mismatches are caught by `debug_assert_eq!`; release
/// builds stay branchless. Non-finite `f_x` or `g_x · d` returns
/// [`LineSearchError::NumericalFailure`] up-front.
///
/// # Zero-allocation contract
///
/// Every trial point `x + α d` is written into
/// `workspace.x_trial` in place; the associated gradient (when
/// needed) into `workspace.g_trial` in place. The oracle must
/// honor its own zero-alloc contract. On success, the accepted
/// iterate is in `workspace.x_trial`; for Strong-Wolfe, the
/// gradient at that point is in `workspace.g_trial`.
// Eight arguments is the natural signature — the line search
// needs the method, config, oracle, current iterate / direction /
// function value / gradient, and a scratch workspace. Bundling
// any subset into a helper struct is strictly worse for the call
// site (and we own the API).
#[allow(clippy::too_many_arguments)]
pub fn run<O: Oracle>(
    method: LineSearch,
    cfg: &LineSearchConfig,
    oracle: &mut O,
    x: &[f64],
    d: &[f64],
    f_x: f64,
    g_x: &[f64],
    workspace: &mut LineSearchWorkspace,
) -> Result<LineSearchStep, LineSearchError> {
    debug_assert_eq!(x.len(), d.len());
    debug_assert_eq!(x.len(), g_x.len());
    debug_assert_eq!(x.len(), workspace.n());

    if !f_x.is_finite() {
        return Err(LineSearchError::NumericalFailure);
    }

    let dphi0 = dot(g_x, d);
    if !dphi0.is_finite() {
        return Err(LineSearchError::NumericalFailure);
    }
    if dphi0 >= 0.0 {
        return Err(LineSearchError::NonDescentDirection);
    }

    match method {
        LineSearch::BacktrackingArmijo => {
            backtracking_armijo(cfg, oracle, x, d, f_x, dphi0, workspace)
        }
        LineSearch::StrongWolfe => strong_wolfe(cfg, oracle, x, d, f_x, dphi0, workspace),
    }
}

// ============================================================
// Internal helpers — all `#[inline]` and allocation-free.
// ============================================================

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

/// `x_trial ← x + α · d`, written in place.
#[inline]
fn set_trial(x_trial: &mut [f64], x: &[f64], alpha: f64, d: &[f64]) {
    debug_assert_eq!(x_trial.len(), x.len());
    debug_assert_eq!(x_trial.len(), d.len());
    for i in 0..x_trial.len() {
        x_trial[i] = x[i] + alpha * d[i];
    }
}

// ============================================================
// Backtracking Armijo.
// ============================================================

fn backtracking_armijo<O: Oracle>(
    cfg: &LineSearchConfig,
    oracle: &mut O,
    x: &[f64],
    d: &[f64],
    f0: f64,
    dphi0: f64,
    ws: &mut LineSearchWorkspace,
) -> Result<LineSearchStep, LineSearchError> {
    let mut alpha = cfg.initial_step;

    for iter in 0..cfg.max_iters {
        set_trial(&mut ws.x_trial, x, alpha, d);
        let f_trial = oracle.value(&ws.x_trial);

        if !f_trial.is_finite() {
            return Err(LineSearchError::NumericalFailure);
        }

        // Armijo: f(x + α d) ≤ f(x) + c₁ · α · ∇f(x)·d.
        // Exactly one oracle.value() call per iteration — so
        // `iters` and `f_evals` always agree here.
        if f_trial <= f0 + cfg.c1 * alpha * dphi0 {
            return Ok(LineSearchStep {
                alpha,
                f_new: f_trial,
                g_new_dot_d: None,
                iters: iter + 1,
                f_evals: iter + 1,
            });
        }

        alpha *= cfg.shrink;
    }

    Err(LineSearchError::ArmijoFailed)
}

// ============================================================
// Strong-Wolfe (N&W Alg. 3.5 + 3.6).
//
// Phase 1 (bracket): drive α upward from `initial_step`, evaluating
// φ(α) and φ'(α) until we either
//   (i)   satisfy Strong Wolfe directly,                → accept,
//   (ii)  violate sufficient decrease or φ increases,   → zoom(prev, α),
//   (iii) observe φ'(α) ≥ 0,                            → zoom(α, prev).
//
// Phase 2 (zoom): refine within the bracket using safeguarded
// quadratic interpolation of (α_lo, f_lo, φ'(α_lo), α_hi, f_hi),
// shrinking the bracket until Strong Wolfe is met.
// ============================================================

fn strong_wolfe<O: Oracle>(
    cfg: &LineSearchConfig,
    oracle: &mut O,
    x: &[f64],
    d: &[f64],
    f0: f64,
    dphi0: f64,
    ws: &mut LineSearchWorkspace,
) -> Result<LineSearchStep, LineSearchError> {
    let neg_c2_dphi0 = -cfg.c2 * dphi0; // > 0 since dphi0 < 0

    let mut alpha_prev = 0.0_f64;
    let mut f_prev = f0;
    let mut dphi_prev = dphi0;
    let mut alpha = cfg.initial_step.min(cfg.max_step);

    let mut iters = 0u32;
    let mut f_evals = 0u32;

    while iters < cfg.max_iters {
        iters += 1;

        set_trial(&mut ws.x_trial, x, alpha, d);
        let f_a = oracle.value_grad(&ws.x_trial, &mut ws.g_trial);
        f_evals += 1;

        if !f_a.is_finite() {
            return Err(LineSearchError::NumericalFailure);
        }

        // (ii) Armijo violation OR monotonicity failure → bracket found.
        if f_a > f0 + cfg.c1 * alpha * dphi0 || (iters > 1 && f_a >= f_prev) {
            return zoom(
                cfg,
                oracle,
                x,
                d,
                f0,
                dphi0,
                alpha_prev,
                f_prev,
                dphi_prev,
                alpha,
                f_a,
                ws,
                iters,
                f_evals,
            );
        }

        let dphi_a = dot(&ws.g_trial, d);
        if !dphi_a.is_finite() {
            return Err(LineSearchError::NumericalFailure);
        }

        // (i) Strong Wolfe curvature satisfied → accept.
        if dphi_a.abs() <= neg_c2_dphi0 {
            return Ok(LineSearchStep {
                alpha,
                f_new: f_a,
                g_new_dot_d: Some(dphi_a),
                iters,
                f_evals,
            });
        }

        // (iii) Slope turned non-negative → minimum is in
        // (alpha_prev, alpha). Bracket is reversed: (α, α_prev).
        if dphi_a >= 0.0 {
            return zoom(
                cfg,
                oracle,
                x,
                d,
                f0,
                dphi0,
                alpha,
                f_a,
                dphi_a,
                alpha_prev,
                f_prev,
                ws,
                iters,
                f_evals,
            );
        }

        // Still descending; extrapolate.
        alpha_prev = alpha;
        f_prev = f_a;
        dphi_prev = dphi_a;

        let next = (alpha * 2.0).min(cfg.max_step);
        if next <= alpha {
            // Hit max_step without bracketing — objective may be
            // unbounded below along `d` within the allowed range.
            return Err(LineSearchError::BracketingFailed);
        }
        alpha = next;
    }

    Err(LineSearchError::BracketingFailed)
}

#[allow(clippy::too_many_arguments)]
fn zoom<O: Oracle>(
    cfg: &LineSearchConfig,
    oracle: &mut O,
    x: &[f64],
    d: &[f64],
    f0: f64,
    dphi0: f64,
    mut alpha_lo: f64,
    mut f_lo: f64,
    mut dphi_lo: f64,
    mut alpha_hi: f64,
    mut f_hi: f64,
    ws: &mut LineSearchWorkspace,
    mut iters: u32,
    mut f_evals: u32,
) -> Result<LineSearchStep, LineSearchError> {
    let neg_c2_dphi0 = -cfg.c2 * dphi0;

    while iters < cfg.max_iters {
        iters += 1;

        let alpha_j = interpolate(alpha_lo, f_lo, dphi_lo, alpha_hi, f_hi);
        if !alpha_j.is_finite() {
            return Err(LineSearchError::NumericalFailure);
        }

        set_trial(&mut ws.x_trial, x, alpha_j, d);
        let f_j = oracle.value_grad(&ws.x_trial, &mut ws.g_trial);
        f_evals += 1;

        if !f_j.is_finite() {
            return Err(LineSearchError::NumericalFailure);
        }

        // Tighten upper bracket on Armijo violation / monotonicity.
        if f_j > f0 + cfg.c1 * alpha_j * dphi0 || f_j >= f_lo {
            alpha_hi = alpha_j;
            f_hi = f_j;
            continue;
        }

        let dphi_j = dot(&ws.g_trial, d);
        if !dphi_j.is_finite() {
            return Err(LineSearchError::NumericalFailure);
        }

        // Curvature satisfied → accept.
        if dphi_j.abs() <= neg_c2_dphi0 {
            return Ok(LineSearchStep {
                alpha: alpha_j,
                f_new: f_j,
                g_new_dot_d: Some(dphi_j),
                iters,
                f_evals,
            });
        }

        // Standard N&W bracket update: if new slope points away
        // from alpha_hi, swap the bracket endpoints.
        if dphi_j * (alpha_hi - alpha_lo) >= 0.0 {
            alpha_hi = alpha_lo;
            f_hi = f_lo;
        }
        alpha_lo = alpha_j;
        f_lo = f_j;
        dphi_lo = dphi_j;
    }

    Err(LineSearchError::CurvatureFailed)
}

/// Safeguarded quadratic interpolation.
///
/// Fits `q(α) = f_lo + φ'(α_lo)(α − α_lo) + A(α − α_lo)²` through
/// `(α_lo, f_lo, φ'(α_lo))` and `(α_hi, f_hi)`, and returns the
/// quadratic's minimizer if it lies inside the safeguard interval
/// `[lo + 0.1·w, hi − 0.1·w]` (where `lo`/`hi` are the sorted
/// bracket endpoints and `w` is its width). Otherwise falls back
/// to bisection.
#[inline]
fn interpolate(alpha_lo: f64, f_lo: f64, dphi_lo: f64, alpha_hi: f64, f_hi: f64) -> f64 {
    let (lo, hi) = if alpha_lo <= alpha_hi {
        (alpha_lo, alpha_hi)
    } else {
        (alpha_hi, alpha_lo)
    };
    let width = alpha_hi - alpha_lo;
    let mid = 0.5 * (lo + hi);

    if width.abs() < f64::EPSILON {
        return alpha_lo;
    }

    // q(α) coefficient of (α − α_lo)².
    let a = (f_hi - f_lo - dphi_lo * width) / (width * width);
    if a.abs() < f64::EPSILON {
        return mid;
    }

    let alpha_q = alpha_lo - dphi_lo / (2.0 * a);

    let sg = 0.1 * (hi - lo);
    if alpha_q >= lo + sg && alpha_q <= hi - sg {
        alpha_q
    } else {
        mid
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::{Objective, Oracle};

    /// f(x) = ½ ‖x − center‖² ;  ∇f(x) = x − center.
    struct ShiftedQuadratic {
        center: Vec<f64>,
    }

    impl Objective for ShiftedQuadratic {
        fn n(&self) -> usize {
            self.center.len()
        }
        fn value(&mut self, x: &[f64]) -> f64 {
            0.5 * x
                .iter()
                .zip(&self.center)
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
                let di = x[i] - self.center[i];
                g[i] = di;
                s += di * di;
            }
            0.5 * s
        }
    }

    // Always-descending objective for bracketing-failure tests.
    struct LinearDescent;
    impl Objective for LinearDescent {
        fn n(&self) -> usize {
            2
        }
        fn value(&mut self, x: &[f64]) -> f64 {
            -x[0] - x[1]
        }
    }
    impl Oracle for LinearDescent {
        fn value_grad(&mut self, x: &[f64], g: &mut [f64]) -> f64 {
            g[0] = -1.0;
            g[1] = -1.0;
            -x[0] - x[1]
        }
    }

    fn eval_at<O: Oracle>(oracle: &mut O, x: &[f64]) -> (f64, Vec<f64>) {
        let mut g = vec![0.0; x.len()];
        let f = oracle.value_grad(x, &mut g);
        (f, g)
    }

    #[test]
    fn non_descent_direction_rejected_before_any_eval() {
        let mut oracle = ShiftedQuadratic { center: vec![0.0, 0.0] };
        let x = vec![1.0, 0.0];
        let (f, g) = eval_at(&mut oracle, &x);
        let d = vec![1.0, 0.0]; // same sign as gradient → not descent
        let mut ws = LineSearchWorkspace::new(2);
        let cfg = LineSearchConfig::default();

        let r = run(LineSearch::StrongWolfe, &cfg, &mut oracle, &x, &d, f, &g, &mut ws);
        assert_eq!(r, Err(LineSearchError::NonDescentDirection));
    }

    #[test]
    fn strong_wolfe_accepts_unit_step_on_quadratic() {
        let mut oracle = ShiftedQuadratic { center: vec![0.0, 0.0] };
        let x = vec![1.0, 0.0];
        let (f, g) = eval_at(&mut oracle, &x);
        let d = vec![-1.0, 0.0]; // exact Newton direction
        let mut ws = LineSearchWorkspace::new(2);
        let cfg = LineSearchConfig::default();

        let step = run(LineSearch::StrongWolfe, &cfg, &mut oracle, &x, &d, f, &g, &mut ws)
            .expect("strong-wolfe must succeed on a convex quadratic");

        assert!((step.alpha - 1.0).abs() < 1e-12);
        assert!(step.f_new.abs() < 1e-12);
        assert!(step.g_new_dot_d.is_some());
        assert!(step.g_new_dot_d.unwrap().abs() < 1e-12);

        // x_trial and g_trial hold the accepted iterate / gradient.
        assert!((ws.x_trial()[0]).abs() < 1e-12);
        assert!((ws.x_trial()[1]).abs() < 1e-12);
        assert!((ws.g_trial()[0]).abs() < 1e-12);
    }

    #[test]
    fn strong_wolfe_zoom_recovers_from_overshoot() {
        // Start at x = 1, direction = -1 (Newton step is α = 1).
        // Initial step = 2 → α = 2 overshoots (f unchanged but slope
        // flipped positive), forcing the bracket/zoom path.
        let mut oracle = ShiftedQuadratic { center: vec![0.0] };
        let x = vec![1.0];
        let (f, g) = eval_at(&mut oracle, &x);
        let d = vec![-1.0];
        let mut ws = LineSearchWorkspace::new(1);
        let cfg = LineSearchConfig {
            initial_step: 2.0,
            ..LineSearchConfig::default()
        };

        let step = run(LineSearch::StrongWolfe, &cfg, &mut oracle, &x, &d, f, &g, &mut ws)
            .expect("zoom must recover from a deliberate overshoot");

        assert!((step.alpha - 1.0).abs() < 1e-6);
        assert!(step.iters >= 2, "zoom should have been invoked");
    }

    #[test]
    fn armijo_accepts_unit_step_on_quadratic() {
        let mut oracle = ShiftedQuadratic { center: vec![0.0, 0.0] };
        let x = vec![1.0, 0.0];
        let (f, g) = eval_at(&mut oracle, &x);
        let d = vec![-1.0, 0.0];
        let mut ws = LineSearchWorkspace::new(2);
        let cfg = LineSearchConfig::default();

        let step = run(
            LineSearch::BacktrackingArmijo,
            &cfg,
            &mut oracle,
            &x,
            &d,
            f,
            &g,
            &mut ws,
        )
        .expect("armijo must succeed on a convex quadratic");

        assert!((step.alpha - 1.0).abs() < 1e-12);
        assert_eq!(step.g_new_dot_d, None);
    }

    #[test]
    fn armijo_backtracks_when_initial_step_too_large() {
        // Same quadratic, initial_step = 4 → first trial fails
        // Armijo (f_trial > f0 by a wide margin), backtracking
        // shrinks to an acceptable step.
        let mut oracle = ShiftedQuadratic { center: vec![0.0] };
        let x = vec![1.0];
        let (f, g) = eval_at(&mut oracle, &x);
        let d = vec![-1.0];
        let mut ws = LineSearchWorkspace::new(1);
        let cfg = LineSearchConfig {
            initial_step: 4.0,
            ..LineSearchConfig::default()
        };

        let step = run(
            LineSearch::BacktrackingArmijo,
            &cfg,
            &mut oracle,
            &x,
            &d,
            f,
            &g,
            &mut ws,
        )
        .expect("armijo must eventually accept a step after backtracking");

        assert!(step.alpha <= 1.0 + 1e-12);
        assert!(step.iters >= 2);
    }

    #[test]
    fn bracketing_failure_returns_structured_error() {
        // Linear descent objective; no α satisfies the curvature
        // condition, and eventually α hits max_step without
        // triggering zoom.
        let mut oracle = LinearDescent;
        let x = vec![0.0, 0.0];
        let (f, g) = eval_at(&mut oracle, &x);
        let d = vec![1.0, 1.0];
        let mut ws = LineSearchWorkspace::new(2);
        let cfg = LineSearchConfig {
            initial_step: 1.0,
            max_step: 4.0,
            max_iters: 10,
            ..LineSearchConfig::default()
        };

        let r = run(LineSearch::StrongWolfe, &cfg, &mut oracle, &x, &d, f, &g, &mut ws);
        assert_eq!(r, Err(LineSearchError::BracketingFailed));
    }

    #[test]
    fn nan_f_x_returns_numerical_failure() {
        let mut oracle = ShiftedQuadratic { center: vec![0.0] };
        let x = vec![1.0];
        let g = vec![1.0];
        let d = vec![-1.0];
        let mut ws = LineSearchWorkspace::new(1);
        let cfg = LineSearchConfig::default();

        let r = run(
            LineSearch::StrongWolfe,
            &cfg,
            &mut oracle,
            &x,
            &d,
            f64::NAN,
            &g,
            &mut ws,
        );
        assert_eq!(r, Err(LineSearchError::NumericalFailure));
    }

    #[test]
    #[should_panic]
    fn workspace_rejects_zero_dimension() {
        let _ = LineSearchWorkspace::new(0);
    }

    #[test]
    fn default_config_uses_20_max_iters() {
        let c = LineSearchConfig::default();
        assert_eq!(c.max_iters, 20);
        assert!(c.c1 < c.c2);
        assert!(c.c2 < 1.0);
        assert!(c.shrink > 0.0 && c.shrink < 1.0);
    }
}
