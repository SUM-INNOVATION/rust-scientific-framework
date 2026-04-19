//! Descent algorithms and per-method memory.
//!
//! User-facing selection is the `Copy` [`Method`] enum. The
//! solver owns a [`MethodState`] that carries the per-algorithm
//! scratch; allocation happens exactly once in `MethodState::new`
//! and every subsequent call to [`MethodState::compute_direction`]
//! / [`MethodState::update`] is heap-free.

// Crate-level clippy escape hatches for this module:
//   - BFGS/LBFGS are canonical industry acronyms; capitalizing
//     them in `Method::BFGS` / `Method::LBFGS` matches every
//     textbook and published paper.
//   - `needless_range_loop` is counter-productive in numerical
//     kernels that index multiple slices in lockstep; the
//     `enumerate()` transform would still have to index the
//     other slices by `i`.
#![allow(clippy::upper_case_acronyms, clippy::needless_range_loop)]

use crate::oracle::Oracle;
use crate::workspace::LBFGSWorkspace;

#[cfg(feature = "faer")]
use crate::oracle::HessianOracle;

/// User-facing selection of the descent algorithm.
///
/// Algorithm parameters are carried in-variant so the builder API
/// reads top-to-bottom without an associated config table.
#[derive(Clone, Copy, Debug)]
pub enum Method {
    /// `d = −∇f(x)`. Pipeline-validation baseline — zero
    /// per-method state, zero per-iteration arithmetic beyond the
    /// negation.
    SteepestDescent,

    /// Exact Newton with Levenberg–Marquardt damping fallback.
    /// Requires the `faer` feature and a [`HessianOracle`].
    ///
    /// The solver tries `(∇²f) d = −∇f` via Cholesky; on a
    /// non-positive-definite failure it re-factors
    /// `(∇²f + μ I)` with monotonically increasing `μ`.
    #[cfg(feature = "faer")]
    Newton {
        /// Initial damping probed on the first Cholesky failure.
        /// `0.0` asks the solver to pick a conservative default.
        initial_mu: f64,
        /// Multiplicative growth factor applied to `μ` after each
        /// failed Cholesky. Typical: `10.0`.
        mu_growth: f64,
        /// Hard cap on `μ`; exceeded ⇒ direction step reports
        /// [`MethodError::HessianNotFactorable`].
        max_mu: f64,
    },

    /// Dense BFGS with inverse-Hessian approximation `Hₖ`.
    /// Strict curvature safeguard: skip the rank-2 update if
    /// `sᵀy ≤ curvature_epsilon`.
    BFGS {
        /// Threshold `ε` on `sᵀy`. `0.0` is legal (skip only when
        /// non-positive); positive values reject tiny-curvature
        /// pairs that would destabilize `H`.
        curvature_epsilon: f64,
    },

    /// Limited-memory BFGS driving the two-loop recursion over
    /// the Phase-1 [`LBFGSWorkspace`] ring.
    LBFGS {
        /// History depth; fixed at build time, seeded into
        /// `LBFGSWorkspace::new`.
        m: usize,
        /// Curvature safeguard — skip ring push when
        /// `sᵀy ≤ curvature_epsilon`.
        curvature_epsilon: f64,
    },
}

impl Method {
    /// `SteepestDescent`.
    pub fn steepest_descent() -> Self {
        Self::SteepestDescent
    }

    /// `BFGS { curvature_epsilon = 1e-10 }`.
    pub fn bfgs() -> Self {
        Self::BFGS {
            curvature_epsilon: 1e-10,
        }
    }

    /// `LBFGS { m = DEFAULT_M, curvature_epsilon = 1e-10 }`.
    pub fn lbfgs() -> Self {
        Self::LBFGS {
            m: LBFGSWorkspace::DEFAULT_M,
            curvature_epsilon: 1e-10,
        }
    }

    /// `Newton { initial_mu = 1e-4, mu_growth = 10.0, max_mu = 1e12 }`.
    #[cfg(feature = "faer")]
    pub fn newton() -> Self {
        Self::Newton {
            initial_mu: 1e-4,
            mu_growth: 10.0,
            max_mu: 1e12,
        }
    }
}

/// Internal failure path for direction computation. Mapped to
/// [`crate::TerminationStatus`] at the solver boundary.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MethodError {
    /// `Newton`: `μ` exceeded `max_mu` before Cholesky succeeded.
    HessianNotFactorable,
    /// Produced direction contains NaN or `±∞`.
    NumericalFailure,
    /// [`Method::Newton`] was selected but the solver was driven
    /// through [`crate::Solver::run`] (which only has an [`Oracle`]).
    /// Use [`crate::Solver::run_with_hessian`] instead.
    #[cfg(feature = "faer")]
    HessianOracleRequired,
}

/// Internal per-method workspace. Not public — callers only name
/// the `Method` enum and the builder constructs the matching
/// state variant for them.
pub(crate) enum MethodState {
    SteepestDescent,
    BFGS(BFGSState),
    LBFGS(LBFGSState),
    #[cfg(feature = "faer")]
    Newton(NewtonState),
}

impl MethodState {
    /// Build the per-method scratch for problem dimension `n`.
    ///
    /// All heap allocation for the method occurs here.
    pub(crate) fn new(n: usize, method: Method) -> Self {
        match method {
            Method::SteepestDescent => Self::SteepestDescent,
            Method::BFGS { curvature_epsilon } => {
                Self::BFGS(BFGSState::new(n, curvature_epsilon))
            }
            Method::LBFGS {
                m,
                curvature_epsilon,
            } => Self::LBFGS(LBFGSState::new(n, m, curvature_epsilon)),
            #[cfg(feature = "faer")]
            Method::Newton {
                initial_mu,
                mu_growth,
                max_mu,
            } => Self::Newton(NewtonState::new(n, initial_mu, mu_growth, max_mu)),
        }
    }

    /// Compute the search direction for methods that do not need
    /// the Hessian (SteepestDescent / BFGS / L-BFGS).
    ///
    /// Returns [`MethodError::HessianOracleRequired`] if the
    /// user selected [`Method::Newton`] — the solver escalates
    /// that into a hard failure. Callers with a `HessianOracle`
    /// should instead route through [`Self::compute_direction_hess`].
    pub(crate) fn compute_direction<O: Oracle>(
        &mut self,
        _oracle: &mut O,
        _x: &[f64],
        g: &[f64],
        d: &mut [f64],
    ) -> Result<(), MethodError> {
        match self {
            Self::SteepestDescent => {
                steepest_direction(g, d);
                Ok(())
            }
            Self::BFGS(s) => s.compute_direction(g, d),
            Self::LBFGS(s) => s.compute_direction(g, d),
            #[cfg(feature = "faer")]
            Self::Newton(_) => Err(MethodError::HessianOracleRequired),
        }
    }

    /// Hessian-aware direction dispatch. Identical to
    /// [`Self::compute_direction`] for non-Newton methods — the
    /// Newton arm uses the oracle's Hessian plus the solver's
    /// pre-allocated factorization scratch.
    #[cfg(feature = "faer")]
    pub(crate) fn compute_direction_hess<O: HessianOracle>(
        &mut self,
        oracle: &mut O,
        x: &[f64],
        g: &[f64],
        d: &mut [f64],
    ) -> Result<(), MethodError> {
        match self {
            Self::SteepestDescent => {
                steepest_direction(g, d);
                Ok(())
            }
            Self::BFGS(s) => s.compute_direction(g, d),
            Self::LBFGS(s) => s.compute_direction(g, d),
            Self::Newton(s) => s.compute_direction(oracle, x, g, d),
        }
    }

    /// Commit the step `(x → x_new, g → g_new)` into per-method
    /// state: rank-2 update for BFGS, ring push for L-BFGS,
    /// no-op for SteepestDescent / Newton.
    ///
    /// `sᵀy` is computed in one pass from the live buffers; no
    /// dedicated `s` / `y` scratch is materialized on the
    /// L-BFGS path (BFGS keeps them for the mat-vec).
    pub(crate) fn update(
        &mut self,
        x: &[f64],
        x_new: &[f64],
        g: &[f64],
        g_new: &[f64],
    ) {
        match self {
            Self::SteepestDescent => {}
            Self::BFGS(s) => s.update(x, x_new, g, g_new),
            Self::LBFGS(s) => s.update(x, x_new, g, g_new),
            #[cfg(feature = "faer")]
            Self::Newton(_) => {}
        }
    }
}

// =============================================================
// Steepest descent.
// =============================================================

#[inline]
fn steepest_direction(g: &[f64], d: &mut [f64]) {
    debug_assert_eq!(g.len(), d.len());
    for i in 0..g.len() {
        d[i] = -g[i];
    }
}

// =============================================================
// Dense BFGS.
// =============================================================

pub(crate) struct BFGSState {
    n: usize,
    // Inverse-Hessian approximation `H`, n·n row-major.
    h: Vec<f64>,
    // Scratch for s = x_new − x. Kept in-state so the rank-2
    // update can read it after computing `sᵀy` and `Hy`.
    s: Vec<f64>,
    // Scratch for y = g_new − g.
    y: Vec<f64>,
    // Scratch for `H y`, consumed once per rank-2 update.
    hy: Vec<f64>,
    // `true` after the first successful update. Before that we
    // apply the standard `γ I` scaling (γ = sᵀy / yᵀy) to the
    // identity.
    first_update_done: bool,
    curvature_epsilon: f64,
}

impl BFGSState {
    fn new(n: usize, curvature_epsilon: f64) -> Self {
        assert!(n > 0, "BFGSState::new: n must be > 0");
        // Reject NaN / ±∞ / negative ε at construction — a bad
        // threshold here would silently poison every downstream
        // rank-2 update decision.
        assert!(
            curvature_epsilon.is_finite() && curvature_epsilon >= 0.0,
            "BFGSState::new: curvature_epsilon must be finite and non-negative (got {})",
            curvature_epsilon
        );
        let mut h = vec![0.0; n * n];
        // Initialise to identity.
        for i in 0..n {
            h[i * n + i] = 1.0;
        }
        Self {
            n,
            h,
            s: vec![0.0; n],
            y: vec![0.0; n],
            hy: vec![0.0; n],
            first_update_done: false,
            curvature_epsilon,
        }
    }

    /// `d = −H g` via a single dense mat-vec, in place.
    fn compute_direction(&self, g: &[f64], d: &mut [f64]) -> Result<(), MethodError> {
        debug_assert_eq!(g.len(), self.n);
        debug_assert_eq!(d.len(), self.n);

        let n = self.n;
        for i in 0..n {
            let mut acc = 0.0;
            let row = &self.h[i * n..(i + 1) * n];
            for j in 0..n {
                acc += row[j] * g[j];
            }
            d[i] = -acc;
            if !d[i].is_finite() {
                return Err(MethodError::NumericalFailure);
            }
        }
        Ok(())
    }

    fn update(&mut self, x: &[f64], x_new: &[f64], g: &[f64], g_new: &[f64]) {
        let n = self.n;
        debug_assert_eq!(x.len(), n);
        debug_assert_eq!(x_new.len(), n);
        debug_assert_eq!(g.len(), n);
        debug_assert_eq!(g_new.len(), n);

        // Form s, y and s·y in a single pass.
        let mut ys = 0.0;
        let mut yy = 0.0;
        for i in 0..n {
            let si = x_new[i] - x[i];
            let yi = g_new[i] - g[i];
            self.s[i] = si;
            self.y[i] = yi;
            ys += si * yi;
            yy += yi * yi;
        }

        // Curvature safeguard — reject bad pairs outright.
        // Written as explicit conjunction (`> ε` AND finite) so a
        // NaN `ys` takes the early-return branch. Clippy's
        // `neg_cmp_op_on_partial_ord` is what steered us here.
        if !(ys.is_finite() && ys > self.curvature_epsilon) {
            return;
        }

        // On the very first accepted pair, rescale H from
        // identity to `γ I` where γ = sᵀy / yᵀy. Classical
        // BFGS warm-start; keeps the first direction in the
        // right ballpark on badly-scaled problems.
        if !self.first_update_done && yy > 0.0 {
            let gamma = ys / yy;
            for i in 0..n {
                for j in 0..n {
                    self.h[i * n + j] = if i == j { gamma } else { 0.0 };
                }
            }
        }

        // hy = H y, O(n²).
        for i in 0..n {
            let mut acc = 0.0;
            let row = &self.h[i * n..(i + 1) * n];
            for j in 0..n {
                acc += row[j] * self.y[j];
            }
            self.hy[i] = acc;
        }

        // y·H·y scalar.
        let mut yhy = 0.0;
        for i in 0..n {
            yhy += self.y[i] * self.hy[i];
        }

        let rho = 1.0 / ys;
        let coeff = rho * (1.0 + rho * yhy);
        //
        //   H ← H − ρ (s·hyᵀ + hy·sᵀ) + coeff · s·sᵀ
        //
        // Single O(n²) pass over the dense buffer, no further
        // scratch required.
        for i in 0..n {
            let si = self.s[i];
            let hyi = self.hy[i];
            let row = &mut self.h[i * n..(i + 1) * n];
            for j in 0..n {
                let sj = self.s[j];
                let hyj = self.hy[j];
                row[j] += coeff * si * sj - rho * (si * hyj + hyi * sj);
            }
        }

        self.first_update_done = true;
    }
}

// =============================================================
// L-BFGS — wraps the Phase 1 ring buffer verbatim.
// =============================================================

pub(crate) struct LBFGSState {
    ring: LBFGSWorkspace,
    curvature_epsilon: f64,
}

impl LBFGSState {
    fn new(n: usize, m: usize, curvature_epsilon: f64) -> Self {
        // Same safeguard as `BFGSState::new` — the ring-push
        // decision on every successful step hinges on this
        // threshold, so a non-finite or negative value is treated
        // as misconfiguration and rejected loudly at build time.
        assert!(
            curvature_epsilon.is_finite() && curvature_epsilon >= 0.0,
            "LBFGSState::new: curvature_epsilon must be finite and non-negative (got {})",
            curvature_epsilon
        );
        Self {
            ring: LBFGSWorkspace::new(n, m),
            curvature_epsilon,
        }
    }

    /// Two-loop recursion over the Phase-1 ring.
    ///
    /// `d` doubles as the working vector (`q` in the first loop,
    /// `r` in the second). The ring's own `alpha` buffer is the
    /// only scratch used — no per-iteration allocation.
    fn compute_direction(&mut self, g: &[f64], d: &mut [f64]) -> Result<(), MethodError> {
        debug_assert_eq!(g.len(), d.len());
        debug_assert_eq!(d.len(), self.ring.n());

        let n = self.ring.n();
        let count = self.ring.count();

        // q ← g.
        d.copy_from_slice(g);

        // First pass (newest → oldest).
        // `alpha` is the ring's own scratch buffer. We read and
        // write it through disjoint indices per iteration, so
        // borrowing the slice once is safe and alloc-free.
        //
        // Borrow-checker choreography: we cannot hold `alpha_mut`
        // and call ring.s_slot(k) / ring.y_slot(k) simultaneously
        // (both borrow `ring`). Instead we compute the alphas in
        // one loop that only reads the ring, storing into a local
        // 1-element variable per k and immediately writing back
        // through `alpha_mut()`.
        for k in (0..count).rev() {
            let s_k = self.ring.s_slot(k);
            let y_k = self.ring.y_slot(k);
            let rho_k = self.ring.rho_at(k);
            let alpha_k = rho_k * dot(s_k, d);
            // Update d before stashing alpha — order matters only
            // for borrow sequencing, not for correctness here.
            for i in 0..n {
                d[i] -= alpha_k * y_k[i];
            }
            self.ring.alpha_mut()[k] = alpha_k;
        }

        // H₀ scaling: γ = (sᵀy) / (yᵀy) on the newest pair.
        // If no history yet, leave d untouched — caller applied
        // the identity `H₀ = I` implicitly.
        if count > 0 {
            let newest = count - 1;
            let s_new = self.ring.s_slot(newest);
            let y_new = self.ring.y_slot(newest);
            let sy = dot(s_new, y_new);
            let yy = dot(y_new, y_new);
            if yy > 0.0 {
                let gamma = sy / yy;
                for i in 0..n {
                    d[i] *= gamma;
                }
            }
        }

        // Second pass (oldest → newest).
        //
        // Borrow sequencing: we must read `alpha[k]` (mutable
        // borrow of the ring) before taking the immutable borrows
        // on `s_slot(k)` / `y_slot(k)` — otherwise NLL keeps the
        // immutable borrows live and blocks the `alpha_mut()`
        // call. `alpha_k` and `rho_k` are plain `f64`s, so lifting
        // them out is free.
        for k in 0..count {
            let alpha_k = self.ring.alpha_mut()[k];
            let rho_k = self.ring.rho_at(k);
            let s_k = self.ring.s_slot(k);
            let y_k = self.ring.y_slot(k);
            let beta = rho_k * dot(y_k, d);
            let coeff = alpha_k - beta;
            for i in 0..n {
                d[i] += coeff * s_k[i];
            }
        }

        // d ← -r (the search direction).
        for i in 0..n {
            d[i] = -d[i];
            if !d[i].is_finite() {
                return Err(MethodError::NumericalFailure);
            }
        }
        Ok(())
    }

    fn update(&mut self, x: &[f64], x_new: &[f64], g: &[f64], g_new: &[f64]) {
        // Compute sᵀy from the live buffers — no dedicated
        // s / y scratch is needed on the L-BFGS path.
        let n = x.len();
        let mut ys = 0.0;
        for i in 0..n {
            ys += (x_new[i] - x[i]) * (g_new[i] - g[i]);
        }

        if !(ys.is_finite() && ys > self.curvature_epsilon) {
            // Silently drop — Phase-1 write/commit split means
            // the ring head is never advanced, so no half-written
            // pair survives.
            return;
        }

        // Curvature OK — write the new pair into the ring head.
        {
            let s = self.ring.head_s_mut();
            for i in 0..n {
                s[i] = x_new[i] - x[i];
            }
        }
        {
            let y = self.ring.head_y_mut();
            for i in 0..n {
                y[i] = g_new[i] - g[i];
            }
        }
        self.ring.advance(1.0 / ys);
    }

    #[cfg(test)]
    pub(crate) fn ring(&self) -> &LBFGSWorkspace {
        &self.ring
    }
}

// =============================================================
// Newton (feature-gated).
// =============================================================

#[cfg(feature = "faer")]
pub(crate) struct NewtonState {
    n: usize,
    // Raw Hessian as delivered by the oracle. Preserved across
    // retries so we can add different μ·I perturbations without
    // asking the oracle to re-evaluate.
    h: faer::Mat<f64>,
    // Factorization workspace. `h` is copied here, `μ I` added,
    // then Cholesky is performed in place.
    h_damped: faer::Mat<f64>,
    // n×1 RHS column; overwritten by `solve_in_place` with `d`.
    rhs: faer::Mat<f64>,
    // Pre-sized scratch for faer's in-place factorization.
    mem_buf: faer::dyn_stack::MemBuffer,
    initial_mu: f64,
    mu_growth: f64,
    max_mu: f64,
}

#[cfg(feature = "faer")]
impl NewtonState {
    fn new(n: usize, initial_mu: f64, mu_growth: f64, max_mu: f64) -> Self {
        assert!(n > 0, "NewtonState::new: n must be > 0");
        assert!(
            initial_mu >= 0.0 && mu_growth > 1.0 && max_mu.is_finite(),
            "NewtonState::new: invalid LM parameters"
        );

        use faer::linalg::cholesky::llt::factor::cholesky_in_place_scratch;
        use faer::Par;

        // Pre-size faer's scratch once. Factor is strictly larger
        // than solve for LLT, so sizing for factor covers both.
        let stack_req = cholesky_in_place_scratch::<f64>(n, Par::Seq, Default::default());
        let mem_buf = faer::dyn_stack::MemBuffer::new(stack_req);

        Self {
            n,
            h: faer::Mat::zeros(n, n),
            h_damped: faer::Mat::zeros(n, n),
            rhs: faer::Mat::zeros(n, 1),
            mem_buf,
            initial_mu,
            mu_growth,
            max_mu,
        }
    }

    fn compute_direction<O: HessianOracle>(
        &mut self,
        oracle: &mut O,
        x: &[f64],
        g: &[f64],
        d: &mut [f64],
    ) -> Result<(), MethodError> {
        use faer::linalg::cholesky::llt::factor::{cholesky_in_place, LltRegularization};
        use faer::linalg::cholesky::llt::solve::solve_in_place;
        use faer::{dyn_stack::MemStack, Par};

        let n = self.n;
        debug_assert_eq!(x.len(), n);
        debug_assert_eq!(g.len(), n);
        debug_assert_eq!(d.len(), n);

        // Fill `h` with the user's Hessian (lower triangle at
        // minimum — we never read the upper half).
        oracle.hessian(x, self.h.as_mut());

        // First try μ = 0 (exact Newton), then ramp up if needed.
        let mut mu = 0.0_f64;

        loop {
            // h_damped ← h + μ I  (lower triangle only — upper
            // half is untouched by Cholesky with Side::Lower).
            for j in 0..n {
                for i in j..n {
                    self.h_damped[(i, j)] = self.h[(i, j)];
                }
                self.h_damped[(j, j)] += mu;
            }

            // rhs ← -g.
            for i in 0..n {
                self.rhs[(i, 0)] = -g[i];
            }

            // In-place Cholesky (allocation-free).
            let stack = MemStack::new(&mut self.mem_buf);
            let factor_result = cholesky_in_place(
                self.h_damped.as_mut(),
                LltRegularization::default(),
                Par::Seq,
                stack,
                Default::default(),
            );

            match factor_result {
                Ok(_) => {
                    // Solve L Lᵀ d = -g into `rhs`. `solve_in_place`
                    // needs a MemStack but its scratch requirement
                    // is empty, so we reuse `mem_buf` safely.
                    let stack = MemStack::new(&mut self.mem_buf);
                    solve_in_place(self.h_damped.as_ref(), self.rhs.as_mut(), Par::Seq, stack);
                    for i in 0..n {
                        let di = self.rhs[(i, 0)];
                        if !di.is_finite() {
                            return Err(MethodError::NumericalFailure);
                        }
                        d[i] = di;
                    }
                    return Ok(());
                }
                Err(_) => {
                    // Non-PSD — bump μ and retry.
                    mu = if mu == 0.0 {
                        if self.initial_mu > 0.0 {
                            self.initial_mu
                        } else {
                            1e-4
                        }
                    } else {
                        mu * self.mu_growth
                    };
                    if !mu.is_finite() || mu > self.max_mu {
                        return Err(MethodError::HessianNotFactorable);
                    }
                }
            }
        }
    }
}

// =============================================================
// Internal dot product helper.
// =============================================================

#[inline]
fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn steepest_direction_is_negated_gradient() {
        let g = [1.0_f64, -2.0, 3.0];
        let mut d = [0.0; 3];
        steepest_direction(&g, &mut d);
        assert_eq!(d, [-1.0, 2.0, -3.0]);
    }

    #[test]
    fn bfgs_first_direction_equals_negative_gradient() {
        // H₀ = I  ⇒ d = -g.
        let state = BFGSState::new(3, 1e-10);
        let g = [1.0, -2.0, 3.0];
        let mut d = [0.0; 3];
        state.compute_direction(&g, &mut d).unwrap();
        assert_eq!(d, [-1.0, 2.0, -3.0]);
    }

    #[test]
    fn bfgs_skips_update_on_nonpositive_curvature() {
        let mut state = BFGSState::new(2, 1e-10);
        // s = (1, 0), y = (-1, 0)  ⇒  sᵀy = -1 < 0  ⇒  skip.
        let x = [0.0, 0.0];
        let x_new = [1.0, 0.0];
        let g = [0.0, 0.0];
        let g_new = [-1.0, 0.0];
        state.update(&x, &x_new, &g, &g_new);
        assert!(!state.first_update_done, "BFGS must skip bad pairs");

        // H untouched — still identity.
        assert_eq!(state.h[0], 1.0);
        assert_eq!(state.h[3], 1.0);
    }

    #[test]
    fn bfgs_applies_gamma_scaling_on_first_good_pair() {
        let mut state = BFGSState::new(2, 1e-10);
        // Construct a pair with sᵀy = 4, yᵀy = 8  ⇒  γ = 0.5.
        let x = [0.0, 0.0];
        let x_new = [1.0, 1.0];
        let g = [0.0, 0.0];
        let g_new = [2.0, 2.0];
        state.update(&x, &x_new, &g, &g_new);
        assert!(state.first_update_done);
        // After one update H is NOT plain γI anymore, but for this
        // carefully-chosen pair we can at least verify an update
        // happened (diagonal moved off 1.0).
        assert!((state.h[0] - 1.0).abs() > 1e-6);
    }

    #[test]
    fn lbfgs_with_empty_history_gives_negative_gradient() {
        let mut state = LBFGSState::new(3, 5, 1e-10);
        let g = [1.0, -2.0, 3.0];
        let mut d = [0.0; 3];
        state.compute_direction(&g, &mut d).unwrap();
        // No history ⇒ two-loop recursion produces `d = -g`
        // (the single-iteration fallback).
        assert_eq!(d, [-1.0, 2.0, -3.0]);
    }

    #[test]
    fn lbfgs_update_pushes_good_pair_onto_ring() {
        let mut state = LBFGSState::new(2, 3, 1e-10);
        let x = [0.0, 0.0];
        let x_new = [1.0, 0.0];
        let g = [0.0, 0.0];
        let g_new = [1.0, 0.0];
        state.update(&x, &x_new, &g, &g_new);
        assert_eq!(state.ring().count(), 1);
        assert_eq!(state.ring().s_slot(0), &[1.0, 0.0]);
        assert_eq!(state.ring().y_slot(0), &[1.0, 0.0]);
    }

    #[test]
    fn lbfgs_skips_push_on_nonpositive_curvature() {
        let mut state = LBFGSState::new(2, 3, 1e-10);
        // s = (1, 0), y = (-1, 0)  ⇒  sᵀy = -1  ⇒  skip.
        state.update(&[0.0, 0.0], &[1.0, 0.0], &[0.0, 0.0], &[-1.0, 0.0]);
        assert_eq!(state.ring().count(), 0, "ring must stay empty");
    }

    #[test]
    #[should_panic(expected = "curvature_epsilon must be finite and non-negative")]
    fn bfgs_rejects_negative_curvature_epsilon() {
        let _ = BFGSState::new(3, -1e-12);
    }

    #[test]
    #[should_panic(expected = "curvature_epsilon must be finite and non-negative")]
    fn bfgs_rejects_nan_curvature_epsilon() {
        let _ = BFGSState::new(3, f64::NAN);
    }

    #[test]
    #[should_panic(expected = "curvature_epsilon must be finite and non-negative")]
    fn lbfgs_rejects_negative_curvature_epsilon() {
        let _ = LBFGSState::new(3, 5, -1.0);
    }

    #[test]
    #[should_panic(expected = "curvature_epsilon must be finite and non-negative")]
    fn lbfgs_rejects_infinite_curvature_epsilon() {
        let _ = LBFGSState::new(3, 5, f64::INFINITY);
    }
}
