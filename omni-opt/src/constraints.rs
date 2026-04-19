//! Projected-clipping box constraints  `lᵢ ≤ xᵢ ≤ uᵢ`.
//!
//! # Scope & limitations
//!
//! This module implements **projected clipping** — the simplest
//! constraint-enforcement strategy. After every accepted step,
//! each coordinate is clipped to its feasible interval via
//!
//! ```text
//!     xᵢ ← max(lᵢ, min(xᵢ, uᵢ)).
//! ```
//!
//! The iterate is always feasible on exit.
//!
//! Projected clipping is deliberately NOT a full active-set
//! L-BFGS-B solver. It does not perform:
//!
//! * the generalized Cauchy-point search that drives L-BFGS-B,
//! * active-set identification with complementary-slackness
//!   safeguards,
//! * reduced-space curvature updates that prune `sₖ / yₖ`
//!   components along active constraints.
//!
//! Quasi-Newton curvature (`BFGS` / `LBFGS`) can therefore be
//! mildly corrupted when many bounds are active; convergence
//! near the boundary may slow accordingly. A full L-BFGS-B
//! implementation is scoped for a future milestone; callers
//! requiring certified first-order optimality under active
//! bounds should use that instead.

// Clippy: numerical kernels index multiple slices in lockstep.
#![allow(clippy::needless_range_loop)]

/// Componentwise lower and upper bounds on `x`.
///
/// `f64::NEG_INFINITY` / `f64::INFINITY` denote an unbounded
/// side; mixing finite and infinite endpoints across coordinates
/// is legal. The struct owns two `Vec<f64>` of length `n`, both
/// allocated in the constructor; every subsequent
/// [`Self::project_in_place`] / [`Self::reduced_gradient`] call
/// operates entirely in place.
///
/// See the [module docs](self) for a full discussion of the
/// limitations of projected clipping versus true L-BFGS-B.
#[derive(Clone, Debug)]
pub struct BoxConstraints {
    lower: Vec<f64>,
    upper: Vec<f64>,
}

impl BoxConstraints {
    /// Build from user-supplied bounds.
    ///
    /// # Panics
    ///
    /// * `lower.len() != upper.len()` or either is empty.
    /// * any `lower[i] > upper[i]`.
    /// * any NaN endpoint.
    pub fn new(lower: Vec<f64>, upper: Vec<f64>) -> Self {
        assert_eq!(
            lower.len(),
            upper.len(),
            "BoxConstraints::new: lower.len() != upper.len()"
        );
        assert!(!lower.is_empty(), "BoxConstraints::new: bounds must be non-empty");
        for i in 0..lower.len() {
            assert!(
                !lower[i].is_nan() && !upper[i].is_nan(),
                "BoxConstraints::new: NaN endpoint at index {}",
                i
            );
            assert!(
                lower[i] <= upper[i],
                "BoxConstraints::new: lower[{i}] ({}) > upper[{i}] ({})",
                lower[i],
                upper[i],
                i = i
            );
        }
        Self { lower, upper }
    }

    /// Uniform scalar bounds applied to every coordinate.
    pub fn uniform(n: usize, lower: f64, upper: f64) -> Self {
        assert!(n > 0, "BoxConstraints::uniform: n must be > 0");
        assert!(
            !lower.is_nan() && !upper.is_nan(),
            "BoxConstraints::uniform: endpoints must be non-NaN"
        );
        assert!(
            lower <= upper,
            "BoxConstraints::uniform: lower ({}) > upper ({})",
            lower,
            upper
        );
        Self {
            lower: vec![lower; n],
            upper: vec![upper; n],
        }
    }

    /// Upper bounds only; lower set to `-∞` on every coordinate.
    pub fn upper_only(upper: Vec<f64>) -> Self {
        assert!(
            !upper.is_empty(),
            "BoxConstraints::upper_only: bounds must be non-empty"
        );
        let lower = vec![f64::NEG_INFINITY; upper.len()];
        Self::new(lower, upper)
    }

    /// Lower bounds only; upper set to `+∞` on every coordinate.
    pub fn lower_only(lower: Vec<f64>) -> Self {
        assert!(
            !lower.is_empty(),
            "BoxConstraints::lower_only: bounds must be non-empty"
        );
        let upper = vec![f64::INFINITY; lower.len()];
        Self::new(lower, upper)
    }

    /// Dimensionality `n`.
    #[inline]
    pub fn len(&self) -> usize {
        self.lower.len()
    }

    /// `true` iff `len() == 0`. Cannot actually occur — the
    /// constructors reject empty bounds — but keeps clippy happy
    /// alongside `len()`.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.lower.is_empty()
    }

    /// Read-only view of the lower bounds.
    #[inline]
    pub fn lower(&self) -> &[f64] {
        &self.lower
    }

    /// Read-only view of the upper bounds.
    #[inline]
    pub fn upper(&self) -> &[f64] {
        &self.upper
    }

    /// In-place projected clipping:
    /// `xᵢ ← max(lᵢ, min(xᵢ, uᵢ))`.
    ///
    /// Returns `true` iff at least one coordinate moved. The
    /// solver uses that bit to decide whether a gradient
    /// re-evaluation is required — projection that leaves the
    /// iterate untouched costs exactly nothing downstream.
    ///
    /// # Preconditions (`debug_assert!`)
    ///
    /// `x.len() == self.len()`.
    pub fn project_in_place(&self, x: &mut [f64]) -> bool {
        debug_assert_eq!(x.len(), self.lower.len());
        let mut clipped = false;
        for i in 0..x.len() {
            let xi = x[i];
            // NaN handling is deliberate and explicit: Rust's
            // `f64::max` follows IEEE 2008 `maxNum`, which picks
            // the *non-NaN* argument, silently coercing NaN to
            // the bound. We want the opposite — a NaN iterate
            // means the upstream oracle produced a bad number
            // and we must surface it. We therefore preserve
            // NaN as-is and report the coordinate as "moved"
            // so the solver re-evaluates the gradient (which
            // will also be NaN) and escalates to
            // `TerminationStatus::NumericalFailure`.
            if xi.is_nan() {
                clipped = true;
                continue;
            }
            let projected = xi.max(self.lower[i]).min(self.upper[i]);
            if projected != xi {
                clipped = true;
            }
            x[i] = projected;
        }
        clipped
    }

    /// `true` iff every `xᵢ` lies in `[lᵢ, uᵢ]`.
    ///
    /// # Preconditions (`debug_assert!`)
    ///
    /// `x.len() == self.len()`.
    pub fn is_feasible(&self, x: &[f64]) -> bool {
        debug_assert_eq!(x.len(), self.lower.len());
        for i in 0..x.len() {
            if !(x[i] >= self.lower[i] && x[i] <= self.upper[i]) {
                return false;
            }
        }
        true
    }

    /// Write the reduced gradient into `out`:
    ///
    /// ```text
    ///   g̃ᵢ = 0   if xᵢ = lᵢ and gᵢ > 0
    ///   g̃ᵢ = 0   if xᵢ = uᵢ and gᵢ < 0
    ///   g̃ᵢ = gᵢ  otherwise.
    /// ```
    ///
    /// First-order optimality under the box `[l, u]` is exactly
    /// `g̃ = 0`, so `‖g̃‖∞ ≤ grad_inf_tol` is the right
    /// convergence test under bounds.
    ///
    /// # Preconditions (`debug_assert!`)
    ///
    /// `x.len() == g.len() == out.len() == self.len()`.
    pub fn reduced_gradient(&self, x: &[f64], g: &[f64], out: &mut [f64]) {
        debug_assert_eq!(x.len(), self.lower.len());
        debug_assert_eq!(g.len(), self.lower.len());
        debug_assert_eq!(out.len(), self.lower.len());
        for i in 0..x.len() {
            let xi = x[i];
            let gi = g[i];
            let pinned_below = xi <= self.lower[i] && gi > 0.0;
            let pinned_above = xi >= self.upper[i] && gi < 0.0;
            out[i] = if pinned_below || pinned_above { 0.0 } else { gi };
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn uniform_constructs_symmetric_box() {
        let b = BoxConstraints::uniform(3, -1.0, 1.0);
        assert_eq!(b.len(), 3);
        assert_eq!(b.lower(), &[-1.0, -1.0, -1.0]);
        assert_eq!(b.upper(), &[1.0, 1.0, 1.0]);
    }

    #[test]
    fn upper_only_leaves_lower_unbounded() {
        let b = BoxConstraints::upper_only(vec![5.0, 10.0]);
        assert_eq!(b.lower(), &[f64::NEG_INFINITY, f64::NEG_INFINITY]);
        assert_eq!(b.upper(), &[5.0, 10.0]);
    }

    #[test]
    fn lower_only_leaves_upper_unbounded() {
        let b = BoxConstraints::lower_only(vec![-3.0, 0.0]);
        assert_eq!(b.lower(), &[-3.0, 0.0]);
        assert_eq!(b.upper(), &[f64::INFINITY, f64::INFINITY]);
    }

    #[test]
    fn project_noop_when_inside_box() {
        let b = BoxConstraints::uniform(3, -1.0, 1.0);
        let mut x = vec![0.0, 0.5, -0.5];
        assert!(!b.project_in_place(&mut x));
        assert_eq!(x, vec![0.0, 0.5, -0.5]);
    }

    #[test]
    fn project_clips_violating_coordinates_and_reports_true() {
        let b = BoxConstraints::uniform(3, -1.0, 1.0);
        let mut x = vec![-5.0, 0.5, 7.0];
        assert!(b.project_in_place(&mut x));
        assert_eq!(x, vec![-1.0, 0.5, 1.0]);
    }

    #[test]
    fn project_handles_asymmetric_bounds() {
        let b = BoxConstraints::new(vec![0.0, -10.0], vec![5.0, 10.0]);
        let mut x = vec![-1.0, 15.0];
        assert!(b.project_in_place(&mut x));
        assert_eq!(x, vec![0.0, 10.0]);
    }

    #[test]
    fn project_propagates_nan() {
        let b = BoxConstraints::uniform(2, -1.0, 1.0);
        let mut x = vec![f64::NAN, 0.0];
        // NaN != NaN so the "clipped" flag trips; that is the
        // intended behavior — the solver will re-evaluate, the
        // oracle returns NaN, and the descent loop escalates to
        // NumericalFailure.
        assert!(b.project_in_place(&mut x));
        assert!(x[0].is_nan());
        assert_eq!(x[1], 0.0);
    }

    #[test]
    fn is_feasible_true_inside() {
        let b = BoxConstraints::uniform(3, -1.0, 1.0);
        assert!(b.is_feasible(&[0.0, 1.0, -1.0]));
    }

    #[test]
    fn is_feasible_false_outside() {
        let b = BoxConstraints::uniform(2, -1.0, 1.0);
        assert!(!b.is_feasible(&[1.5, 0.0]));
        assert!(!b.is_feasible(&[0.0, -2.0]));
    }

    #[test]
    fn reduced_gradient_zeros_coord_pinned_below_with_outward_grad() {
        // x[0] = lower[0], g[0] > 0 → pinned below, zeroed.
        let b = BoxConstraints::new(vec![0.0, 0.0], vec![10.0, 10.0]);
        let x = vec![0.0, 5.0];
        let g = vec![3.0, 1.0];
        let mut out = vec![0.0; 2];
        b.reduced_gradient(&x, &g, &mut out);
        assert_eq!(out, vec![0.0, 1.0]);
    }

    #[test]
    fn reduced_gradient_zeros_coord_pinned_above_with_outward_grad() {
        // x[1] = upper[1], g[1] < 0 → pinned above, zeroed.
        let b = BoxConstraints::new(vec![0.0, 0.0], vec![10.0, 10.0]);
        let x = vec![5.0, 10.0];
        let g = vec![1.0, -2.0];
        let mut out = vec![0.0; 2];
        b.reduced_gradient(&x, &g, &mut out);
        assert_eq!(out, vec![1.0, 0.0]);
    }

    #[test]
    fn reduced_gradient_keeps_coord_pinned_but_pointing_inward() {
        // x[0] = lower[0], g[0] < 0 → pointing INTO the feasible
        // set, keep the gradient.
        let b = BoxConstraints::new(vec![0.0], vec![10.0]);
        let x = vec![0.0];
        let g = vec![-1.0];
        let mut out = vec![0.0];
        b.reduced_gradient(&x, &g, &mut out);
        assert_eq!(out, vec![-1.0]);
    }

    #[test]
    #[should_panic(expected = "lower")]
    fn new_rejects_inverted_bounds() {
        let _ = BoxConstraints::new(vec![1.0], vec![0.0]);
    }

    #[test]
    #[should_panic(expected = "NaN")]
    fn new_rejects_nan_endpoint() {
        let _ = BoxConstraints::new(vec![f64::NAN], vec![1.0]);
    }

    #[test]
    #[should_panic(expected = "lower.len")]
    fn new_rejects_length_mismatch() {
        let _ = BoxConstraints::new(vec![0.0, 0.0], vec![1.0]);
    }

    #[test]
    #[should_panic]
    fn uniform_rejects_zero_n() {
        let _ = BoxConstraints::uniform(0, -1.0, 1.0);
    }
}
