//! Central finite-difference gradient approximation:
//!
//! ```text
//!     ∂f/∂xᵢ  ≈  [ f(x + h eᵢ) − f(x − h eᵢ) ] / (2h)
//! ```
//!
//! # Zero-allocation strategy
//!
//! The perturbed point `x ± h eᵢ` is formed by overwriting a
//! single scalar inside a caller-owned `scratch` buffer, then
//! restoring it before moving to the next coordinate. Per
//! coordinate:
//!
//! ```text
//!     scratch[i] = x[i] + h;   f_plus  = oracle.value(scratch);
//!     scratch[i] = x[i] - h;   f_minus = oracle.value(scratch);
//!     scratch[i] = x[i];                      // restore
//!     out[i]     = (f_plus - f_minus) / (2h);
//! ```
//!
//! Only two `f64` writes per coordinate — no new allocations, no
//! full-vector copies in the inner loop.

use crate::oracle::{Objective, Oracle};

/// Recommended scalar step for `f64` central differences:
/// `h ≈ ε_mach^(1/3) ≈ 6.055 · 10⁻⁶`.
///
/// Rationale: central differences have truncation error `O(h²)`
/// and round-off error `O(ε/h)`; the two balance at `h ≈ ε^(1/3)`.
/// Callers handling variables that vary wildly in scale may want
/// to multiply by `max(|xᵢ|, 1)` per coordinate, but that is
/// outside the scope of the generic helper.
#[inline]
pub fn default_step() -> f64 {
    f64::EPSILON.cbrt()
}

/// Compute `∇f(x)` by central differences into `out`.
///
/// # Preconditions (`debug_assert!`)
///
/// * `scratch.len() == x.len()`
/// * `out.len() == x.len()`
/// * `h > 0.0 && h.is_finite()`
/// * `scratch` must equal `x` on entry.
///
/// # Why "scratch == x on entry" instead of copying
///
/// An automatic `scratch.copy_from_slice(x)` here would cost
/// `O(n)` writes every call. Callers that already maintain a
/// synced trial buffer (e.g., the line-search `x_trial`) can
/// skip that re-copy entirely. The [`CentralDifferenceOracle`]
/// adapter pays the `O(n)` memcpy once per `value_grad` call
/// to uphold the invariant on behalf of its owner; direct
/// callers of this function manage the invariant themselves.
///
/// On return, `scratch == x` once again — the buffer is safe to
/// reuse across successive points.
pub fn central_difference<O: Objective>(
    oracle: &mut O,
    x: &[f64],
    h: f64,
    scratch: &mut [f64],
    out: &mut [f64],
) {
    debug_assert_eq!(scratch.len(), x.len());
    debug_assert_eq!(out.len(), x.len());
    debug_assert!(
        h > 0.0 && h.is_finite(),
        "central_difference: h must be positive and finite"
    );
    // Enforce the "scratch == x on entry" contract. O(n) comparison
    // is acceptable here: `debug_assert!` is stripped in release
    // builds, so the hot path stays branchless while debug builds
    // catch any caller that violated the invariant.
    debug_assert!(
        scratch == x,
        "central_difference: scratch must equal x on entry"
    );

    let two_h_inv = 0.5 / h;
    for i in 0..x.len() {
        let xi = x[i];
        scratch[i] = xi + h;
        let f_plus = oracle.value(scratch);
        scratch[i] = xi - h;
        let f_minus = oracle.value(scratch);
        scratch[i] = xi;
        out[i] = (f_plus - f_minus) * two_h_inv;
    }
}

/// Adapter that promotes any [`Objective`] to a full [`Oracle`]
/// by computing the gradient via central differences.
///
/// # Allocation policy
///
/// A single `Vec<f64>` of length `n` is allocated in [`new`]
/// to serve as the permanent scratch for [`central_difference`].
/// All subsequent [`Oracle::value_grad`] calls reuse it — zero
/// heap allocation on the solver hot path.
///
/// # Cost
///
/// Each [`Oracle::value_grad`] performs `2n + 1` objective
/// evaluations (`f(x)` once, then `f(x ± h eᵢ)` for each `i`).
/// Prefer an analytical gradient when the objective is cheap to
/// evaluate only at `x`; use this adapter when a gradient is
/// unavailable or algorithmically expensive to hand-derive.
///
/// [`new`]: CentralDifferenceOracle::new
pub struct CentralDifferenceOracle<O: Objective> {
    inner: O,
    h: f64,
    // Scratch for `central_difference`. Resynced to `x` at the
    // start of every `value_grad` call so the
    // "scratch == x on entry" precondition of the helper holds.
    scratch: Vec<f64>,
}

impl<O: Objective> CentralDifferenceOracle<O> {
    /// Wrap `inner` with step size `h`.
    ///
    /// # Panics
    ///
    /// Panics if `inner.n() == 0` or `h <= 0` or `!h.is_finite()`.
    /// Misconfiguration fails at construction time, never inside
    /// the solver loop.
    pub fn new(inner: O, h: f64) -> Self {
        let n = inner.n();
        assert!(n > 0, "CentralDifferenceOracle::new: inner.n() must be > 0");
        assert!(
            h > 0.0 && h.is_finite(),
            "CentralDifferenceOracle::new: h must be positive and finite"
        );
        Self {
            scratch: vec![0.0; n],
            inner,
            h,
        }
    }

    /// Convenience constructor using [`default_step`].
    pub fn with_default_step(inner: O) -> Self {
        Self::new(inner, default_step())
    }

    /// Shared borrow of the wrapped objective.
    pub fn inner(&self) -> &O {
        &self.inner
    }

    /// Mutable borrow of the wrapped objective.
    pub fn inner_mut(&mut self) -> &mut O {
        &mut self.inner
    }

    /// Step size `h` used for the central-difference stencil.
    pub fn step(&self) -> f64 {
        self.h
    }
}

impl<O: Objective> Objective for CentralDifferenceOracle<O> {
    #[inline]
    fn n(&self) -> usize {
        self.inner.n()
    }

    #[inline]
    fn value(&mut self, x: &[f64]) -> f64 {
        self.inner.value(x)
    }
}

impl<O: Objective> Oracle for CentralDifferenceOracle<O> {
    fn value_grad(&mut self, x: &[f64], g: &mut [f64]) -> f64 {
        debug_assert_eq!(x.len(), self.inner.n());
        debug_assert_eq!(g.len(), self.inner.n());

        // Resync scratch ← x (uphold the helper's precondition).
        self.scratch.copy_from_slice(x);
        central_difference(&mut self.inner, x, self.h, &mut self.scratch, g);
        self.inner.value(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oracle::Objective;

    /// f(x) = ½ ‖x‖² ;  ∇f(x) = x.
    struct HalfSquaredNorm {
        n: usize,
        calls: usize,
    }

    impl Objective for HalfSquaredNorm {
        fn n(&self) -> usize {
            self.n
        }

        fn value(&mut self, x: &[f64]) -> f64 {
            self.calls += 1;
            0.5 * x.iter().map(|v| v * v).sum::<f64>()
        }
    }

    #[test]
    fn central_difference_matches_analytical_gradient() {
        let mut f = HalfSquaredNorm { n: 4, calls: 0 };
        let x = [1.0_f64, -2.0, 3.0, 0.5];
        let mut scratch = x.to_vec();
        let mut g = [0.0_f64; 4];
        central_difference(&mut f, &x, default_step(), &mut scratch, &mut g);

        // For a quadratic, central differences are exact up to
        // round-off — tolerance is determined by ε / h ≈ ε^(2/3).
        let tol = 1e-8;
        for (got, want) in g.iter().zip(x.iter()) {
            assert!((got - want).abs() < tol, "got {}, want {}", got, want);
        }
    }

    #[test]
    fn central_difference_restores_scratch_to_x() {
        let mut f = HalfSquaredNorm { n: 3, calls: 0 };
        let x = [1.5_f64, -0.25, 4.0];
        let mut scratch = x.to_vec();
        let mut g = [0.0_f64; 3];
        central_difference(&mut f, &x, 1e-5, &mut scratch, &mut g);
        assert_eq!(scratch, x.to_vec());
    }

    #[test]
    fn central_difference_uses_2n_oracle_calls() {
        let mut f = HalfSquaredNorm { n: 5, calls: 0 };
        let x = vec![1.0_f64; 5];
        let mut scratch = x.clone();
        let mut g = vec![0.0_f64; 5];
        central_difference(&mut f, &x, 1e-5, &mut scratch, &mut g);
        assert_eq!(f.calls, 2 * 5);
    }

    #[test]
    fn adapter_value_grad_matches_analytical() {
        let inner = HalfSquaredNorm { n: 3, calls: 0 };
        let mut oracle = CentralDifferenceOracle::with_default_step(inner);
        let x = [2.0_f64, -1.0, 0.5];
        let mut g = [0.0_f64; 3];
        let v = oracle.value_grad(&x, &mut g);
        assert!((v - 0.5 * (4.0 + 1.0 + 0.25)).abs() < 1e-12);

        let tol = 1e-8;
        for (got, want) in g.iter().zip(x.iter()) {
            assert!((got - want).abs() < tol);
        }
    }

    #[test]
    #[should_panic]
    fn new_rejects_zero_dimension() {
        let inner = HalfSquaredNorm { n: 0, calls: 0 };
        let _ = CentralDifferenceOracle::new(inner, 1e-5);
    }

    #[test]
    #[should_panic]
    fn new_rejects_nonpositive_step() {
        let inner = HalfSquaredNorm { n: 3, calls: 0 };
        let _ = CentralDifferenceOracle::new(inner, 0.0);
    }

    #[test]
    #[should_panic(expected = "scratch must equal x on entry")]
    fn central_difference_rejects_desynced_scratch() {
        let mut f = HalfSquaredNorm { n: 3, calls: 0 };
        let x = [1.0_f64, 2.0, 3.0];
        // Deliberately out of sync: one element differs from x.
        let mut scratch = [1.0_f64, 2.0, 3.5];
        let mut g = [0.0_f64; 3];
        central_difference(&mut f, &x, 1e-5, &mut scratch, &mut g);
    }
}
