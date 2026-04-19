//! Objective-function abstractions.
//!
//! All evaluation points cross this API as `&[f64]` — the exact
//! view produced by [`crate::OmniVec::as_slice`], so any Phase-1
//! storage backend (`Vec<f64>`, `&[f64]`, `faer::Col<f64>`) flows
//! through unchanged and without a copy.

/// Objective `f: ℝⁿ → ℝ`.
///
/// Minimum surface required by any solver that only consumes
/// function values (derivative-free methods, or gradient methods
/// paired with the central-difference fallback in
/// [`crate::gradient`]).
///
/// `&mut self` is deliberate: an implementor may hold private
/// scratch (intermediate buffers, autodiff tape, FFI handles) and
/// mutate it during evaluation without resorting to interior
/// mutability.
pub trait Objective {
    /// Dimensionality of the input. Fixed for the lifetime of the
    /// oracle — solvers read this once at construction and
    /// pre-size their workspaces against it.
    fn n(&self) -> usize;

    /// Return `f(x)`.
    ///
    /// # Precondition
    ///
    /// `x.len() == self.n()`. Implementations should enforce this
    /// with `debug_assert_eq!` so release builds stay branchless.
    fn value(&mut self, x: &[f64]) -> f64;
}

/// Objective with analytical gradient.
///
/// The single [`Oracle::value_grad`] entry point returns `f(x)`
/// and writes `∇f(x)` into the caller-owned `g`. There is
/// deliberately no separate `gradient` method: having one would
/// invite implementors to recompute `f`'s intermediate state on
/// every gradient call, defeating the whole point of the fused
/// interface.
pub trait Oracle: Objective {
    /// Evaluate `f` and `∇f` at `x`.
    ///
    /// # Preconditions
    ///
    /// * `x.len() == self.n()`
    /// * `g.len() == self.n()`
    ///
    /// Enforced by `debug_assert_eq!` in implementors.
    ///
    /// # Zero-allocation contract
    ///
    /// `g` is caller-owned; the implementor MUST NOT allocate on
    /// the heap during this call. Any internal scratch must live
    /// on `self` (allocated in the constructor).
    fn value_grad(&mut self, x: &[f64], g: &mut [f64]) -> f64;
}

/// Objective with an analytical Hessian `∇²f(x)`.
///
/// Required only by [`crate::Method::Newton`]. Feature-gated
/// because the only in-crate consumer factors the Hessian via
/// `faer::linalg::cholesky`; users without the `faer` feature
/// rely on BFGS / L-BFGS / steepest descent and never see this
/// trait.
#[cfg(feature = "faer")]
pub trait HessianOracle: Oracle {
    /// Write `∇²f(x)` into `h` as a symmetric `n × n` matrix.
    ///
    /// Only the lower triangle needs to be populated — the solver
    /// factorizes `h` through `Side::Lower` and never reads the
    /// upper triangle.
    ///
    /// # Preconditions
    ///
    /// * `x.len() == self.n()`
    /// * `h.nrows() == h.ncols() == self.n()`
    ///
    /// # Zero-allocation contract
    ///
    /// `h` is a mutable view into the solver's pre-allocated
    /// Hessian buffer. Implementors MUST NOT allocate; any
    /// internal scratch lives on `self` (allocated in the
    /// constructor).
    fn hessian(&mut self, x: &[f64], h: faer::MatMut<'_, f64>);
}

#[cfg(test)]
mod tests {
    use super::*;

    /// f(x) = ½ ‖x‖² ;  ∇f(x) = x.
    struct HalfSquaredNorm {
        n: usize,
    }

    impl Objective for HalfSquaredNorm {
        fn n(&self) -> usize {
            self.n
        }

        fn value(&mut self, x: &[f64]) -> f64 {
            debug_assert_eq!(x.len(), self.n);
            0.5 * x.iter().map(|v| v * v).sum::<f64>()
        }
    }

    impl Oracle for HalfSquaredNorm {
        fn value_grad(&mut self, x: &[f64], g: &mut [f64]) -> f64 {
            debug_assert_eq!(x.len(), self.n);
            debug_assert_eq!(g.len(), self.n);
            g.copy_from_slice(x);
            0.5 * x.iter().map(|v| v * v).sum::<f64>()
        }
    }

    #[test]
    fn objective_value_matches_analytical() {
        let mut f = HalfSquaredNorm { n: 3 };
        let x = [1.0_f64, 2.0, 3.0];
        assert_eq!(f.value(&x), 0.5 * (1.0 + 4.0 + 9.0));
    }

    #[test]
    fn oracle_value_grad_returns_fused_result() {
        let mut f = HalfSquaredNorm { n: 3 };
        let x = [1.0_f64, -2.0, 4.0];
        let mut g = [0.0_f64; 3];
        let v = f.value_grad(&x, &mut g);
        assert_eq!(v, 0.5 * (1.0 + 4.0 + 16.0));
        assert_eq!(g, [1.0, -2.0, 4.0]);
    }
}
