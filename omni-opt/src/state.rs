//! User-facing solver configuration and structured termination
//! reporting.

/// User-configured tolerances for solver termination.
///
/// Plain-old-data and [`Copy`], so it can be stamped into the
/// solver once at the start of `run()` and then referenced without
/// borrow-checker friction in the hot loop. Every field is public
/// by design — there is no invariant worth enforcing behind a
/// setter.
#[derive(Clone, Copy, Debug)]
pub struct StoppingCriteria {
    /// Terminate when `‖∇f(x)‖_∞ ≤ grad_inf_tol`.
    pub grad_inf_tol: f64,

    /// Terminate when `‖x_{k+1} − x_k‖₂ ≤ step_tol · (1 + ‖x_k‖₂)`.
    ///
    /// The relative-plus-absolute form avoids scale-dependence
    /// for badly-scaled problems while still handling `‖x‖ ≈ 0`.
    pub step_tol: f64,

    /// Terminate when
    ///
    /// ```text
    ///   (f_k − f_{k+1}) / max(|f_k|, |f_{k+1}|, 1.0) ≤ rel_f_tol.
    /// ```
    ///
    /// The symmetric denominator with a unit floor prevents the
    /// test from being trivially satisfied when the objective is
    /// plunging toward zero (common for loss functions in deep
    /// learning) — a one-sided `|f_k|` denominator would falsely
    /// certify convergence long before the optimum is reached.
    pub rel_f_tol: f64,

    /// Number of consecutive non-improving iterations that
    /// triggers [`TerminationStatus::StagnationDetected`].
    /// A value of `0` disables the check.
    pub stagnation_window: usize,

    /// Hard cap on iterations, enforced regardless of any other
    /// criterion.
    pub max_iter: usize,
}

impl Default for StoppingCriteria {
    /// Conservative defaults suitable for smooth, medium-scale
    /// unconstrained problems.
    fn default() -> Self {
        Self {
            grad_inf_tol: 1e-6,
            step_tol: 1e-10,
            rel_f_tol: 1e-12,
            stagnation_window: 10,
            max_iter: 1000,
        }
    }
}

/// Structured, non-allocating solver exit state.
///
/// Solvers return this inside a future `Report` struct (dimension,
/// iterations, final `f`, final gradient norm, …); we deliberately
/// do **not** use `Result<_, String>` or `Box<dyn Error>` — the
/// termination path itself must stay allocation-free, so the
/// reason is carried as an enum discriminant and any human-readable
/// rendering is deferred to [`Display`].
///
/// [`Display`]: core::fmt::Display
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TerminationStatus {
    /// All active convergence tests passed.
    Converged,
    /// `StoppingCriteria::max_iter` reached before convergence.
    MaxIterationsReached,
    /// The line search could not produce a valid step (e.g., Wolfe
    /// conditions could not be satisfied within the bracket limit).
    LineSearchFailed,
    /// Objective failed to improve for
    /// `StoppingCriteria::stagnation_window` consecutive iterations.
    StagnationDetected,
    /// A NaN or infinity appeared in `f`, `∇f`, or the step.
    NumericalFailure,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_criteria_are_conservative() {
        let c = StoppingCriteria::default();
        assert!(c.grad_inf_tol > 0.0 && c.grad_inf_tol < 1.0);
        assert!(c.step_tol > 0.0 && c.step_tol < c.grad_inf_tol);
        assert!(c.rel_f_tol > 0.0 && c.rel_f_tol < c.step_tol);
        assert!(c.max_iter > 0);
    }

    #[test]
    fn stopping_criteria_is_copy() {
        fn assert_copy<T: Copy>() {}
        assert_copy::<StoppingCriteria>();
    }

    #[test]
    fn termination_status_is_comparable() {
        assert_eq!(TerminationStatus::Converged, TerminationStatus::Converged);
        assert_ne!(
            TerminationStatus::Converged,
            TerminationStatus::MaxIterationsReached
        );
    }
}
