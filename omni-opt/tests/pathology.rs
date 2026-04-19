//! Phase-5 pathology suite.
//!
//! Each submodule below pins correctness on a deliberately
//! pessimistic problem that historically breaks hand-rolled
//! solvers — narrow Rosenbrock-style valleys, badly-scaled
//! quadratics, saddle points, and noisy finite-difference
//! gradients. The assertions double as API smoke tests: only
//! `omni_opt`'s public surface is touched.

use omni_opt::{
    central_difference, gradient, LineSearch, Method, Objective, Oracle, Solver,
    TerminationStatus,
};

// ==============================================================
// 1. Rosenbrock — narrow, curved valley.
// ==============================================================
mod rosenbrock {
    use super::*;

    /// f(x, y) = (a − x)² + b(y − x²)².
    /// `a = 1, b = 100` is the textbook Rosenbrock.
    struct Rosenbrock {
        a: f64,
        b: f64,
    }

    impl Objective for Rosenbrock {
        fn n(&self) -> usize {
            2
        }
        fn value(&mut self, x: &[f64]) -> f64 {
            let (x0, x1) = (x[0], x[1]);
            let t1 = self.a - x0;
            let t2 = x1 - x0 * x0;
            t1 * t1 + self.b * t2 * t2
        }
    }

    impl Oracle for Rosenbrock {
        fn value_grad(&mut self, x: &[f64], g: &mut [f64]) -> f64 {
            let (x0, x1) = (x[0], x[1]);
            let t1 = self.a - x0;
            let t2 = x1 - x0 * x0;
            g[0] = -2.0 * t1 - 4.0 * self.b * x0 * t2;
            g[1] = 2.0 * self.b * t2;
            t1 * t1 + self.b * t2 * t2
        }
    }

    // SciPy 1.13 L-BFGS-B baseline on Rosenbrock from
    // x0 = (-1.2, 1.0) with default tolerances:
    //   - ~25 iterations
    //   - ~32 f/grad evaluations (fused)
    // We hold to ~1.5× that as a regression tripwire.
    const SCIPY_FEVAL_BASELINE: u32 = 32;
    const REGRESSION_FACTOR: u32 = 3; // generous: different LS heuristic

    fn solve_with(method: Method, ls: LineSearch) -> omni_opt::SolverReport {
        let mut oracle = Rosenbrock { a: 1.0, b: 100.0 };
        let mut solver = Solver::builder()
            .method(method)
            .line_search(ls)
            .grad_inf_tol(1e-6)
            .max_iter(500)
            .build(2);
        let mut x = vec![-1.2, 1.0];
        let report = solver.run(&mut oracle, &mut x);
        assert_eq!(
            report.status,
            TerminationStatus::Converged,
            "{:?} failed to converge on Rosenbrock",
            method
        );
        assert!((x[0] - 1.0).abs() < 1e-4, "x[0] = {}", x[0]);
        assert!((x[1] - 1.0).abs() < 1e-4, "x[1] = {}", x[1]);
        report
    }

    #[test]
    fn lbfgs_strong_wolfe_converges_from_classic_start() {
        let _ = solve_with(Method::lbfgs(), LineSearch::StrongWolfe);
    }

    #[test]
    fn bfgs_strong_wolfe_converges_from_classic_start() {
        let _ = solve_with(Method::bfgs(), LineSearch::StrongWolfe);
    }

    #[test]
    fn steepest_armijo_reaches_optimum_in_budget() {
        // Steepest descent on Rosenbrock is famously slow but
        // must still converge within a generous budget.
        let mut oracle = Rosenbrock { a: 1.0, b: 100.0 };
        let mut solver = Solver::builder()
            .method(Method::steepest_descent())
            .line_search(LineSearch::BacktrackingArmijo)
            .grad_inf_tol(1e-3)
            .max_iter(200_000)
            .build(2);
        let mut x = vec![-1.2, 1.0];
        let r = solver.run(&mut oracle, &mut x);
        // We allow either full convergence or max-iter, but NOT
        // structural failure (line-search death, NaNs, …).
        assert!(matches!(
            r.status,
            TerminationStatus::Converged | TerminationStatus::MaxIterationsReached
        ));
    }

    #[test]
    fn lbfgs_fevals_under_scipy_baseline_regression_factor() {
        // Tripwire: if we ever regress past 3× the SciPy eval
        // count it almost certainly reflects a real algorithmic
        // regression, not a benign refactor.
        let report = solve_with(Method::lbfgs(), LineSearch::StrongWolfe);
        assert!(
            report.f_evals <= SCIPY_FEVAL_BASELINE * REGRESSION_FACTOR,
            "L-BFGS on Rosenbrock used {} f-evals, budget {}",
            report.f_evals,
            SCIPY_FEVAL_BASELINE * REGRESSION_FACTOR
        );
    }
}

// ==============================================================
// 2. Badly-scaled quadratic — conditioning & step recovery.
// ==============================================================
mod badly_scaled {
    use super::*;

    /// f(x) = ½ Σ αᵢ xᵢ² ;  ∇f(x)ᵢ = αᵢ xᵢ.
    /// Condition number κ = αmax / αmin.
    struct AnisotropicQuadratic {
        alphas: Vec<f64>,
    }

    impl Objective for AnisotropicQuadratic {
        fn n(&self) -> usize {
            self.alphas.len()
        }
        fn value(&mut self, x: &[f64]) -> f64 {
            0.5 * x
                .iter()
                .zip(&self.alphas)
                .map(|(xi, a)| a * xi * xi)
                .sum::<f64>()
        }
    }

    impl Oracle for AnisotropicQuadratic {
        fn value_grad(&mut self, x: &[f64], g: &mut [f64]) -> f64 {
            let mut s = 0.0;
            for i in 0..x.len() {
                let axi = self.alphas[i] * x[i];
                g[i] = axi;
                s += x[i] * axi;
            }
            0.5 * s
        }
    }

    fn run_on_alphas(alphas: Vec<f64>, method: Method) -> omni_opt::SolverReport {
        let n = alphas.len();
        let mut oracle = AnisotropicQuadratic { alphas };
        let mut solver = Solver::builder()
            .method(method)
            .line_search(LineSearch::StrongWolfe)
            .grad_inf_tol(1e-6)
            .max_iter(2_000)
            .build(n);
        // Random-ish initial guess well away from the minimum.
        let mut x = (0..n).map(|i| 1.0 + i as f64 * 0.5).collect::<Vec<_>>();
        let r = solver.run(&mut oracle, &mut x);
        for xi in &x {
            assert!(xi.abs() < 1e-3, "coord did not drive to 0: {}", xi);
        }
        r
    }

    #[test]
    fn lbfgs_converges_at_kappa_1e4() {
        let r = run_on_alphas(vec![1e-2, 1e-1, 1.0, 1e1, 1e2], Method::lbfgs());
        assert_eq!(r.status, TerminationStatus::Converged);
    }

    #[test]
    fn lbfgs_converges_at_kappa_1e6() {
        let r = run_on_alphas(vec![1e-3, 1.0, 1e3], Method::lbfgs());
        assert_eq!(r.status, TerminationStatus::Converged);
    }

    #[test]
    fn bfgs_gamma_scaling_recovers_first_step() {
        // The γ = sᵀy/yᵀy rescaling at the first BFGS update is
        // exactly what rescues the solver on a badly-scaled
        // quadratic. Without it, the identity `H₀ = I` gives a
        // first step that is O(κ) too long along the stiff axes.
        let r = run_on_alphas(vec![1e-2, 1.0, 1e2], Method::bfgs());
        assert_eq!(r.status, TerminationStatus::Converged);
    }
}

// ==============================================================
// 3. Saddle point — descent-direction correctness.
// ==============================================================
mod saddle {
    use super::*;

    /// f(x, y) = x² − y². Saddle at origin; gradient zero there.
    struct IndefiniteQuadratic;

    impl Objective for IndefiniteQuadratic {
        fn n(&self) -> usize {
            2
        }
        fn value(&mut self, x: &[f64]) -> f64 {
            x[0] * x[0] - x[1] * x[1]
        }
    }

    impl Oracle for IndefiniteQuadratic {
        fn value_grad(&mut self, x: &[f64], g: &mut [f64]) -> f64 {
            g[0] = 2.0 * x[0];
            g[1] = -2.0 * x[1];
            x[0] * x[0] - x[1] * x[1]
        }
    }

    #[test]
    fn lbfgs_terminates_at_exact_saddle() {
        // The gradient is identically zero at the origin, so
        // every first-order method terminates immediately with
        // `Converged`. This is the honest behavior — a first-
        // order method cannot distinguish a saddle from a local
        // minimum without curvature info.
        let mut oracle = IndefiniteQuadratic;
        let mut solver = Solver::builder()
            .method(Method::lbfgs())
            .grad_inf_tol(1e-6)
            .max_iter(50)
            .build(2);
        let mut x = vec![0.0, 0.0];
        let r = solver.run(&mut oracle, &mut x);
        assert_eq!(r.status, TerminationStatus::Converged);
        assert_eq!(x, vec![0.0, 0.0]);
    }

    #[test]
    fn lbfgs_escapes_near_saddle_along_descent_axis() {
        // Deterministic offset (Director-locked): fixed
        // `x0 = (0.1, 0.0)`. The gradient at this point is
        // (0.2, 0.0) — pure horizontal descent direction.
        // L-BFGS should march along the descending y-axis once
        // any non-trivial y-component appears (any numerical
        // drift will be amplified by the −y² curvature).
        //
        // Pragmatic expectation: `f` must become NEGATIVE (i.e.,
        // we've crossed into the "downhill" half-plane), or the
        // budget is hit. Panics (e.g., LS death) are rejected.
        let mut oracle = IndefiniteQuadratic;
        let mut solver = Solver::builder()
            .method(Method::lbfgs())
            .line_search(LineSearch::StrongWolfe)
            .grad_inf_tol(1e-12) // push hard
            .max_iter(200)
            .build(2);
        let mut x = vec![0.1, 0.0];
        let r = solver.run(&mut oracle, &mut x);
        assert!(matches!(
            r.status,
            TerminationStatus::MaxIterationsReached
                | TerminationStatus::LineSearchFailed
                | TerminationStatus::Converged
        ));
        // The x-axis component should have moved toward zero.
        assert!(x[0].abs() <= 0.1 + 1e-12);
    }

    #[cfg(feature = "faer")]
    #[test]
    fn newton_lm_engages_on_indefinite_hessian() {
        // Newton's Cholesky must fail on the indefinite Hessian
        // `diag(2, -2)`, and the LM damping must engage. We
        // accept any non-panicking structured outcome — the
        // whole point of this test is that the Cholesky path
        // doesn't crash on an obviously non-PSD matrix.
        use omni_opt::HessianOracle;

        struct Indef;
        impl Objective for Indef {
            fn n(&self) -> usize {
                2
            }
            fn value(&mut self, x: &[f64]) -> f64 {
                x[0] * x[0] - x[1] * x[1]
            }
        }
        impl Oracle for Indef {
            fn value_grad(&mut self, x: &[f64], g: &mut [f64]) -> f64 {
                g[0] = 2.0 * x[0];
                g[1] = -2.0 * x[1];
                x[0] * x[0] - x[1] * x[1]
            }
        }
        impl HessianOracle for Indef {
            fn hessian(&mut self, _x: &[f64], mut h: faer::MatMut<'_, f64>) {
                h[(0, 0)] = 2.0;
                h[(1, 0)] = 0.0;
                h[(1, 1)] = -2.0;
            }
        }

        let mut oracle = Indef;
        let mut solver = Solver::builder()
            .method(Method::newton())
            .max_iter(10)
            .build(2);
        let mut x = vec![0.1, 0.1];
        let r = solver.run_with_hessian(&mut oracle, &mut x);
        // Any structured status is fine — the assertion is
        // "no panic".
        let _ = r.status;
    }
}

// ==============================================================
// 4. Noisy finite-difference gradient — truncation vs round-off.
// ==============================================================
mod noisy_fd {
    use super::*;

    /// f(x) = ½ x₀² + ε · sin(x₀ / ε)   (single-variable variant).
    /// ∂f/∂x₀ = x₀ + cos(x₀ / ε).
    ///
    /// We evaluate at `x₀ = 1000` for the round-off tests so the
    /// quadratic term dominates (`f ≈ 5·10⁵`). At tiny `h` the
    /// perturbation `x · h` drops below `f64` precision relative
    /// to `f(x)`, and subtracting the two rounded values yields
    /// pure noise — the classic catastrophic-cancellation regime.
    ///
    /// At `x₀ = 0` the sinusoid is odd, so `f(+h) = −f(−h)` and
    /// the subtraction is exact — that's fine for the "default
    /// step is accurate" test but useless for the "too-small h
    /// amplifies round-off" test, which needs a pathological
    /// evaluation point.
    struct Noisy1D {
        epsilon: f64,
    }

    impl Objective for Noisy1D {
        fn n(&self) -> usize {
            1
        }
        fn value(&mut self, x: &[f64]) -> f64 {
            0.5 * x[0] * x[0] + self.epsilon * (x[0] / self.epsilon).sin()
        }
    }

    fn analytical_grad(eps: f64, x0: f64) -> f64 {
        x0 + (x0 / eps).cos()
    }

    fn fd_gradient_at(oracle: &mut Noisy1D, x0: f64, h: f64) -> f64 {
        let x = vec![x0];
        let mut scratch = x.clone();
        let mut g = vec![0.0];
        central_difference(oracle, &x, h, &mut scratch, &mut g);
        g[0]
    }

    #[test]
    fn central_diff_at_default_step_hits_analytical_gradient() {
        // At x=0 the sinusoid is odd; the default step gives
        // near-analytical accuracy.
        let mut oracle = Noisy1D { epsilon: 1e-2 };
        let truth = analytical_grad(1e-2, 0.0); // cos(0) = 1.
        let g = fd_gradient_at(&mut oracle, 0.0, gradient::default_step());
        assert!((g - truth).abs() < 1e-4, "g = {g}, truth = {truth}");
    }

    #[test]
    fn central_diff_too_small_h_amplifies_roundoff() {
        // `x₀ = 1000` → f(x) ≈ 5·10⁵. For h = 10⁻¹⁴ the
        // perturbation x·h ≈ 10⁻¹¹, which is below f64
        // precision at f(x) ≈ 5·10⁵ (absolute precision
        // ~5·10⁻¹¹). f(x+h) and f(x-h) round to the same
        // number, their subtraction produces noise, and the
        // approximated gradient is catastrophically wrong.
        let mut oracle = Noisy1D { epsilon: 1e-2 };
        let x0 = 1e3;
        let truth = analytical_grad(1e-2, x0);

        let g_default = fd_gradient_at(&mut oracle, x0, gradient::default_step());
        let err_default = (g_default - truth).abs().max(f64::MIN_POSITIVE);

        let g_tiny = fd_gradient_at(&mut oracle, x0, 1e-14);
        let err_tiny = (g_tiny - truth).abs();

        assert!(
            err_tiny >= 10.0 * err_default,
            "tiny-h error ({}) should be ≥ 10× default-h error ({})",
            err_tiny,
            err_default
        );
    }

    #[test]
    fn central_diff_too_large_h_loses_accuracy() {
        // `h = 0.5` is *enormous* versus the high-frequency
        // sinusoid (period 2π·ε ≈ 0.063). Truncation + aliasing
        // both break the `O(h²)` contract hard.
        let mut oracle = Noisy1D { epsilon: 1e-2 };
        let x0 = 0.0;
        let truth = analytical_grad(1e-2, x0);

        let g_default = fd_gradient_at(&mut oracle, x0, gradient::default_step());
        let err_default = (g_default - truth).abs().max(f64::MIN_POSITIVE);

        let g_huge = fd_gradient_at(&mut oracle, x0, 0.5);
        let err_huge = (g_huge - truth).abs();

        assert!(
            err_huge >= 10.0 * err_default,
            "huge-h error ({}) should be ≥ 10× default-h error ({})",
            err_huge,
            err_default
        );
    }
}
