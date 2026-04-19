//! Phase-5 dual-axis benchmarks.
//!
//! This binary exposes two criterion groups through a single
//! `cargo bench` entry point:
//!
//! * **`algorithmic`** — measures *function / gradient evaluation
//!   counts*, using a custom `Measurement` that replaces
//!   criterion's default wall-clock timer with a thread-local
//!   evaluation counter. Gives the full criterion statistics
//!   (mean, stddev, outliers, change detection) against the
//!   algorithmic axis rather than the systems axis.
//! * **`systems`** — measures end-to-end wall-clock latency on
//!   the same problems with criterion's default timer. No
//!   third-party solver comparisons yet — the `argmin`
//!   comparison module is scoped for a post-release milestone
//!   so the core `cargo bench` does not drag extra linear-algebra
//!   crates into the dependency graph.
//!
//! Run a single axis:
//!
//! ```bash
//! cargo bench --bench solver_bench -- algorithmic
//! cargo bench --bench solver_bench -- systems
//! ```

// criterion's `criterion_group!` macro emits undocumented
// helper functions; allow missing docs for the whole bench.
#![allow(missing_docs)]

use std::cell::Cell;
use std::time::Duration;

use criterion::{
    criterion_group, criterion_main,
    measurement::{Measurement, ValueFormatter},
    BenchmarkId, Criterion, Throughput,
};

use omni_opt::{LineSearch, Method, Objective, Oracle, Solver};

// ==============================================================
// Evaluation counter — shared infrastructure for axis 1.
// ==============================================================

thread_local! {
    /// Running total of `oracle.value` + `oracle.value_grad`
    /// calls observed on this thread. Read by the
    /// `EvalCountMeasurement`; incremented by `CountingOracle`.
    static EVAL_COUNTER: Cell<u64> = const { Cell::new(0) };
}

/// Oracle wrapper that ticks the thread-local counter on every
/// value or value_grad call.
struct CountingOracle<O> {
    inner: O,
}

impl<O: Objective> Objective for CountingOracle<O> {
    fn n(&self) -> usize {
        self.inner.n()
    }
    fn value(&mut self, x: &[f64]) -> f64 {
        EVAL_COUNTER.with(|c| c.set(c.get().wrapping_add(1)));
        self.inner.value(x)
    }
}

impl<O: Oracle> Oracle for CountingOracle<O> {
    fn value_grad(&mut self, x: &[f64], g: &mut [f64]) -> f64 {
        EVAL_COUNTER.with(|c| c.set(c.get().wrapping_add(1)));
        self.inner.value_grad(x, g)
    }
}

// --------------------------------------------------------------
// Custom criterion Measurement: reports `evals` in place of ns.
// --------------------------------------------------------------

/// Zero-sized measurement token. Plug into a criterion config
/// via `Criterion::default().with_measurement(EvalCountMeasurement)`.
pub struct EvalCountMeasurement;

impl Measurement for EvalCountMeasurement {
    type Intermediate = u64;
    type Value = u64;

    fn start(&self) -> Self::Intermediate {
        EVAL_COUNTER.with(|c| c.get())
    }

    fn end(&self, start: Self::Intermediate) -> Self::Value {
        EVAL_COUNTER.with(|c| c.get().wrapping_sub(start))
    }

    fn add(&self, v1: &Self::Value, v2: &Self::Value) -> Self::Value {
        v1.wrapping_add(*v2)
    }

    fn zero(&self) -> Self::Value {
        0
    }

    fn to_f64(&self, value: &Self::Value) -> f64 {
        *value as f64
    }

    fn formatter(&self) -> &dyn ValueFormatter {
        &EvalCountFormatter
    }
}

struct EvalCountFormatter;

impl ValueFormatter for EvalCountFormatter {
    fn format_value(&self, value: f64) -> String {
        format!("{:.0} evals", value)
    }

    fn format_throughput(&self, throughput: &Throughput, value: f64) -> String {
        match throughput {
            Throughput::Elements(n) => format!("{:.2} evals/element", value / *n as f64),
            Throughput::Bytes(n) | Throughput::BytesDecimal(n) => {
                format!("{:.2} evals/byte", value / *n as f64)
            }
        }
    }

    fn scale_values(&self, _typical_value: f64, _values: &mut [f64]) -> &'static str {
        "evals"
    }

    fn scale_throughputs(
        &self,
        _typical_value: f64,
        throughput: &Throughput,
        values: &mut [f64],
    ) -> &'static str {
        match throughput {
            Throughput::Elements(n) => {
                for v in values.iter_mut() {
                    *v /= *n as f64;
                }
                "evals/element"
            }
            Throughput::Bytes(n) | Throughput::BytesDecimal(n) => {
                for v in values.iter_mut() {
                    *v /= *n as f64;
                }
                "evals/byte"
            }
        }
    }

    fn scale_for_machines(&self, _values: &mut [f64]) -> &'static str {
        "evals"
    }
}

// ==============================================================
// Test oracles used across both axes.
// ==============================================================

/// Classical Rosenbrock with a=1, b=100.
#[derive(Clone, Copy)]
struct Rosenbrock;

impl Objective for Rosenbrock {
    fn n(&self) -> usize {
        2
    }
    fn value(&mut self, x: &[f64]) -> f64 {
        let (a, b) = (1.0, 100.0);
        let (x0, x1) = (x[0], x[1]);
        (a - x0).powi(2) + b * (x1 - x0 * x0).powi(2)
    }
}

impl Oracle for Rosenbrock {
    fn value_grad(&mut self, x: &[f64], g: &mut [f64]) -> f64 {
        let (a, b) = (1.0, 100.0);
        let (x0, x1) = (x[0], x[1]);
        let t1 = a - x0;
        let t2 = x1 - x0 * x0;
        g[0] = -2.0 * t1 - 4.0 * b * x0 * t2;
        g[1] = 2.0 * b * t2;
        t1 * t1 + b * t2 * t2
    }
}

/// Anisotropic quadratic f(x) = ½ Σ αᵢ xᵢ².
#[derive(Clone)]
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

// ==============================================================
// Shared bench helpers.
// ==============================================================

fn fresh_solver(method: Method, ls: LineSearch, n: usize) -> Solver {
    Solver::builder()
        .method(method)
        .line_search(ls)
        .grad_inf_tol(1e-6)
        .max_iter(500)
        .build(n)
}

// ==============================================================
// AXIS 1 — algorithmic efficiency (f/g eval counts).
// ==============================================================

fn counts_rosenbrock(c: &mut Criterion<EvalCountMeasurement>) {
    let mut group = c.benchmark_group("rosenbrock_2d");
    for (method_name, method) in [
        ("steepest_descent", Method::steepest_descent()),
        ("bfgs", Method::bfgs()),
        ("lbfgs", Method::lbfgs()),
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(method_name),
            &method,
            |b, &method| {
                b.iter(|| {
                    let mut oracle = CountingOracle { inner: Rosenbrock };
                    let mut solver = fresh_solver(method, LineSearch::StrongWolfe, 2);
                    let mut x = vec![-1.2, 1.0];
                    let _ = solver.run(&mut oracle, &mut x);
                });
            },
        );
    }
    group.finish();
}

fn counts_badly_scaled(c: &mut Criterion<EvalCountMeasurement>) {
    let mut group = c.benchmark_group("badly_scaled_quadratic");
    let alphas = vec![1e-3, 1e-1, 1.0, 1e1, 1e3];
    for (method_name, method) in [("bfgs", Method::bfgs()), ("lbfgs", Method::lbfgs())] {
        group.bench_with_input(
            BenchmarkId::from_parameter(method_name),
            &method,
            |b, &method| {
                b.iter(|| {
                    let inner = AnisotropicQuadratic { alphas: alphas.clone() };
                    let mut oracle = CountingOracle { inner };
                    let mut solver = fresh_solver(method, LineSearch::StrongWolfe, 5);
                    let mut x = vec![1.0, 1.0, 1.0, 1.0, 1.0];
                    let _ = solver.run(&mut oracle, &mut x);
                });
            },
        );
    }
    group.finish();
}

// ==============================================================
// AXIS 2 — systems performance (wall-clock).
// ==============================================================

fn wall_clock_rosenbrock(c: &mut Criterion) {
    let mut group = c.benchmark_group("rosenbrock_2d_wall");
    // Tight sample config — 2D Rosenbrock is fast, statistics
    // need many repetitions to stabilize.
    group
        .sample_size(100)
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(2));

    for (method_name, method) in [
        ("bfgs_strong_wolfe", Method::bfgs()),
        ("lbfgs_strong_wolfe", Method::lbfgs()),
    ] {
        group.bench_with_input(
            BenchmarkId::from_parameter(method_name),
            &method,
            |b, &method| {
                b.iter(|| {
                    let mut oracle = Rosenbrock;
                    let mut solver = fresh_solver(method, LineSearch::StrongWolfe, 2);
                    let mut x = vec![-1.2, 1.0];
                    let _ = solver.run(&mut oracle, &mut x);
                });
            },
        );
    }
    group.finish();
}

fn wall_clock_badly_scaled(c: &mut Criterion) {
    let mut group = c.benchmark_group("badly_scaled_wall");
    group
        .sample_size(60)
        .warm_up_time(Duration::from_millis(200))
        .measurement_time(Duration::from_secs(2));

    let alphas = vec![1e-3, 1e-1, 1.0, 1e1, 1e3];
    for (method_name, method) in [("bfgs", Method::bfgs()), ("lbfgs", Method::lbfgs())] {
        group.bench_with_input(
            BenchmarkId::from_parameter(method_name),
            &method,
            |b, &method| {
                b.iter(|| {
                    let inner = AnisotropicQuadratic { alphas: alphas.clone() };
                    let mut solver = fresh_solver(method, LineSearch::StrongWolfe, 5);
                    let mut x = vec![1.0, 1.0, 1.0, 1.0, 1.0];
                    let mut oracle = inner;
                    let _ = solver.run(&mut oracle, &mut x);
                });
            },
        );
    }
    group.finish();
}

// ==============================================================
// Group + main registration.
// ==============================================================

criterion_group!(
    name    = algorithmic;
    config  = Criterion::default()
                 .with_measurement(EvalCountMeasurement);
    targets = counts_rosenbrock, counts_badly_scaled
);

criterion_group!(
    name    = systems;
    config  = Criterion::default();
    targets = wall_clock_rosenbrock, wall_clock_badly_scaled
);

criterion_main!(algorithmic, systems);
