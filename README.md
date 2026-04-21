# rust-scientific-framework

[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)
[![Rust Edition](https://img.shields.io/badge/rust-2021-orange)](https://doc.rust-lang.org/edition-guide/rust-2021/index.html)
[![MSRV](https://img.shields.io/badge/MSRV-1.75-informational)](https://blog.rust-lang.org/2023/12/28/Rust-1.75.0/)
[![Status](https://img.shields.io/badge/status-alpha-yellow)](#project-status)
[![Crates](https://img.shields.io/badge/crates-1%20of%204%20in%20dev-blueviolet)](#planned-crates)

**Completed crates** &nbsp;
[![omni-opt](https://img.shields.io/badge/omni--opt-pre--1.0%20feature%20complete-success?logo=rust)](./omni-opt/)

A pure-Rust scientific-computing framework aimed at making Rust a first-class
citizen for numerical, optimization, and statistical work in academia and
industry alike.

## Project status

The framework is under active development. Each crate in the table below
is developed independently and published separately; consumers may depend
on one or several.

| Crate                | Status                          | Notes                                      |
|----------------------|---------------------------------|--------------------------------------------|
| **omni-opt**         | Pre-1.0, feature-complete       | All five implementation phases shipped.    |
| **graph**            | Planned                         | Mathematical graphs + solvers.             |
| **stats**            | Planned                         | R-like statistics for Rust.                |
| **numerical-solver** | Planned                         | Native-Rust numerical analysis.            |

The **completed-crates** badge row at the top of this README gains a new
button each time a crate reaches its pre-1.0 milestone. Right now it
lists only **[omni-opt](./omni-opt/)**; the other three crates are scoped
but have not yet begun implementation.

## Planned crates

| Crate                | Description                                                                 |
|----------------------|-----------------------------------------------------------------------------|
| **omni-opt**         | An extensive, exhaustive library of optimization algorithms.                |
| **graph**            | Mathematical graph modeling together with a suite of graph solvers.         |
| **stats**            | An R-like statistics library for Rust.                                      |
| **numerical-solver** | A native-Rust numerical-analysis solver.                                    |

## `omni-opt` — architecture and implementations

`omni-opt` is a pure-Rust optimization framework built under a strict
**zero-allocation invariant**: every heap allocation happens during the
builder / constructor phase, and the per-iteration hot loop of every
solver is allocation-free. See the [crate-level README](./omni-opt/) for
installation, quickstart, and full API surface.

### Five-phase architecture

| Phase | Theme                              | Public surfaces                                                                                                              |
|-------|------------------------------------|------------------------------------------------------------------------------------------------------------------------------|
| 1     | Abstraction layer                  | `OmniVec` / `OmniVecMut`, `RowAccess`, `LBFGSWorkspace`, `StoppingCriteria`, `TerminationStatus`                             |
| 2     | Oracle & line-search foundation    | `Objective` / `Oracle` / `HessianOracle`, `LineSearch` (Strong Wolfe / Armijo), `central_difference`, `CentralDifferenceOracle` |
| 3     | Unconstrained descent engine       | `Solver` / `SolverBuilder` / `SolverReport`, `Method` (SteepestDescent / BFGS / L-BFGS / Newton+LM), `MethodError`           |
| 4     | Linear & streaming subsystem       | `linear::solve` (QR / col-piv QR / SVD, **no normal equations**), `kaczmarz::run` over `RowAccess`, xorshift64 RNG            |
| 5     | Constraints, pathology & hardening | `BoxConstraints` with projected clipping, `tests/pathology.rs`, dual-axis criterion benches                                  |

### Core features

- **Storage-generic kernels.** `OmniVec` / `OmniVecMut` let callers bring
  their own backing buffer (`[f64]`, `Vec<f64>`, `faer::Col<f64>` under
  the `faer` feature). Monomorphization collapses every kernel to the
  same machine code as a hand-written `&[f64]` loop.
- **Descent methods.** SteepestDescent; dense BFGS with γ-scaled
  warm-start and strict curvature safeguard; limited-memory BFGS with
  flat `m · n` ring buffer and two-loop recursion; exact Newton with
  Levenberg–Marquardt damping on non-PSD Hessians (`faer` feature).
- **Line searches.** Strong-Wolfe bracketing + safeguarded quadratic
  zoom (Nocedal & Wright Alg. 3.5 / 3.6) and Backtracking Armijo with
  geometric shrink. Both drive from the same pre-allocated
  `LineSearchWorkspace`.
- **Finite-difference fallback.** `CentralDifferenceOracle<O>` promotes
  any `Objective` to a full `Oracle` via in-place perturbation of a
  single caller-owned scratch buffer — `2n + 1` value evaluations, zero
  hot-path allocations.
- **Dense orthogonal least squares.** `linear::solve` wires faer's
  Householder QR, column-pivoted QR (default, rank-revealing), and
  thin SVD. Conditioning stays at `κ(A)`; the normal-equations path is
  deliberately absent.
- **Streaming Kaczmarz.** `kaczmarz::run` operates entirely through
  the `RowAccess` trait — dense / CSR / CSC / file-backed /
  generator-style matrices plug in unchanged. Uniform,
  Strohmer–Vershynin, and cyclic row sampling; deterministic xorshift64
  RNG seeded from `cfg.seed`.
- **Box constraints.** `BoxConstraints` with projected clipping
  integrated into the descent loop: infeasible `x₀` silently projected,
  reduced-gradient convergence test under bounds, gradient refresh only
  when a coordinate was actually clipped. Documented as projected
  clipping (not full L-BFGS-B).
- **Structured termination.** `Copy`, heap-free exit enums
  (`TerminationStatus`, `LineSearchError`, `LinearSolveError`,
  `MethodError`) — the exit path itself allocates nothing.

### Quality gates

- Unit + integration tests covering every public surface, including a
  pathology suite (Rosenbrock, badly-scaled quadratics, saddle points,
  noisy finite-difference gradients).
- Clippy clean at `--all-features --all-targets -- -D warnings`.
- Dual-axis criterion benches — a custom `EvalCountMeasurement` reports
  function / gradient evaluation counts with full statistical machinery,
  alongside a standard wall-clock group on the same problems.
- Documented scope limits: projected clipping is called out in the
  module, the struct, the builder method, and this README as distinct
  from full L-BFGS-B; normal-equations least squares is explicitly
  absent.

## Organizers

This project is organized and maintained by:

- Leonard S. Wang — [@sunhaoxiangwang](https://github.com/sunhaoxiangwang)
- Michael Mansour — [@Mike-Mans](https://github.com/Mike-Mans)
- Allen Chu — [@forajii](https://github.com/forajii)
- Anibal Lopez-Anguiano — [@Anibal-ML](https://github.com/Anibal-ML)

## License

This project is dual-licensed under either of

- Apache License, Version 2.0 — see [LICENSE-APACHE](./LICENSE-APACHE)
- MIT License — see [LICENSE-MIT](./LICENSE-MIT)

at your option.

Copyright © 2026 SUM INNOVATION INC.

### Contribution

Unless you explicitly state otherwise, any contribution intentionally
submitted for inclusion in this project by you, as defined in the
Apache-2.0 license, shall be dual-licensed as above, without any
additional terms or conditions.
