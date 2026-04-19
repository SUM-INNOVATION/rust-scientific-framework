//! # omni-opt
//!
//! A pure-Rust, high-performance mathematical optimization framework
//! designed under a strict zero-allocation invariant: every heap
//! allocation occurs during the builder / construction phase, and
//! the per-iteration hot loop of every solver is allocation-free.
//!
//! Phase 1 — abstraction layer:
//!
//! * [`OmniVec`] / [`OmniVecMut`] — generic read/write access over
//!   contiguous `f64` storage (`&[f64]`, `Vec<f64>`, optionally
//!   `faer::Col<f64>` behind the `faer` feature).
//! * [`RowAccess`] — streaming row-wise matrix access for
//!   row-action solvers (Kaczmarz, etc.) that never materializes a
//!   full row.
//! * [`LBFGSWorkspace`] — pre-allocated ring-buffer workspace for
//!   the L-BFGS two-loop recursion.
//! * [`StoppingCriteria`] / [`TerminationStatus`] — user-facing
//!   configuration and structured, non-allocating exit reporting.
//!
//! Phase 2 — oracle and line-search foundation:
//!
//! * [`Objective`] / [`Oracle`] — storage-generic evaluation seams
//!   with fused `value_grad` for analytical gradients.
//! * [`LineSearch`] / [`LineSearchConfig`] / [`LineSearchWorkspace`]
//!   / [`run`] — Strong-Wolfe and Armijo line searches with
//!   caller-owned trial / trial-gradient buffers.
//! * [`central_difference`] and [`CentralDifferenceOracle`] —
//!   zero-allocation finite-difference fallback and `Objective → Oracle`
//!   adapter.
//!
//! [`Objective`]: oracle::Objective
//! [`Oracle`]: oracle::Oracle
//! [`LineSearch`]: line_search::LineSearch
//! [`LineSearchConfig`]: line_search::LineSearchConfig
//! [`LineSearchWorkspace`]: line_search::LineSearchWorkspace
//! [`run`]: line_search::run
//! [`central_difference`]: gradient::central_difference
//! [`CentralDifferenceOracle`]: gradient::CentralDifferenceOracle

pub mod constraints;
pub mod gradient;
pub mod kaczmarz;
pub mod line_search;
#[cfg(feature = "faer")]
pub mod linear;
pub mod methods;
pub mod oracle;
pub mod solver;
pub mod state;
pub mod traits;
pub mod workspace;

pub use constraints::BoxConstraints;
pub use gradient::{central_difference, CentralDifferenceOracle};
pub use kaczmarz::{
    KaczmarzConfig, KaczmarzReport, KaczmarzSampling, KaczmarzWorkspace,
};
pub use line_search::{
    LineSearch, LineSearchConfig, LineSearchError, LineSearchStep, LineSearchWorkspace,
};
#[cfg(feature = "faer")]
pub use linear::{
    LinearSolveError, LinearSolveReport, LinearSolver, LinearSolverWorkspace,
};
pub use methods::{Method, MethodError};
pub use oracle::{Objective, Oracle};
#[cfg(feature = "faer")]
pub use oracle::HessianOracle;
pub use solver::{Solver, SolverBuilder, SolverReport};
pub use state::{StoppingCriteria, TerminationStatus};
pub use traits::{OmniVec, OmniVecMut, RowAccess};
pub use workspace::LBFGSWorkspace;
