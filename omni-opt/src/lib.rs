//! # omni-opt
//!
//! A pure-Rust, high-performance mathematical optimization framework
//! designed under a strict zero-allocation invariant: every heap
//! allocation occurs during the builder / construction phase, and
//! the per-iteration hot loop of every solver is allocation-free.
//!
//! Phase 1 establishes the abstraction layer:
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

pub mod state;
pub mod traits;
pub mod workspace;

pub use state::{StoppingCriteria, TerminationStatus};
pub use traits::{OmniVec, OmniVecMut, RowAccess};
pub use workspace::LBFGSWorkspace;
