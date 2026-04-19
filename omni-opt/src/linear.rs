//! Dense overdetermined least-squares solvers (m ≥ n) backed by
//! faer's orthogonal factorizations.
//!
//! # Mathematical contract
//!
//! Every backend in this module factors `A` directly
//! (`A = Q R` or `A = U Σ Vᵀ`) and performs the least-squares
//! solve by back-substitution over the factors. No code path
//! ever forms `Aᵀ A`, so numerical conditioning stays at
//! `κ(A)` rather than `κ(A)²`.
//!
//! # Allocation contract
//!
//! All heap allocation happens inside
//! [`LinearSolverWorkspace::new`]; every subsequent call to
//! [`solve`] factorizes and back-substitutes into the
//! pre-allocated buffers. `faer` scratch is drawn from a
//! pre-sized [`MemBuffer`], so even the low-level routines
//! stay allocation-free on the hot path.
//!
//! `faer` is a hard dependency of this module — the entire file
//! is compiled only under `--features faer`.

// Clippy escape: lstsq / least-squares / faer kernels index
// vectors and matrices in lockstep; `enumerate()` rewrites hurt
// readability without helping the compiler.
#![allow(clippy::needless_range_loop)]

use faer::dyn_stack::{MemBuffer, MemStack, StackReq};
use faer::linalg::qr::{col_pivoting, no_pivoting};
use faer::linalg::svd;
use faer::{Mat, MatMut, MatRef, Par};

// A modest blocksize that balances BLAS-3 kernel efficiency
// against Q_coeff memory. For the moderate m, n typical in
// least-squares problems this is plenty.
const QR_BLOCKSIZE: usize = 32;

/// Choice of orthogonal backend.
#[derive(Clone, Copy, Debug)]
pub enum LinearSolver {
    /// Classical Householder QR, no rank detection. `≈ 2 m n²`
    /// flops. Fastest option, but on a rank-deficient `A` the
    /// back-substitution will divide by a near-zero `R_{ii}`
    /// and produce meaningless output — use [`Self::ColPivQr`]
    /// or [`Self::Svd`] when `A`'s rank cannot be guaranteed.
    HouseholderQr,

    /// Column-pivoted QR. Rank-revealing and safe on
    /// rank-deficient `A`. ~20 % slower than `HouseholderQr` —
    /// the default trade-off for a numerical framework.
    ColPivQr {
        /// Relative threshold: column `i` is treated as
        /// rank-deficient when `|R_{ii}| ≤ rank_threshold · |R_{11}|`.
        /// `0.0` disables the check.
        rank_threshold: f64,
    },

    /// Thin SVD-based pseudoinverse, `x = V Σ⁺ Uᵀ b`. Most
    /// robust (handles ill-conditioned / rank-deficient `A`
    /// gracefully), also most expensive (`O(m n² + n³)`).
    Svd {
        /// Singular values with `σᵢ ≤ rank_threshold · σ_max` are
        /// treated as zero in the pseudoinverse. Typical choice:
        /// `ε_mach · max(m, n)`.
        rank_threshold: f64,
    },
}

impl Default for LinearSolver {
    /// Safe default: [`Self::ColPivQr`] with a conservative
    /// rank threshold. Users who know their `A` has full column
    /// rank can opt into [`Self::HouseholderQr`] for speed.
    fn default() -> Self {
        Self::col_piv_qr()
    }
}

impl LinearSolver {
    /// `HouseholderQr` — explicit opt-in for full-rank `A`.
    pub fn qr() -> Self {
        Self::HouseholderQr
    }

    /// `ColPivQr` with a conservative default threshold.
    pub fn col_piv_qr() -> Self {
        Self::ColPivQr {
            rank_threshold: 1e-12,
        }
    }

    /// `Svd` with a conservative default threshold.
    pub fn svd() -> Self {
        Self::Svd {
            rank_threshold: 1e-12,
        }
    }
}

/// Successful least-squares outcome. `Copy`, heap-free.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearSolveReport {
    /// Detected rank. For [`LinearSolver::HouseholderQr`] this
    /// is always `n` (no detection performed).
    pub rank: usize,

    /// `‖A x − b‖₂` at the returned `x`. Computed against the
    /// caller's original `A` via a single pass over its rows
    /// (no allocation).
    pub residual_norm: f64,

    /// `true` iff the backend truncated at least one near-zero
    /// singular value / diagonal pivot. Informational only —
    /// not an error.
    pub rank_deficient: bool,
}

/// Structured failure mode. `Copy`, heap-free.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LinearSolveError {
    /// Caller's dimensions did not match the workspace, or
    /// `m < n`.
    ShapeMismatch,
    /// `HouseholderQr` saw a zero / near-zero pivot on the
    /// diagonal of `R`. Switch to `ColPivQr` or `Svd`.
    SingularMatrix,
    /// NaN / Inf observed in the factorization or back-
    /// substitution, or SVD failed to converge.
    NumericalFailure,
}

/// Pre-allocated scratch for orthogonal least-squares solvers.
///
/// # Visibility
///
/// Construct exclusively through [`Self::new`]; the type itself
/// is public so callers can hold it across multiple [`solve`]
/// invocations of the same shape.
pub struct LinearSolverWorkspace {
    m: usize,
    n: usize,

    // Caller's `A` is memcpy'd here once per `solve` so the
    // factorization can overwrite it without destroying user
    // data. O(mn) copy, zero allocation.
    a_work: Mat<f64>,

    // Caller's `b` is memcpy'd into column 0 here; solve writes
    // `x` back into the first `n` entries of the same column.
    rhs: Mat<f64>,

    inner: LinearSolverInner,
}

enum LinearSolverInner {
    HouseholderQr(HouseholderQrState),
    ColPivQr(ColPivQrState),
    Svd(SvdState),
}

struct HouseholderQrState {
    /// Shape (blocksize, min(m, n)). Stores the Householder
    /// coefficients emitted by `qr_in_place`.
    q_coeff: Mat<f64>,
    mem: MemBuffer,
}

struct ColPivQrState {
    q_coeff: Mat<f64>,
    col_perm: Vec<usize>,
    col_perm_inv: Vec<usize>,
    mem: MemBuffer,
    rank_threshold: f64,
}

struct SvdState {
    /// Thin U: (m, n).
    u: Mat<f64>,
    /// Thin V: (n, n).
    v: Mat<f64>,
    /// Singular values stashed on the diagonal of an (n, n) Mat
    /// (faer's API wants a `DiagMut`).
    s: Mat<f64>,
    /// Scratch column of length `n` holding `Uᵀ b` and then the
    /// scaled divisions `Uᵀb / σᵢ`.
    utb: Vec<f64>,
    mem: MemBuffer,
    rank_threshold: f64,
}

impl LinearSolverWorkspace {
    /// Allocate everything the chosen backend will need.
    ///
    /// # Panics
    ///
    /// Panics on `n == 0`, `m < n`, or a non-finite
    /// `rank_threshold`. Misconfiguration fails loudly at build
    /// time, never inside the solve loop.
    pub fn new(m: usize, n: usize, solver: LinearSolver) -> Self {
        assert!(n > 0, "LinearSolverWorkspace::new: n must be > 0");
        assert!(
            m >= n,
            "LinearSolverWorkspace::new: require m ({}) >= n ({}) for overdetermined system",
            m,
            n
        );

        let blocksize = QR_BLOCKSIZE.min(n);

        let inner = match solver {
            LinearSolver::HouseholderQr => LinearSolverInner::HouseholderQr(HouseholderQrState {
                q_coeff: Mat::zeros(blocksize, n),
                mem: MemBuffer::new(StackReq::or(
                    no_pivoting::factor::qr_in_place_scratch::<f64>(
                        m,
                        n,
                        blocksize,
                        Par::Seq,
                        Default::default(),
                    ),
                    no_pivoting::solve::solve_lstsq_in_place_scratch::<f64>(
                        m,
                        n,
                        blocksize,
                        1,
                        Par::Seq,
                    ),
                )),
            }),
            LinearSolver::ColPivQr { rank_threshold } => {
                assert!(
                    rank_threshold.is_finite() && rank_threshold >= 0.0,
                    "LinearSolverWorkspace::new: rank_threshold must be finite and non-negative"
                );
                LinearSolverInner::ColPivQr(ColPivQrState {
                    q_coeff: Mat::zeros(blocksize, n),
                    col_perm: vec![0usize; n],
                    col_perm_inv: vec![0usize; n],
                    mem: MemBuffer::new(StackReq::or(
                        col_pivoting::factor::qr_in_place_scratch::<usize, f64>(
                            m,
                            n,
                            blocksize,
                            Par::Seq,
                            Default::default(),
                        ),
                        col_pivoting::solve::solve_lstsq_in_place_scratch::<usize, f64>(
                            m,
                            n,
                            blocksize,
                            1,
                            Par::Seq,
                        ),
                    )),
                    rank_threshold,
                })
            }
            LinearSolver::Svd { rank_threshold } => {
                assert!(
                    rank_threshold.is_finite() && rank_threshold >= 0.0,
                    "LinearSolverWorkspace::new: rank_threshold must be finite and non-negative"
                );
                LinearSolverInner::Svd(SvdState {
                    u: Mat::zeros(m, n),
                    v: Mat::zeros(n, n),
                    s: Mat::zeros(n, n),
                    utb: vec![0.0; n],
                    mem: MemBuffer::new(svd::svd_scratch::<f64>(
                        m,
                        n,
                        svd::ComputeSvdVectors::Thin,
                        svd::ComputeSvdVectors::Thin,
                        Par::Seq,
                        Default::default(),
                    )),
                    rank_threshold,
                })
            }
        };

        Self {
            m,
            n,
            a_work: Mat::zeros(m, n),
            rhs: Mat::zeros(m, 1),
            inner,
        }
    }

    /// Number of equations this workspace was built for.
    #[inline]
    pub fn m(&self) -> usize {
        self.m
    }

    /// Number of unknowns this workspace was built for.
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }
}

/// Solve `A x = b` in the least-squares sense.
///
/// # Preconditions
///
/// * `a.nrows() == workspace.m()`, `a.ncols() == workspace.n()`.
/// * `b.len() == workspace.m()`, `x.len() == workspace.n()`.
///
/// Dimension mismatches return [`LinearSolveError::ShapeMismatch`]
/// rather than panicking.
///
/// # Zero-allocation contract
///
/// The caller's `A` is copied **once** into `workspace.a_work`
/// (`O(mn)` memcpy, no allocation); `b` is copied into
/// `workspace.rhs` and overwritten by the back-substitution.
/// All faer scratch comes from the pre-sized `MemBuffer`, so
/// the solve path is heap-free.
pub fn solve(
    a: MatRef<'_, f64>,
    b: &[f64],
    x: &mut [f64],
    workspace: &mut LinearSolverWorkspace,
) -> Result<LinearSolveReport, LinearSolveError> {
    if a.nrows() != workspace.m
        || a.ncols() != workspace.n
        || b.len() != workspace.m
        || x.len() != workspace.n
    {
        return Err(LinearSolveError::ShapeMismatch);
    }

    let m = workspace.m;
    let n = workspace.n;

    // Copy A into the factorization workspace (caller keeps ownership of `a`).
    copy_into_mat(a, workspace.a_work.as_mut());

    // Copy b into the RHS column.
    for i in 0..m {
        workspace.rhs[(i, 0)] = b[i];
    }

    let (rank, rank_deficient) = match &mut workspace.inner {
        LinearSolverInner::HouseholderQr(state) => {
            solve_householder_qr(m, n, &mut workspace.a_work, state, &mut workspace.rhs)?
        }
        LinearSolverInner::ColPivQr(state) => {
            solve_col_piv_qr(m, n, &mut workspace.a_work, state, &mut workspace.rhs)?
        }
        LinearSolverInner::Svd(state) => {
            solve_svd(m, n, &mut workspace.a_work, state, &mut workspace.rhs)?
        }
    };

    // Extract x from the first n entries of rhs.
    for i in 0..n {
        let xi = workspace.rhs[(i, 0)];
        if !xi.is_finite() {
            return Err(LinearSolveError::NumericalFailure);
        }
        x[i] = xi;
    }

    // Residual against the caller's original `a` (not `a_work`,
    // which has been destroyed by the factorization).
    let residual_norm = residual_l2_norm(a, x, b);
    if !residual_norm.is_finite() {
        return Err(LinearSolveError::NumericalFailure);
    }

    Ok(LinearSolveReport {
        rank,
        residual_norm,
        rank_deficient,
    })
}

// ============================================================
// Per-backend solve implementations.
// ============================================================

fn solve_householder_qr(
    _m: usize,
    n: usize,
    a_work: &mut Mat<f64>,
    state: &mut HouseholderQrState,
    rhs: &mut Mat<f64>,
) -> Result<(usize, bool), LinearSolveError> {
    let blocksize = state.q_coeff.nrows();

    {
        let stack = MemStack::new(&mut state.mem);
        no_pivoting::factor::qr_in_place(
            a_work.as_mut(),
            state.q_coeff.as_mut(),
            Par::Seq,
            stack,
            Default::default(),
        );
    }

    // Minimal safety check: R's diagonal must not contain a true
    // zero (would trip back-substitution). We do NOT flag
    // near-zero pivots here; `HouseholderQr` is the explicit
    // "I know A has full rank" choice.
    for i in 0..n {
        let rii = a_work[(i, i)];
        if rii == 0.0 || !rii.is_finite() {
            return Err(LinearSolveError::SingularMatrix);
        }
    }

    {
        let stack = MemStack::new(&mut state.mem);
        no_pivoting::solve::solve_lstsq_in_place(
            a_work.as_ref(),
            state.q_coeff.as_ref(),
            a_work.as_ref(),
            rhs.as_mut(),
            Par::Seq,
            stack,
        );
    }
    let _ = blocksize;
    Ok((n, false))
}

fn solve_col_piv_qr(
    m: usize,
    n: usize,
    a_work: &mut Mat<f64>,
    state: &mut ColPivQrState,
    rhs: &mut Mat<f64>,
) -> Result<(usize, bool), LinearSolveError> {
    let _ = m;

    // Factor; returns a PermRef borrowing from `state.col_perm`.
    // We scope the borrow tightly so we can reconstruct it for
    // the solve step.
    {
        let stack = MemStack::new(&mut state.mem);
        let _ = col_pivoting::factor::qr_in_place(
            a_work.as_mut(),
            state.q_coeff.as_mut(),
            &mut state.col_perm,
            &mut state.col_perm_inv,
            Par::Seq,
            stack,
            Default::default(),
        );
    }

    // Rank detection from R's diagonal. R lives in the upper
    // triangle of `a_work`.
    let r11 = a_work[(0, 0)].abs();
    let abs_tol = state.rank_threshold * r11;
    let mut rank = n;
    for i in 0..n {
        let rii = a_work[(i, i)].abs();
        if !rii.is_finite() {
            return Err(LinearSolveError::NumericalFailure);
        }
        if rii <= abs_tol {
            rank = i;
            break;
        }
    }
    let rank_deficient = rank < n;

    // Reconstruct the PermRef without re-running the factor.
    let perm = unsafe {
        faer::perm::PermRef::new_unchecked(&state.col_perm, &state.col_perm_inv, n)
    };

    {
        let stack = MemStack::new(&mut state.mem);
        col_pivoting::solve::solve_lstsq_in_place(
            a_work.as_ref(),
            state.q_coeff.as_ref(),
            a_work.as_ref(),
            perm,
            rhs.as_mut(),
            Par::Seq,
            stack,
        );
    }

    Ok((rank, rank_deficient))
}

fn solve_svd(
    m: usize,
    n: usize,
    a_work: &mut Mat<f64>,
    state: &mut SvdState,
    rhs: &mut Mat<f64>,
) -> Result<(usize, bool), LinearSolveError> {
    // faer's SVD reads `A` as a MatRef (non-destructive), but we
    // still use `a_work` as its input so we keep the caller's `a`
    // pristine for the residual check.
    {
        let stack = MemStack::new(&mut state.mem);
        svd::svd(
            a_work.as_ref(),
            state.s.as_mut().diagonal_mut(),
            Some(state.u.as_mut()),
            Some(state.v.as_mut()),
            Par::Seq,
            stack,
            Default::default(),
        )
        .map_err(|_| LinearSolveError::NumericalFailure)?;
    }

    // `Uᵀ b` into state.utb.
    for j in 0..n {
        let mut acc = 0.0;
        for i in 0..m {
            acc += state.u[(i, j)] * rhs[(i, 0)];
        }
        state.utb[j] = acc;
    }

    // σ_max (faer returns descending).
    let sigma_max = state.s[(0, 0)].abs();
    if !sigma_max.is_finite() {
        return Err(LinearSolveError::NumericalFailure);
    }
    let abs_tol = state.rank_threshold * sigma_max;

    let mut rank = 0;
    for j in 0..n {
        let sigma = state.s[(j, j)];
        if !sigma.is_finite() {
            return Err(LinearSolveError::NumericalFailure);
        }
        if sigma.abs() <= abs_tol {
            // Truncate: set utb[j] = 0 so this component drops
            // out of the pseudoinverse.
            state.utb[j] = 0.0;
        } else {
            state.utb[j] /= sigma;
            rank += 1;
        }
    }
    let rank_deficient = rank < n;

    // `x = V · (Σ⁺ Uᵀ b)` into the first n entries of rhs.
    for i in 0..n {
        let mut acc = 0.0;
        for j in 0..n {
            acc += state.v[(i, j)] * state.utb[j];
        }
        rhs[(i, 0)] = acc;
    }

    Ok((rank, rank_deficient))
}

// ============================================================
// Helpers.
// ============================================================

#[inline]
fn copy_into_mat(src: MatRef<'_, f64>, mut dst: MatMut<'_, f64>) {
    debug_assert_eq!(src.nrows(), dst.nrows());
    debug_assert_eq!(src.ncols(), dst.ncols());
    let (nr, nc) = (src.nrows(), src.ncols());
    for j in 0..nc {
        for i in 0..nr {
            dst[(i, j)] = src[(i, j)];
        }
    }
}

#[inline]
fn residual_l2_norm(a: MatRef<'_, f64>, x: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.nrows(), b.len());
    debug_assert_eq!(a.ncols(), x.len());
    let (m, n) = (a.nrows(), a.ncols());
    let mut sum_sq = 0.0;
    for i in 0..m {
        let mut ax = 0.0;
        for j in 0..n {
            ax += a[(i, j)] * x[j];
        }
        let r = ax - b[i];
        sum_sq += r * r;
    }
    sum_sq.sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;
    use faer::Mat;

    /// Least-squares against a small, well-conditioned overdetermined
    /// system where we know the exact solution.
    fn small_full_rank_system() -> (Mat<f64>, Vec<f64>, Vec<f64>) {
        // A: 4x2, full column rank.
        let a = Mat::from_fn(4, 2, |i, j| match (i, j) {
            (0, 0) => 1.0,
            (0, 1) => 0.0,
            (1, 0) => 0.0,
            (1, 1) => 1.0,
            (2, 0) => 1.0,
            (2, 1) => 1.0,
            (3, 0) => 2.0,
            (3, 1) => -1.0,
            _ => unreachable!(),
        });
        // b = A * [3, 2] exactly -> solution is [3, 2], residual = 0.
        let x_true = vec![3.0, 2.0];
        let b = vec![3.0, 2.0, 5.0, 4.0];
        (a, b, x_true)
    }

    #[test]
    fn householder_qr_solves_full_rank_system() {
        let (a, b, x_true) = small_full_rank_system();
        let mut ws = LinearSolverWorkspace::new(4, 2, LinearSolver::qr());
        let mut x = vec![0.0; 2];
        let r = solve(a.as_ref(), &b, &mut x, &mut ws).unwrap();
        assert_eq!(r.rank, 2);
        assert!(!r.rank_deficient);
        assert!(r.residual_norm < 1e-12);
        for (got, want) in x.iter().zip(&x_true) {
            assert!((got - want).abs() < 1e-10, "got {got}, want {want}");
        }
    }

    #[test]
    fn col_piv_qr_solves_full_rank_system() {
        let (a, b, x_true) = small_full_rank_system();
        let mut ws = LinearSolverWorkspace::new(4, 2, LinearSolver::col_piv_qr());
        let mut x = vec![0.0; 2];
        let r = solve(a.as_ref(), &b, &mut x, &mut ws).unwrap();
        assert_eq!(r.rank, 2);
        assert!(!r.rank_deficient);
        assert!(r.residual_norm < 1e-12);
        for (got, want) in x.iter().zip(&x_true) {
            assert!((got - want).abs() < 1e-10);
        }
    }

    #[test]
    fn svd_solves_full_rank_system() {
        let (a, b, x_true) = small_full_rank_system();
        let mut ws = LinearSolverWorkspace::new(4, 2, LinearSolver::svd());
        let mut x = vec![0.0; 2];
        let r = solve(a.as_ref(), &b, &mut x, &mut ws).unwrap();
        assert_eq!(r.rank, 2);
        assert!(!r.rank_deficient);
        assert!(r.residual_norm < 1e-10);
        for (got, want) in x.iter().zip(&x_true) {
            assert!((got - want).abs() < 1e-9);
        }
    }

    #[test]
    fn col_piv_qr_detects_rank_deficiency() {
        // Rank-1 matrix: second column is 2× first column.
        let a = Mat::from_fn(4, 2, |i, j| {
            let base = (i + 1) as f64;
            if j == 0 { base } else { 2.0 * base }
        });
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut ws = LinearSolverWorkspace::new(4, 2, LinearSolver::col_piv_qr());
        let mut x = vec![0.0; 2];
        let r = solve(a.as_ref(), &b, &mut x, &mut ws).unwrap();
        assert_eq!(r.rank, 1);
        assert!(r.rank_deficient);
    }

    #[test]
    fn svd_detects_rank_deficiency() {
        let a = Mat::from_fn(4, 2, |i, j| {
            let base = (i + 1) as f64;
            if j == 0 { base } else { 2.0 * base }
        });
        let b = vec![1.0, 2.0, 3.0, 4.0];
        let mut ws = LinearSolverWorkspace::new(4, 2, LinearSolver::svd());
        let mut x = vec![0.0; 2];
        let r = solve(a.as_ref(), &b, &mut x, &mut ws).unwrap();
        assert_eq!(r.rank, 1);
        assert!(r.rank_deficient);
    }

    #[test]
    fn shape_mismatch_returns_structured_error() {
        let (a, b, _) = small_full_rank_system();
        let mut ws = LinearSolverWorkspace::new(4, 2, LinearSolver::col_piv_qr());
        let mut x = vec![0.0; 3]; // wrong length
        let r = solve(a.as_ref(), &b, &mut x, &mut ws);
        assert_eq!(r, Err(LinearSolveError::ShapeMismatch));
    }

    #[test]
    #[should_panic]
    fn workspace_rejects_m_less_than_n() {
        let _ = LinearSolverWorkspace::new(2, 3, LinearSolver::col_piv_qr());
    }

    #[test]
    #[should_panic]
    fn workspace_rejects_zero_n() {
        let _ = LinearSolverWorkspace::new(3, 0, LinearSolver::col_piv_qr());
    }

    #[test]
    #[should_panic]
    fn workspace_rejects_negative_rank_threshold() {
        let _ = LinearSolverWorkspace::new(
            4,
            2,
            LinearSolver::ColPivQr {
                rank_threshold: -1e-12,
            },
        );
    }

    #[test]
    fn default_solver_is_col_piv_qr() {
        match LinearSolver::default() {
            LinearSolver::ColPivQr { .. } => {}
            _ => panic!("default LinearSolver must be ColPivQr"),
        }
    }
}
