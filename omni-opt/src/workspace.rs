//! Pre-allocated workspace for the limited-memory BFGS two-loop
//! recursion.

use core::ops::Range;

/// Pre-allocated two-loop-recursion workspace for L-BFGS with
/// memory parameter `m` on an `n`-dimensional optimization
/// variable.
///
/// # Allocation policy
///
/// [`LBFGSWorkspace::new`] performs **exactly four** heap
/// allocations (`s`, `y`, `rho`, `alpha`). No further allocation
/// occurs on any subsequent method; history is maintained as a
/// ring buffer over the same flat storage, satisfying the Phase-1
/// zero-allocation invariant.
///
/// # Layout
///
/// `s` and `y` are stored as flat `m · n` row-major buffers,
/// **not** as `Vec<Vec<f64>>`. Benefits:
///
/// * one heap allocation per array, not `m + 1`;
/// * each history slot is a contiguous `&[f64]` of length `n`,
///   maximally cache-friendly for the dot products that dominate
///   the two-loop recursion;
/// * slot indexing collapses to `slice[k*n .. (k+1)*n]` — one
///   pointer add, zero indirection.
///
/// # Ring semantics
///
/// * `head` is the index of the **next slot to overwrite** (i.e.,
///   the oldest pair once `count == m`).
/// * `count` saturates at `m`; needed because the first `m`
///   iterations run with a partial history, and because pairs
///   failing the curvature condition (`yᵀs ≤ 0`) are skipped
///   without advancing `count` past a valid frontier.
///
/// # Write / commit split
///
/// Callers fill the next slot via [`LBFGSWorkspace::head_s_mut`]
/// and [`LBFGSWorkspace::head_y_mut`], then commit with
/// [`LBFGSWorkspace::advance`]. A caller that fails the curvature
/// check after writing can simply **not** call `advance` — the
/// ring cursor and pair count stay intact, avoiding torn state.
pub struct LBFGSWorkspace {
    n: usize,
    m: usize,

    // Row-major `m × n` flat buffers. Physical slot p ∈ [0, m)
    // occupies `[p*n .. (p+1)*n]`.
    s: Vec<f64>,
    y: Vec<f64>,

    // One scalar per stored pair.
    rho: Vec<f64>,

    // Two-loop recursion scratch, length `m`, indexed by logical
    // slot. Lives on the workspace (not on the stack) so `m` can
    // be chosen at runtime without const generics or `alloca`.
    alpha: Vec<f64>,

    head: usize,
    count: usize,
}

impl LBFGSWorkspace {
    /// Default history depth for L-BFGS.
    ///
    /// `m = 10` matches SciPy and the majority of production
    /// L-BFGS-B implementations; it balances the `2 · m · n`
    /// memory footprint against curvature-approximation fidelity
    /// on highly non-linear objectives.
    pub const DEFAULT_M: usize = 10;

    /// Allocate a fresh workspace.
    ///
    /// # Panics
    ///
    /// Panics if `n == 0`, if `m == 0`, or if the ring-buffer
    /// length `n · m` overflows `usize`. Overflow is detected
    /// **before** any allocation so the failure mode is a clean
    /// named panic rather than a wrapped length that would later
    /// turn `head_*_mut` into an out-of-bounds slice index. All
    /// downstream arithmetic (`slot_range_physical`) assumes
    /// `n · m ≤ usize::MAX`, which is exactly what this check
    /// establishes.
    pub fn new(n: usize, m: usize) -> Self {
        assert!(n > 0, "LBFGSWorkspace: n must be > 0");
        assert!(m > 0, "LBFGSWorkspace: m must be > 0");
        let len = n
            .checked_mul(m)
            .expect("LBFGSWorkspace::new: n * m overflowed usize");
        Self {
            n,
            m,
            s: vec![0.0; len],
            y: vec![0.0; len],
            rho: vec![0.0; m],
            alpha: vec![0.0; m],
            head: 0,
            count: 0,
        }
    }

    /// Optimization-variable dimension passed to [`Self::new`].
    #[inline]
    pub fn n(&self) -> usize {
        self.n
    }

    /// History capacity passed to [`Self::new`].
    #[inline]
    pub fn m(&self) -> usize {
        self.m
    }

    /// Number of currently valid `(s, y, ρ)` pairs. Saturates at
    /// `m`.
    #[inline]
    pub fn count(&self) -> usize {
        self.count
    }

    /// Physical ring index of the next slot to overwrite.
    #[inline]
    pub fn head(&self) -> usize {
        self.head
    }

    /// Translate a logical slot `k ∈ [0, count)` — where `0` is
    /// the oldest valid pair — into a physical ring index.
    ///
    /// While the ring is partially filled (`count < m`) the
    /// oldest pair sits at physical slot `0`; once full, the
    /// oldest pair sits at `head`.
    #[inline]
    fn physical(&self, k: usize) -> usize {
        debug_assert!(
            k < self.count,
            "LBFGSWorkspace: logical slot {} out of range (count = {})",
            k,
            self.count
        );
        if self.count < self.m {
            k
        } else {
            (self.head + k) % self.m
        }
    }

    /// Compute the half-open element range occupied by physical
    /// slot `p` inside the `s` / `y` flat buffers.
    ///
    /// The `debug_assert!` is the single boundary check that every
    /// slot accessor funnels through; keeping it on this helper
    /// (rather than duplicating the check at every call site)
    /// makes the invariant auditable in one place. Release builds
    /// stay branchless because `debug_assert!` is stripped.
    ///
    /// Safety of the arithmetic: `new` establishes
    /// `n · m ≤ usize::MAX`, and `p < m` is asserted here, so
    /// `p · n ≤ n · m` and `(p + 1) · n ≤ n · m` cannot overflow.
    #[inline]
    fn slot_range_physical(&self, p: usize) -> Range<usize> {
        debug_assert!(
            p < self.m,
            "LBFGSWorkspace: physical slot {} out of range (m = {})",
            p,
            self.m
        );
        p * self.n..(p + 1) * self.n
    }

    /// Immutable view of history slot `k` (`0` = oldest valid).
    ///
    /// The returned slice aliases the ring buffer — no copy.
    pub fn s_slot(&self, k: usize) -> &[f64] {
        let r = self.slot_range_physical(self.physical(k));
        &self.s[r]
    }

    /// Immutable view of gradient-difference slot `k` (`0` =
    /// oldest valid).
    pub fn y_slot(&self, k: usize) -> &[f64] {
        let r = self.slot_range_physical(self.physical(k));
        &self.y[r]
    }

    /// Curvature scalar `ρ_k = 1 / (yᵀ s)` for logical slot `k`.
    pub fn rho_at(&self, k: usize) -> f64 {
        self.rho[self.physical(k)]
    }

    /// Read-only view of the raw `ρ` ring (indexed by *physical*
    /// slot). Intended for diagnostics; solver code should prefer
    /// [`Self::rho_at`].
    pub fn rho(&self) -> &[f64] {
        &self.rho
    }

    /// Mutable slice into the slot currently at the ring head —
    /// i.e., the position-difference slot being written this
    /// iteration.
    ///
    /// # Usage
    ///
    /// 1. Fill this slice and [`Self::head_y_mut`] with `s_new` /
    ///    `y_new`.
    /// 2. Commit via [`Self::advance`] with the freshly computed
    ///    `ρ`.
    ///
    /// If the curvature condition fails (`yᵀs ≤ 0`), simply do
    /// **not** call [`Self::advance`]; the partially-written slot
    /// is untracked and will be overwritten next iteration.
    pub fn head_s_mut(&mut self) -> &mut [f64] {
        let r = self.slot_range_physical(self.head);
        &mut self.s[r]
    }

    /// Mutable slice into the gradient-difference slot currently
    /// at the ring head. See [`Self::head_s_mut`] for the
    /// write/commit protocol.
    pub fn head_y_mut(&mut self) -> &mut [f64] {
        let r = self.slot_range_physical(self.head);
        &mut self.y[r]
    }

    /// Commit the slot currently at `head`: record `rho_new`,
    /// advance the ring cursor, saturate `count` at `m`.
    ///
    /// Does not touch `s` or `y` — the caller is expected to have
    /// written them via [`Self::head_s_mut`] / [`Self::head_y_mut`]
    /// first.
    pub fn advance(&mut self, rho_new: f64) {
        debug_assert!(
            rho_new.is_finite(),
            "LBFGSWorkspace::advance: rho_new must be finite"
        );
        self.rho[self.head] = rho_new;
        self.head = (self.head + 1) % self.m;
        if self.count < self.m {
            self.count += 1;
        }
    }

    /// Drop all history. Does **not** free the underlying storage
    /// — only resets `head` and `count`, so the next iteration
    /// starts from a clean slate with zero allocation cost.
    pub fn reset(&mut self) {
        self.head = 0;
        self.count = 0;
    }

    /// Mutable scratch of length `m` for the first pass of the
    /// two-loop recursion. Indexed by logical slot.
    pub fn alpha_mut(&mut self) -> &mut [f64] {
        &mut self.alpha
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[should_panic]
    fn new_rejects_zero_n() {
        let _ = LBFGSWorkspace::new(0, 5);
    }

    #[test]
    #[should_panic]
    fn new_rejects_zero_m() {
        let _ = LBFGSWorkspace::new(5, 0);
    }

    #[test]
    #[should_panic(expected = "overflowed usize")]
    fn new_rejects_n_times_m_overflow() {
        // Pick an (n, m) pair whose product is guaranteed to
        // wrap on every 64-bit target without allocating
        // anything — `checked_mul` fails first.
        let _ = LBFGSWorkspace::new(usize::MAX, 2);
    }

    #[test]
    fn fresh_workspace_is_empty() {
        let ws = LBFGSWorkspace::new(4, 3);
        assert_eq!(ws.n(), 4);
        assert_eq!(ws.m(), 3);
        assert_eq!(ws.count(), 0);
        assert_eq!(ws.head(), 0);
    }

    #[test]
    fn write_then_commit_round_trip() {
        let mut ws = LBFGSWorkspace::new(3, 4);

        ws.head_s_mut().copy_from_slice(&[1.0, 2.0, 3.0]);
        ws.head_y_mut().copy_from_slice(&[4.0, 5.0, 6.0]);
        ws.advance(0.5);

        assert_eq!(ws.count(), 1);
        assert_eq!(ws.head(), 1);
        assert_eq!(ws.s_slot(0), &[1.0, 2.0, 3.0]);
        assert_eq!(ws.y_slot(0), &[4.0, 5.0, 6.0]);
        assert_eq!(ws.rho_at(0), 0.5);
    }

    #[test]
    fn write_without_commit_preserves_state() {
        let mut ws = LBFGSWorkspace::new(3, 4);

        // Establish one valid pair.
        ws.head_s_mut().copy_from_slice(&[1.0, 1.0, 1.0]);
        ws.head_y_mut().copy_from_slice(&[2.0, 2.0, 2.0]);
        ws.advance(1.0);

        // Start writing a second pair, then abandon it (simulating
        // a curvature-condition failure).
        ws.head_s_mut().copy_from_slice(&[99.0, 99.0, 99.0]);
        ws.head_y_mut().copy_from_slice(&[99.0, 99.0, 99.0]);
        // NOTE: no `advance` call.

        // State must still reflect a single valid pair.
        assert_eq!(ws.count(), 1);
        assert_eq!(ws.head(), 1);
        assert_eq!(ws.s_slot(0), &[1.0, 1.0, 1.0]);
        assert_eq!(ws.rho_at(0), 1.0);
    }

    #[test]
    fn ring_wraps_and_orders_oldest_first() {
        let n = 2;
        let m = 3;
        let mut ws = LBFGSWorkspace::new(n, m);

        // Push m + 2 pairs; slot k's contents are [k as f64, 0].
        for k in 0..(m + 2) {
            ws.head_s_mut().copy_from_slice(&[k as f64, 0.0]);
            ws.head_y_mut().copy_from_slice(&[-(k as f64), 0.0]);
            ws.advance(k as f64 + 1.0);
        }

        assert_eq!(ws.count(), m);

        // After 5 pushes into a 3-slot ring, valid logical slots
        // are pushes 2, 3, 4 (oldest → newest).
        assert_eq!(ws.s_slot(0), &[2.0, 0.0]);
        assert_eq!(ws.s_slot(1), &[3.0, 0.0]);
        assert_eq!(ws.s_slot(2), &[4.0, 0.0]);

        assert_eq!(ws.rho_at(0), 3.0);
        assert_eq!(ws.rho_at(1), 4.0);
        assert_eq!(ws.rho_at(2), 5.0);
    }

    #[test]
    fn reset_clears_history_without_reallocating() {
        let mut ws = LBFGSWorkspace::new(2, 2);

        // Push a pair so we have something to reset.
        ws.head_s_mut().copy_from_slice(&[1.0, 2.0]);
        ws.head_y_mut().copy_from_slice(&[3.0, 4.0]);
        ws.advance(0.1);

        let s_ptr_after_push = ws.s_slot(0).as_ptr() as usize;
        ws.reset();

        assert_eq!(ws.count(), 0);
        assert_eq!(ws.head(), 0);

        // Re-push to prove underlying storage is reused (same
        // allocation — no reallocation on reset).
        ws.head_s_mut().copy_from_slice(&[9.0, 9.0]);
        ws.head_y_mut().copy_from_slice(&[9.0, 9.0]);
        ws.advance(0.2);

        let s_ptr_after_reset = ws.s_slot(0).as_ptr() as usize;
        assert_eq!(s_ptr_after_push, s_ptr_after_reset);
    }

    #[test]
    fn alpha_scratch_is_mutable_and_sized_m() {
        let mut ws = LBFGSWorkspace::new(5, 7);
        let alpha = ws.alpha_mut();
        assert_eq!(alpha.len(), 7);
        alpha[3] = 42.0;
        assert_eq!(ws.alpha_mut()[3], 42.0);
    }
}
