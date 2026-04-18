//! Zero-allocation abstraction seams.
//!
//! Every trait in this module is designed so that monomorphization
//! collapses to the same machine code as a hand-written `&[f64]`
//! kernel. Traits exist to let callers bring their own storage —
//! not to introduce vtables or copies.

/// Read-only view over contiguous `f64` storage.
///
/// # Memory contract
///
/// [`OmniVec::as_slice`] must return a slice that aliases the
/// underlying buffer for the lifetime of `&self`. Implementations
/// MUST NOT copy, reallocate, or lazily materialize on call — the
/// solver hot loop assumes O(1), zero-allocation access.
pub trait OmniVec {
    /// Number of elements.
    fn len(&self) -> usize;

    /// Bounds-checked scalar read.
    ///
    /// The default implementation routes through
    /// [`OmniVec::as_slice`] so implementors only need one method;
    /// bounds-check cost is deliberately left visible here. Hot
    /// loops should call [`OmniVec::as_slice`] once and index the
    /// returned slice directly (or use `unsafe` `get_unchecked`)
    /// rather than calling `get(i)` per element.
    #[inline]
    fn get(&self, i: usize) -> f64 {
        self.as_slice()[i]
    }

    /// Borrow the underlying contiguous buffer.
    ///
    /// Must be O(1) and must not allocate or copy.
    fn as_slice(&self) -> &[f64];

    /// `self.len() == 0`.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Mutable counterpart to [`OmniVec`].
///
/// Split from [`OmniVec`] so read-only consumers (e.g. a line
/// search evaluating `f(x + α·d)` against `&x`, `&d`) are not
/// forced to require `&mut`, which would otherwise poison
/// parallelism.
pub trait OmniVecMut: OmniVec {
    /// Mutable borrow of the underlying contiguous buffer.
    fn as_mut_slice(&mut self) -> &mut [f64];

    /// Bounds-checked scalar write. Same performance caveat as
    /// [`OmniVec::get`]: prefer `as_mut_slice` for hot loops.
    #[inline]
    fn set(&mut self, i: usize, v: f64) {
        self.as_mut_slice()[i] = v;
    }
}

// ---------------------------------------------------------------
// Blanket implementations.
//
// All three backends delegate to the same underlying slice view,
// so after monomorphization the generated code is identical
// across `&[f64]`, `Vec<f64>`, and `faer::Col<f64>` callers.
// ---------------------------------------------------------------

impl OmniVec for [f64] {
    #[inline]
    fn len(&self) -> usize {
        <[f64]>::len(self)
    }

    #[inline]
    fn as_slice(&self) -> &[f64] {
        self
    }
}

impl OmniVecMut for [f64] {
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [f64] {
        self
    }
}

impl OmniVec for &[f64] {
    #[inline]
    fn len(&self) -> usize {
        <[f64]>::len(self)
    }

    #[inline]
    fn as_slice(&self) -> &[f64] {
        self
    }
}

impl OmniVec for &mut [f64] {
    #[inline]
    fn len(&self) -> usize {
        <[f64]>::len(self)
    }

    #[inline]
    fn as_slice(&self) -> &[f64] {
        self
    }
}

impl OmniVecMut for &mut [f64] {
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [f64] {
        self
    }
}

impl OmniVec for Vec<f64> {
    #[inline]
    fn len(&self) -> usize {
        Vec::len(self)
    }

    #[inline]
    fn as_slice(&self) -> &[f64] {
        Vec::as_slice(self)
    }
}

impl OmniVecMut for Vec<f64> {
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [f64] {
        Vec::as_mut_slice(self)
    }
}

#[cfg(feature = "faer")]
impl OmniVec for faer::Col<f64> {
    #[inline]
    fn len(&self) -> usize {
        self.nrows()
    }

    #[inline]
    fn as_slice(&self) -> &[f64] {
        // An owned `faer::Col<T>` is always column-major contiguous
        // (`ContiguousFwd` stride) by construction, so the
        // `try_as_col_major` probe is infallible here. We still
        // expect-rather-than-unwrap to leave a named panic site
        // for future API drift.
        self.try_as_col_major()
            .expect("faer::Col<f64> must be contiguous")
            .as_slice()
    }
}

#[cfg(feature = "faer")]
impl OmniVecMut for faer::Col<f64> {
    #[inline]
    fn as_mut_slice(&mut self) -> &mut [f64] {
        self.try_as_col_major_mut()
            .expect("faer::Col<f64> must be contiguous")
            .as_slice_mut()
    }
}

/// Streaming row-wise access to a matrix `A ∈ ℝ^{m × n}`.
///
/// Designed so that row-action methods (Kaczmarz, coordinate
/// descent, randomized projection, etc.) can operate on dense,
/// CSR, CSC, file-backed, or on-the-fly-generated matrices through
/// a single allocation-free interface.
///
/// The trait intentionally exposes the **operations** a solver
/// actually needs (`row_dot`, `row_sq_norm`, `axpy_row`) rather
/// than a `row(i) -> &[f64]` accessor: a sparse provider cannot
/// produce a contiguous row without allocating, so the operations
/// are pushed into the implementor where the native storage can
/// service them directly.
pub trait RowAccess {
    /// Number of rows (`m`).
    fn nrows(&self) -> usize;

    /// Number of columns (`n`).
    fn ncols(&self) -> usize;

    /// ⟨A[i, :], x⟩.
    ///
    /// # Preconditions
    ///
    /// `i < self.nrows()` and `x.len() == self.ncols()`.
    /// Implementations should enforce these with `debug_assert!`
    /// so release builds remain branchless.
    fn row_dot(&self, i: usize, x: &[f64]) -> f64;

    /// ‖A[i, :]‖₂².
    ///
    /// Implementations MAY cache the per-row squared norm
    /// (dense backends typically do, since it is used every
    /// Kaczmarz step); the trait does not mandate caching so
    /// that zero-allocation providers remain legal.
    fn row_sq_norm(&self, i: usize) -> f64;

    /// `y ← y + α · A[i, :]` in place.
    ///
    /// # Preconditions
    ///
    /// `i < self.nrows()` and `y.len() == self.ncols()`.
    /// No temporary row is ever materialized — this is the
    /// primary reason [`RowAccess`] exists.
    fn axpy_row(&self, i: usize, alpha: f64, y: &mut [f64]);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn slice_blanket_impl() {
        let v: &[f64] = &[1.0, 2.0, 3.0];
        assert_eq!(OmniVec::len(&v), 3);
        // Fully qualified: `slice::get` shadows the trait method.
        assert_eq!(<&[f64] as OmniVec>::get(&v, 1), 2.0);
        assert!(!OmniVec::is_empty(&v));
        assert_eq!(<&[f64] as OmniVec>::as_slice(&v), &[1.0, 2.0, 3.0]);
    }

    #[test]
    fn vec_blanket_impl() {
        let mut v: Vec<f64> = vec![0.0; 4];
        assert_eq!(OmniVec::len(&v), 4);
        OmniVecMut::set(&mut v, 2, 7.5);
        assert_eq!(<Vec<f64> as OmniVec>::get(&v, 2), 7.5);
        OmniVecMut::as_mut_slice(&mut v)[0] = -1.0;
        assert_eq!(
            <Vec<f64> as OmniVec>::as_slice(&v),
            &[-1.0, 0.0, 7.5, 0.0]
        );
    }

    #[test]
    fn mut_slice_blanket_impl() {
        let mut backing = [0.0f64; 3];
        let mut v: &mut [f64] = &mut backing;
        OmniVecMut::as_mut_slice(&mut v)[1] = 2.0;
        assert_eq!(<&mut [f64] as OmniVec>::as_slice(&v), &[0.0, 2.0, 0.0]);
    }

    #[test]
    fn empty_vec_is_empty() {
        let v: Vec<f64> = Vec::new();
        assert!(OmniVec::is_empty(&v));
    }
}
