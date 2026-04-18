# omni-opt

A pure-Rust, high-performance mathematical optimization framework built under
a strict **zero-allocation invariant**: every heap allocation happens during
the builder / constructor phase, and the per-iteration hot loop of every
solver is allocation-free.

> **Status: pre-1.0.** Phase 1 (this release) ships the abstraction layer —
> storage-generic traits, pre-allocated workspaces, and structured
> termination reporting. Solvers (L-BFGS, Kaczmarz, projected gradient, …)
> land in Phase 2+.

## Features

- **`OmniVec` / `OmniVecMut`** — storage-generic traits over contiguous
  `f64` buffers. Blanket impls for `[f64]`, `&[f64]`, `&mut [f64]`,
  `Vec<f64>`, and (behind the `faer` feature) `faer::Col<f64>`. Callers
  bring their own storage; monomorphization collapses each kernel to the
  same machine code as a hand-written `&[f64]` loop.
- **`RowAccess`** — streaming row-wise matrix trait for row-action methods
  (Kaczmarz, coordinate descent, randomized projection). Exposes
  `row_dot` / `row_sq_norm` / `axpy_row` directly, so dense / CSR / CSC /
  file-backed providers stay zero-alloc — no row is ever materialized.
- **`LBFGSWorkspace`** — pre-allocated ring-buffer workspace for the
  L-BFGS two-loop recursion. Four heap allocations in `new`, zero
  thereafter. Flat `m · n` row-major layout (no `Vec<Vec<f64>>`) for
  cache-friendly dot products. A write/commit split lets the solver
  silently discard a slot on curvature-condition failure without tearing
  ring state. `n * m` is overflow-checked before any allocation.
- **`StoppingCriteria` / `TerminationStatus`** — user-configured
  tolerances (gradient ∞-norm, step norm, relative function decrease,
  stagnation window, iteration cap) and a `Copy`, non-allocating exit
  enum. The termination path itself stays heap-free.

## Installation

```toml
[dependencies]
omni-opt = "0.1"
```

Opt in to `faer::Col<f64>` interoperation:

```toml
[dependencies]
omni-opt = { version = "0.1", features = ["faer"] }
```

**MSRV:** Rust 1.75.

## Usage

### Storage-generic kernels with `OmniVec`

Write numerical code once, run it over any supported backend:

```rust
use omni_opt::OmniVec;

fn l_inf_norm<V: OmniVec>(v: &V) -> f64 {
    v.as_slice()
        .iter()
        .fold(0.0_f64, |acc, &x| acc.max(x.abs()))
}

let owned: Vec<f64>  = vec![1.0, -3.5, 2.0];
let borrowed: &[f64] = &[-1.0, 0.5, -7.0];

assert_eq!(l_inf_norm(&owned),    3.5);
assert_eq!(l_inf_norm(&borrowed), 7.0);
```

With the `faer` feature enabled, `faer::Col<f64>` plugs in unchanged:

```rust,ignore
let col = faer::Col::<f64>::from_fn(3, |i| i as f64 - 1.5);
let n   = l_inf_norm(&col);
```

### Streaming rows with `RowAccess`

Implement the three primitives row-action solvers actually need — no row is
ever materialized as an owned buffer:

```rust
use omni_opt::RowAccess;

struct DenseRowMajor {
    nrows: usize,
    ncols: usize,
    data:  Vec<f64>, // row-major, length nrows * ncols
}

impl RowAccess for DenseRowMajor {
    fn nrows(&self) -> usize { self.nrows }
    fn ncols(&self) -> usize { self.ncols }

    fn row_dot(&self, i: usize, x: &[f64]) -> f64 {
        debug_assert_eq!(x.len(), self.ncols);
        let row = &self.data[i * self.ncols..(i + 1) * self.ncols];
        row.iter().zip(x).map(|(a, b)| a * b).sum()
    }

    fn row_sq_norm(&self, i: usize) -> f64 {
        let row = &self.data[i * self.ncols..(i + 1) * self.ncols];
        row.iter().map(|a| a * a).sum()
    }

    fn axpy_row(&self, i: usize, alpha: f64, y: &mut [f64]) {
        debug_assert_eq!(y.len(), self.ncols);
        let row = &self.data[i * self.ncols..(i + 1) * self.ncols];
        for (y_j, &a) in y.iter_mut().zip(row) {
            *y_j += alpha * a;
        }
    }
}
```

A CSR provider would implement the same three methods against its own
`indptr` / `indices` / `values` arrays — unchanged solver code, zero
allocation.

### Pre-allocated L-BFGS history with `LBFGSWorkspace`

```rust
use omni_opt::LBFGSWorkspace;

// Allocate ONCE, outside the solver loop.
let n = 1_000;
let mut ws = LBFGSWorkspace::new(n, LBFGSWorkspace::DEFAULT_M); // m = 10

// ----- per-iteration hot loop (zero allocation) -----
let s_new: Vec<f64> = vec![/* x_{k+1} - x_k */ 0.0; n];
let y_new: Vec<f64> = vec![/* g_{k+1} - g_k */ 0.0; n];

// 1. Fill the next slot.
ws.head_s_mut().copy_from_slice(&s_new);
ws.head_y_mut().copy_from_slice(&y_new);

// 2. Curvature check, then commit — or silently drop the slot.
let ys: f64 = s_new.iter().zip(&y_new).map(|(a, b)| a * b).sum();
if ys > 0.0 {
    ws.advance(1.0 / ys);
}
// ys <= 0.0: skip `advance`; the half-written slot is discarded
// and the ring remains intact.

// 3. Read history in the two-loop recursion (0 = oldest valid).
for k in 0..ws.count() {
    let _s_k   = ws.s_slot(k);
    let _y_k   = ws.y_slot(k);
    let _rho_k = ws.rho_at(k);
    // ... use ws.alpha_mut() as scratch for the first loop ...
}
```

Key guarantees:

- **Four** heap allocations, all inside `new` (`s`, `y`, `rho`, `alpha`).
- `n * m` is overflow-checked before any allocation — misconfiguration
  panics with a named message, never silently wraps.
- `head_*_mut`, `s_slot`, `y_slot`, `rho_at`, `alpha_mut`, `advance`,
  and `reset` are all allocation-free.

### Structured termination with `StoppingCriteria` / `TerminationStatus`

```rust
use omni_opt::{StoppingCriteria, TerminationStatus};

let criteria = StoppingCriteria {
    grad_inf_tol: 1e-8,
    ..StoppingCriteria::default() // step_tol=1e-10, rel_f_tol=1e-12,
                                  // stagnation_window=10, max_iter=1000
};

fn terminate(
    grad_inf: f64,
    iter: usize,
    c: &StoppingCriteria,
) -> Option<TerminationStatus> {
    if !grad_inf.is_finite() {
        Some(TerminationStatus::NumericalFailure)
    } else if grad_inf <= c.grad_inf_tol {
        Some(TerminationStatus::Converged)
    } else if iter >= c.max_iter {
        Some(TerminationStatus::MaxIterationsReached)
    } else {
        None
    }
}
```

`TerminationStatus` is `Copy` and heap-free — returning it preserves the
zero-allocation invariant all the way through the exit path.

## Cargo features

| Feature | Default | Effect                                                               |
|---------|---------|----------------------------------------------------------------------|
| `faer`  | off     | Blanket `OmniVec` / `OmniVecMut` impls for `faer::Col<f64>`.         |

## Design guarantees

- Zero heap allocation in the per-iteration hot loop of every solver.
- Heap allocation confined to builder / `new` constructors.
- Structured, non-panicking termination via enum discriminants.
- Generic over user-supplied storage — no forced copies.
- `debug_assert!` preconditions keep release builds branchless.

## Testing

```sh
cargo test
cargo test --features faer
cargo clippy --all-features --all-targets -- -D warnings
```

## License

Dual-licensed under either of

- Apache License, Version 2.0
- MIT License

at your option.
