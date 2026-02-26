# Global-optimization-toolkit
From-scratch Python implementation of Truncated Newton, Sequential Penalty and Filled Function algorithms for nonlinear optimization. Pure NumPy — no auto-diff, no black-box solvers. 20+ benchmark problems across constrained and global optimization. 

# Nonlinear Optimization Algorithms in Python

> From-scratch implementation of **Truncated Newton**, **Sequential Penalty**, and **Filled Function** algorithms.

No black-box solvers. No automatic differentiation. Pure NumPy.

-----

## Overview

This repository implements three algorithms that together cover the full spectrum of nonlinear optimization — from local unconstrained minimization to global search:

|File         |Algorithm                 |Problem type             |
|-------------|--------------------------|-------------------------|
|`NewtonT.py` |Truncated Newton (NT)     |Unconstrained, local     |
|`penalita.py`|Sequential Penalty (PS)   |Constrained, local       |
|`Filled.py`  |Filled Function (FF)      |Unconstrained, **global**|
|`funzioni.py`|Test functions & utilities|—                        |

-----

## Installation

No external dependencies beyond NumPy:

```bash
git clone https://github.com/<your-username>/<global-optimization-toolkit>.git
cd <global-optimization-toolkit>
pip install numpy
```

Python 3.8+ recommended.

-----

## Usage

### Truncated Newton — unconstrained minimization

```python
from NewtonT import NT
from funzioni import fprova, grad_fprova, initpointprova

x0  = initpointprova(2)          # starting point
sol = NT(x0, fprova(), grad_fprova(), n=2, delta=1e-6)
print(sol)                        # → near (0, 0), f ≈ 0
```

### Sequential Penalty — constrained minimization

```python
from penalita import PS
from funzioni import (initpointP7, Pf7, Pg7, Ph7,
                      gradientPf7, gradientPg7, gradientPh7)

sol = PS(
    x0        = initpointP7(4),
    g         = Pg7(),            # inequality constraints g(x) ≤ 0
    h         = Ph7(),            # equality constraints h(x) = 0
    f         = Pf7(),
    gradientf = gradientPf7(),
    gradientg = gradientPg7(),
    gradienth = gradientPh7(),
    n         = 4,
)
```

### Filled Function — global optimization

```python
from Filled import Filled, gradapprF
from funzioni import initpointF6, Filled6

f   = Filled6()
gf  = gradapprF(f)               # automatic finite-difference gradient
sol, fval = Filled(
    x0      = initpointF6(2),
    f       = f,
    gradientf = gf,
    n       = 2,
    delta   = 1e-8,
    gamma   = 0.5,               # Gaussian hill width
    box_lo  = -3.0,
    box_hi  =  3.0,
)
print(f"f* = {fval:.6f},  x* = {sol}")
# → f* = -1.031628,  x* ≈ [-0.090, 0.713]  (Six-Hump Camel global min)
```

### Run all benchmarks

```bash
python NewtonT.py     # quick NT test on Rastrigin 2D
python penalita.py    # P3, P7, P11
python Filled.py      # F1–F129 (13 benchmark functions)
```

-----

## Algorithms

### 1 · Truncated Newton (`NewtonT.py`)

At each iteration, the Newton system `H·d = −g` is solved **approximately** using the **Truncated Conjugate Gradient** (Steihaug-CG). This avoids forming the full n×n Hessian, keeping cost at O(n) per CG step.

**Key ideas:**

- **Hessian-vector products** via central finite differences on the gradient — no explicit Hessian assembly
- **Early termination** of CG when negative curvature is detected (guarantees descent) or when the relative residual is small enough
- **Adaptive CG tolerance** `η_k = min(0.5, √‖g_k‖)` — coarse inner solutions early, refined near optimality — enables superlinear convergence
- **Armijo backtracking** line search with `γ = 1e-4`, fallback to steepest descent if CG direction is not a descent direction

```
for k = 0, 1, 2, ...:
    g  ← ∇f(xₖ)
    if ‖g‖ ≤ δ: stop
    d  ← GCT(xₖ, g)           # truncated CG
    α  ← Armijo(xₖ, f, d)
    xₖ₊₁ ← xₖ + α·d
```

### 2 · Sequential Penalty (`penalita.py`)

Converts a constrained problem into a sequence of unconstrained ones by penalizing violations:

```
F_ε(x) = f(x) + (1/ε) · Σ max(0, gᵢ(x))²  +  (1/ε) · Σ hⱼ(x)²
```

As `ε → 0`, the penalty grows and forces the minimizer towards feasibility. The outer loop follows the schema from the course notes:

```
P1: estimate KKT multipliers λᵢ = (2/ε)·max(0,gᵢ),  μⱼ = (2/ε)·hⱼ
    check KKT conditions → stop if satisfied
P2: x_new ← NT(x, F_ε, ∇F_ε, δ)      # inner minimization
P3: if violation not improved by θ₁ → ε ← θ₂·ε   (increase penalty)
P4: δ ← θ₃·δ,  x ← x_new,  go to P1
```

**Default parameters:** `ε₀ = 1.0`, `θ₁ = 0.25`, `θ₂ = 0.1`, `θ₃ = 0.5`, `tol_kkt = 1e-4`.

### 3 · Filled Function (`Filled.py`)

Escapes local minima to search for the global optimum. Given the current local minimizer `x*_k`, the **additive filled function of Type 1** is:

```
U_k(x) = τ · (min{0, f(x) − f(x*_k) + ρ})³  +  exp(−‖x − x*_k‖² / γ²)
```

- The **Gaussian term** makes `x*_k` a strict local *maximum* of `U_k`, forcing any local minimizer to move away from it
- The **cubic penalty** is zero where `f(x) ≥ f(x*_k) − ρ` and strongly negative where `f` is better, pulling search towards lower-value basins

By Proposition 3.3.1 (Lucidi, 2024), for sufficiently large `τ` and `0 < ρ < f(x*_k) − f*`, all global minimizers of `U_k` lie in `{x : f(x) < f(x*_k)}` — guaranteed progress.

```
1. x* ← NT(x0, f)                     # local minimizer
2. build U_k centered at x*
3. for each restart point xₛ:
       z ← NT(xₛ, U_k)
       if f(z) < f(x*): x* ← NT(z, f)   # refine on original f
4. if no improvement: stop
   else: go to 2
```

**Multi-start strategy:** geometric perturbations along coordinate axes (at scales `γ·10⁰…γ·10⁻³`) + uniform random samples in the feasible box.

-----

## Benchmark Results

### Sequential Penalty

|Problem|Description              |x*                      |f*    |KKT      |
|-------|-------------------------|------------------------|------|---------|
|P3     |Rosenbrock + constraints |(0.5, 0.866)            |38.199|✓        |
|P7     |Min distance + equalities|(2.0, 2.0, 0.849, 1.131)|13.858|✓ iter 10|
|P11    |Maximize x₁ + mixed      |(1.0, 1.0, 0.0, 0.0)    |−1.000|✓ iter 15|

### Filled Functions

|Problem|Description    |x*             |f*          |
|-------|---------------|---------------|------------|
|F1     |Rosenbrock 4D  |(1, 1, 1, 1)   |0.000000    |
|F5     |Wood function  |(1, 1, 1, 1)   |0.000000    |
|F6     |Six-Hump Camel |(−0.090, 0.713)|−1.031628   |
|F13    |Shubert-like 4D|(1, −1, −1, 1) |−4.400000   |
|F129   |Box volume     |(12, 12)       |−3456.000000|

-----

## Project Structure

```
.
├── NewtonT.py      # Truncated Newton + Steihaug-CG + Armijo line search
├── penalita.py     # Sequential Penalty method (PS)
├── Filled.py       # Filled Function algorithm (FF) for global optimization
└── funzioni.py     # All test functions, gradients, and penalty helpers
```

-----

## References

1. S. Lucidi, *Appunti dalle lezioni di Ottimizzazione Continua*, Sapienza University of Rome, A.A. 2024–2025
1. J. Nocedal, S. J. Wright, *Numerical Optimization*, 2nd ed., Springer, 2006
1. T. Steihaug, *The conjugate gradient method and trust regions in large scale optimization*, SIAM J. Numer. Anal., 1983
1. Y.-P. Zhang, *Filled functions for unconstrained global optimization*, J. Global Optim., 2001
1. R. Fletcher, *Practical Methods of Optimization*, 2nd ed., Wiley, 1987
