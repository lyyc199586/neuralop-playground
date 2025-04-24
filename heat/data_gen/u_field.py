"""
Heat Equation Solver: One Time-Step Implicit Euler Method

This script provides a solver for the heat equation using the implicit Euler method
on a regular 2D grid. The domain is a unit square [0,1]x[0,1]

Key Features:
- Finite difference method for spatial discretization.
- Implicit Euler method for time-stepping.
- Dirichlet boundary conditions: u=0 on the boundaries.
- Batch processing for multiple conductivity samples.

Usage:
Run the script directly to process a dataset of conductivity fields and compute
the corresponding temperature fields after one time step.
"""

import numpy as np
from scipy.sparse import csr_matrix, linalg
from pathlib import Path
from typing import Optional

def idx(i: int, j: int, ny: int) -> int:
    """
    Compute the 1D index for a 2D grid point.

    Parameters:
        i (int): Row index in the 2D grid.
        j (int): Column index in the 2D grid.
        ny (int): Number of rows in the grid.

    Returns:
        int: Flattened 1D index corresponding to (i, j).
    """
    return i * ny + j

def solve_one_step_heat(
    k: np.ndarray,
    f: np.ndarray,
    dt: float = 1.0,
    u_n: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Solve one implicit Euler time step of the heat equation.

    The heat equation is given by:
        (u^{n+1} - u^n) / dt = div(k(x, y) * grad(u^{n+1})) + f(x, y)

    where:
        - u^n is the temperature at the current time step.
        - u^{n+1} is the temperature at the next time step.
        - k(x, y) is the conductivity field.
        - f(x, y) is the source term.
        - dt is the time step size.

    Discrete form:
        Let u[i, j] represent the temperature at grid point (i, j).
        The discrete form of the equation is:
            (u[i, j]^{n+1} - u[i, j]^n) / dt =
            (k_avg_x * (u[i+1, j]^{n+1} - u[i, j]^{n+1}) / dx^2) +
            (k_avg_x * (u[i-1, j]^{n+1} - u[i, j]^{n+1}) / dx^2) +
            (k_avg_y * (u[i, j+1]^{n+1} - u[i, j]^{n+1}) / dy^2) +
            (k_avg_y * (u[i, j-1]^{n+1} - u[i, j]^{n+1}) / dy^2) + f[i, j]

        Here:
        - k_avg_x = 0.5 * (k[i, j] + k[i+1, j]) for the x-direction.
        - k_avg_y = 0.5 * (k[i, j] + k[i, j+1]) for the y-direction.

    Using the finite difference method on a regular 2D grid:
        - The domain is discretized into a grid of size (nx, ny).
        - Spatial derivatives are approximated using central differences.
        - The resulting system of equations is solved implicitly for u^{n+1}.

    Dirichlet boundary conditions:
        - u = 0 on all boundaries of the domain.

    Parameters:
        k (np.ndarray): Conductivity field, shape (nx, ny).
        f (np.ndarray): Source term, shape (nx, ny).
        dt (float): Time step size.
        u_n (Optional[np.ndarray]): Initial temperature field, shape (nx, ny).
                                     If None, it is assumed to be zero.

    Returns:
        np.ndarray: Temperature field after one time step, shape (nx, ny).
    """
    nx, ny = k.shape
    dx, dy = 1.0 / nx, 1.0 / ny
    N = nx * ny

    if u_n is None:
        u_n = np.zeros((nx, ny))
    rhs = u_n.flatten() / dt + f.flatten()

    rows, cols, data = [], [], []
    for i in range(nx):
        for j in range(ny):
            p = idx(i, j, ny)
            main_diag = 1.0 / dt
            entries = [(p, p, main_diag)]
            for di, dj, h2 in [(-1, 0, dx**2), (1, 0, dx**2), (0, -1, dy**2), (0, 1, dy**2)]:
                ni, nj = i + di, j + dj
                if 0 <= ni < nx and 0 <= nj < ny:
                    q = idx(ni, nj, ny)
                    k_avg = 0.5 * (k[i, j] + k[ni, nj])
                    coeff = -k_avg / h2
                    entries.append((p, q, coeff))
                    main_diag += k_avg / h2
            entries[0] = (p, p, main_diag)
            for r, c, v in entries:
                rows.append(r)
                cols.append(c)
                data.append(v)

    A = csr_matrix((data, (rows, cols)), shape=(N, N))

    # Dirichlet BC: u = 0 on all boundaries
    for i in range(nx):
        for j in range(ny):
            if i == 0 or i == nx - 1 or j == 0 or j == ny - 1:
                p = idx(i, j, ny)
                A[p, :] = 0.0
                A[p, p] = 1.0
                rhs[p] = 0.0

    u_np1 = linalg.spsolve(A, rhs).reshape((nx, ny))
    return u_np1

# === Batch Processing ===

if __name__ == "__main__":
    """
    Batch process a dataset of conductivity fields to compute temperature fields.

    This script loads a dataset of conductivity fields, applies the heat equation
    solver to compute the temperature fields after one time step, and saves the
    results to a file.
    """
    # Input path to saved conductivity dataset
    res = 64
    n_samples = 50
    k_path = Path(f"../../data/heat/k_n{n_samples}_res{res}.npy")
    k_dataset = np.load(k_path)
    n_samples, nx, ny = k_dataset.shape 

    # Constant source term (can be customized)
    f = np.ones((nx, ny))   

    # Output buffer
    u_dataset = np.zeros_like(k_dataset)    

    # Compute
    for i in range(n_samples):
        print(f"Solving sample {i+1}/{n_samples}")
        u_dataset[i] = solve_one_step_heat(k_dataset[i], f) 

    # Save
    u_path = k_path.parent / f"u_n{n_samples}_res{nx}.npy"
    np.save(u_path, u_dataset)
    print(f"Saved results to {u_path}")

