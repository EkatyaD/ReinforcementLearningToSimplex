"""Payoff-matrix generation for the matrix (normal-form) game mode.

``Matrix`` holds a base integer payoff matrix and produces per-episode
perturbed copies: ``generate_matrix`` samples a fresh base matrix with entries
uniform in ``[low, high]``, and ``generate_perturbed_matrix`` adds elementwise
uniform noise of magnitude ``epsilon`` around that base (the per-episode
randomization the RL agent trains against).
"""

import numpy as np

class Matrix:
    """An m x n payoff matrix with per-episode epsilon perturbation."""

    def __init__(self, m=3, n=3, low=-1, high=1, epsilon=0.1, base_P=None):
        """Store dimensions, entry range [low, high], noise magnitude and base matrix."""
        self.m, self.n = m, n
        self.low, self.high = low, high
        self.base_P = base_P
        self.epsilon = float(epsilon)

    def resize(self, new_m, new_n):
        """Change the target dimensions (does not touch base_P)."""
        self.m = int(new_m)
        self.n = int(new_n)

    # generating matrix according to params
    def generate_matrix(self, mode="uniform"):
        """Generate an integer matrix with entries uniform in [low, high]."""
        if mode != "uniform":
            raise ValueError(f"Unknown generation mode '{mode}'. Supported: 'uniform'.")
        self.base_P = np.random.randint(self.low, self.high + 1, size=(self.m, self.n))
        return self.base_P

    def return_size(self):
        """Return the (m, n) dimensions."""
        return (self.m, self.n)

    def return_epsilon(self):
        """Return the perturbation magnitude epsilon."""
        return self.epsilon

    # add epsilon noise to matrix
    def generate_perturbed_matrix(self):
        """
        Add elementwise uniform noise in [-epsilon, epsilon] to the base matrix.
        Returns a NEW Matrix instance with epsilon=0 (so it won't re-perturb on top).
        """
        if self.base_P is None:
            raise RuntimeError("Base matrix is None. Call generate_matrix(...) first.")

        noise = np.random.uniform(-self.epsilon, self.epsilon, size=(self.m, self.n))
        noise = np.round(noise, decimals=5)
        perturbed_P = self.base_P + noise
        return Matrix(self.m, self.n, self.low, self.high, epsilon=0.0, base_P=perturbed_P)

    def copy(self):
        """Return an independent copy (base_P deep-copied)."""
        return Matrix(self.m, self.n, self.low, self.high, self.epsilon,
                      np.copy(self.base_P) if self.base_P is not None else None)
