import numpy as np

class Matrix:
    # matrix initialization
    def __init__(self, m=3, n=3, min=-1, max=1, epsilon=0.1, base_P=None):
        self.m, self.n = m, n
        self.min, self.max = min, max
        self.base_P = base_P
        self.epsilon = float(epsilon)

    def resize(self, new_m, new_n):
        self.m = int(new_m)
        self.n = int(new_n)

    # generating matrix according to params
    def generateMatrix(self, mode="uniform"):
        """Generate an integer matrix with entries uniform in [min, max]."""
        if mode != "uniform":
            raise ValueError(f"Unknown generation mode '{mode}'. Supported: 'uniform'.")
        self.base_P = np.random.randint(self.min, self.max + 1, size=(self.m, self.n))
        return self.base_P

    def returnSize(self):
        return (self.m, self.n)

    def returnEpsilon(self):
        return self.epsilon

    # add epsilon noise to matrix
    def generate_perturbed_matrix(self):
        """
        Add elementwise uniform noise in [-epsilon, epsilon] to the base matrix.
        Returns a NEW Matrix instance with epsilon=0 (so it won't re-perturb on top).
        """
        if self.base_P is None:
            raise RuntimeError("Base matrix is None. Call generateMatrix(...) first.")

        noise = np.random.uniform(-self.epsilon, self.epsilon, size=(self.m, self.n))
        noise = np.round(noise, decimals=5)
        perturbed_P = self.base_P + noise
        return Matrix(self.m, self.n, self.min, self.max, epsilon=0.0, base_P=perturbed_P)

    def copy(self):
        return Matrix(self.m, self.n, self.min, self.max, self.epsilon,
                      np.copy(self.base_P) if self.base_P is not None else None)
