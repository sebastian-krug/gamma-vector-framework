"""
Torus-Grid mit Moore-Nachbarschaft.
Jede Zelle hat R (Resources) und S (Degradation).
"""

import numpy as np
from .config import (
    GRID_SIZE, R_MAX, S_MAX, DELTA, MU, G, LAMBDA
)


class Environment:
    def __init__(self, seed=None, params=None):
        if seed is not None:
            np.random.seed(seed)

        self.params = params or {}
        self.mu = self.params.get('MU', MU)
        self.delta = self.params.get('DELTA', DELTA)
        self.lambda_ = self.params.get('LAMBDA', LAMBDA)

        self.R = np.full((GRID_SIZE, GRID_SIZE), R_MAX * 0.8)
        self.S = np.zeros((GRID_SIZE, GRID_SIZE))

        self.extracted = np.zeros((GRID_SIZE, GRID_SIZE))
        self.action_costs = np.zeros((GRID_SIZE, GRID_SIZE))

    def get_neighbors(self, x, y):
        """Moore-Nachbarschaft mit Torus-Wrapping (Spec §1.1)"""
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = (x + dx) % GRID_SIZE
                ny = (y + dy) % GRID_SIZE
                neighbors.append((nx, ny))
        return neighbors

    def diffusion(self):
        """Discrete Laplacian auf Moore-Nachbarschaft (Spec §6.1)"""
        new_S = np.zeros_like(self.S)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                neighbors = self.get_neighbors(x, y)
                neighbor_S = [self.S[nx, ny] for nx, ny in neighbors]
                mean_neighbor_S = np.mean(neighbor_S)
                new_S[x, y] = self.mu * (mean_neighbor_S - self.S[x, y])
        return new_S

    def update(self):
        """Grid-Update nach allen Aktionen (Spec §1.3)"""
        diff = self.diffusion()
        self.S = (1 - self.delta) * self.S + diff + self.action_costs
        self.S = np.clip(self.S, 0, S_MAX)

        regrowth = G * (R_MAX - self.R) * (1 - self.lambda_ * self.S / S_MAX)
        self.R = self.R + regrowth - self.extracted
        self.R = np.clip(self.R, 0, R_MAX)

        self.extracted = np.zeros((GRID_SIZE, GRID_SIZE))
        self.action_costs = np.zeros((GRID_SIZE, GRID_SIZE))

    def add_entropy(self, x, y, amount):
        """Entropie-Kosten einer Aktion hinzufuegen"""
        self.action_costs[x, y] += amount

    def extract_resource(self, x, y, amount):
        """Ressourcen extrahieren (fuer Harvest)"""
        actual = min(amount, self.R[x, y])
        self.extracted[x, y] += actual
        return actual
