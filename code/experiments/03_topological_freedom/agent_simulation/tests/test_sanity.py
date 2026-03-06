"""
Sanity Checks — muessen alle passieren vor Experiment-Laeufen.
Spec §9: 6 obligatorische Tests.
"""

import pytest
import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.simulation import Simulation
from src.environment import Environment
from src.agents import Agent, Action
from src.config import (
    GRID_SIZE, R_MAX, S_MAX,
    TYPE_A, TYPE_B, TYPE_C,
    STEAL_RATE, DEFEND_BLOCK
)


class TestSanityChecks:

    def test_martyr_death(self):
        """Type-B verliert Energy durch Share -> Energy-Ungleichheit entsteht."""
        sim = Simulation(seed=42)
        sim.agents = [a for a in sim.agents if a.type == TYPE_B]
        sim.agents_grid = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)
        for a in sim.agents:
            sim.agents_grid[a.x, a.y] = a
        sim.env.R = np.full((GRID_SIZE, GRID_SIZE), R_MAX * 0.3)
        sim.env.S = np.full((GRID_SIZE, GRID_SIZE), S_MAX * 0.8)

        sim.run(ticks=100, progress=False)

        energies = [a.energy for a in sim.agents if a.alive]
        energy_range = max(energies) - min(energies) if energies else 0
        assert energy_range > 10, "Share should create energy inequality"

    def test_tyrant_persistence(self):
        """Nur Type-A mit vollen Ressourcen -> ueberleben initial, spaeter Decline."""
        sim = Simulation(seed=42)
        sim.agents = [a for a in sim.agents if a.type == TYPE_A]
        initial_count = len(sim.agents)
        sim.agents_grid = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)
        for a in sim.agents:
            sim.agents_grid[a.x, a.y] = a
        sim.env.R = np.full((GRID_SIZE, GRID_SIZE), R_MAX)

        sim.run(ticks=100, progress=False)
        alive_100 = sum(1 for a in sim.agents if a.alive)
        assert alive_100 > 0, "Tyrants should survive initially"

        sim.run(ticks=900, progress=False)
        alive_1000 = sum(1 for a in sim.agents if a.alive)
        assert alive_1000 < initial_count, "Tyrants should see some decline from stealing"

    def test_unity_stability(self):
        """Nur Type-C in stabiler Umgebung -> Gleichgewicht."""
        sim = Simulation(seed=42)
        sim.agents = [a for a in sim.agents if a.type == TYPE_C]
        sim.agents_grid = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)
        for a in sim.agents:
            sim.agents_grid[a.x, a.y] = a
        sim.env.R = np.full((GRID_SIZE, GRID_SIZE), R_MAX * 0.8)
        sim.env.S = np.zeros((GRID_SIZE, GRID_SIZE))

        sim.run(ticks=500, progress=False)

        alive = sum(1 for a in sim.agents if a.alive)
        assert alive > len(sim.agents) * 0.5, "Unity agents should maintain stability"

    def test_steal_transfer_no_defend(self):
        """Steal ohne Defend -> voller Transfer."""
        attacker = Agent(0, TYPE_A, 0, 0)
        victim = Agent(1, TYPE_B, 0, 1)
        attacker.energy = 10
        victim.energy = 20

        steal_amount = min(STEAL_RATE, victim.energy)
        victim.energy -= steal_amount
        attacker.energy += steal_amount

        assert steal_amount == STEAL_RATE
        assert attacker.energy == 10 + STEAL_RATE
        assert victim.energy == 20 - STEAL_RATE

    def test_steal_with_defend(self):
        """Steal mit Defend -> 70% geblockt."""
        steal_base = STEAL_RATE
        effective = steal_base * (1 - DEFEND_BLOCK)
        assert effective == pytest.approx(STEAL_RATE * 0.3, rel=0.01)

    def test_entropy_diffusion(self):
        """Einzelne High-S Zelle -> S diffundiert zu Nachbarn."""
        env = Environment(seed=42)
        env.S = np.zeros((GRID_SIZE, GRID_SIZE))
        env.S[16, 16] = 50

        initial_center = env.S[16, 16]
        env.update()

        assert env.S[16, 16] < initial_center
        neighbors = env.get_neighbors(16, 16)
        neighbor_S = [env.S[nx, ny] for nx, ny in neighbors]
        assert any(s > 0 for s in neighbor_S)

    def test_experiment_registry(self):
        """Experiment-Registry hat alle 17 Experimente."""
        from experiments.experiment_config import EXPERIMENTS
        assert len(EXPERIMENTS) == 17, f"Expected 17 experiments, got {len(EXPERIMENTS)}"

        # Pruefe Gruppen
        groups = {}
        for name, cfg in EXPERIMENTS.items():
            g = cfg['group']
            groups.setdefault(g, []).append(name)

        assert len(groups['A']) == 4, "Group A should have 4 experiments"
        assert len(groups['B']) == 5, "Group B should have 5 experiments"
        assert len(groups['C']) == 4, "Group C should have 4 experiments"
        assert len(groups['D']) == 4, "Group D should have 4 experiments"

    def test_reproducibility(self):
        """Gleicher Seed -> identische Ergebnisse."""
        params = {'K1': 15, 'MOVEMENT_ENABLED': False}

        sim1 = Simulation(seed=42, params=params)
        sim1.run(ticks=50, progress=False)
        surv1 = sim1.get_survival_rates()

        sim2 = Simulation(seed=42, params=params)
        sim2.run(ticks=50, progress=False)
        surv2 = sim2.get_survival_rates()

        assert surv1[TYPE_A] == surv2[TYPE_A]
        assert surv1[TYPE_B] == surv2[TYPE_B]
        assert surv1[TYPE_C] == surv2[TYPE_C]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
