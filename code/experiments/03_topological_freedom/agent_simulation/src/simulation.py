"""
Synchroner Tick-Loop nach Spec §6.1
"""

import numpy as np
from tqdm import tqdm
from .environment import Environment
from .agents import Agent
from .actions import ActionResolver
from .metrics import MetricsCollector
from .config import (
    GRID_SIZE, TOTAL_TICKS, AGENTS_PER_TYPE,
    TYPE_A, TYPE_B, TYPE_C,
    RGS_WINDOW
)


class Simulation:
    def __init__(self, seed=None, params=None):
        self.seed = seed
        self.params = params or {}

        self.env = Environment(seed=seed, params=self.params)
        self.agents = []
        self.agents_grid = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)
        self.tick = 0
        self.metrics = MetricsCollector()
        self.resolver = ActionResolver(self.env, params=self.params)

        Agent.selective_share = self.params.get('SELECTIVE_SHARE', False)
        Agent.policy_v2 = self.params.get('POLICY_V2', False)

        self._initialize_agents()

    def _initialize_agents(self):
        """Agenten zufaellig platzieren (Spec §2.3)"""
        if self.seed is not None:
            np.random.seed(self.seed)

        positions = [(x, y) for x in range(GRID_SIZE) for y in range(GRID_SIZE)]
        np.random.shuffle(positions)

        agent_id = 0

        for i in range(AGENTS_PER_TYPE):
            x, y = positions[agent_id]
            agent = Agent(agent_id, TYPE_A, x, y)
            self.agents.append(agent)
            self.agents_grid[x, y] = agent
            agent_id += 1

        for i in range(AGENTS_PER_TYPE):
            x, y = positions[agent_id]
            agent = Agent(agent_id, TYPE_B, x, y)
            self.agents.append(agent)
            self.agents_grid[agent.x, agent.y] = agent
            agent_id += 1

        for i in range(AGENTS_PER_TYPE):
            x, y = positions[agent_id]
            agent = Agent(agent_id, TYPE_C, x, y)
            self.agents.append(agent)
            self.agents_grid[agent.x, agent.y] = agent
            agent_id += 1

    def run(self, ticks=None, progress=True):
        """Simulation laufen lassen."""
        if ticks is None:
            ticks = TOTAL_TICKS

        iterator = tqdm(range(ticks), desc="Simulating") if progress else range(ticks)

        for _ in iterator:
            self._tick()
            self.tick += 1

            if not any(a.alive for a in self.agents):
                break

        return self.metrics

    def _tick(self):
        """Ein Tick nach Spec §6.1"""
        living_agents = [a for a in self.agents if a.alive]

        # Step 0: Movement
        movement_enabled = self.params.get('MOVEMENT_ENABLED', False)
        if movement_enabled:
            agents_by_pos = {(a.x, a.y): a for a in living_agents}
            move_order = living_agents.copy()
            np.random.shuffle(move_order)

            for agent in move_order:
                if not agent.alive:
                    continue

                new_pos = agent.decide_move(agents_by_pos, self.tick, self.params)

                if new_pos:
                    if new_pos not in agents_by_pos:
                        old_pos = (agent.x, agent.y)
                        del agents_by_pos[old_pos]
                        self.agents_grid[old_pos[0], old_pos[1]] = None

                        agent.execute_move(new_pos[0], new_pos[1], self.params)

                        agents_by_pos[new_pos] = agent
                        self.agents_grid[new_pos[0], new_pos[1]] = agent

        # Step 1: Observe
        for agent in living_agents:
            agent.observe(self.env, self.agents_grid)

        # Step 2: Update Gamma
        for agent in living_agents:
            agent.update_gamma()

        # Step 3: Decide
        decisions = {}
        for agent in living_agents:
            decisions[agent] = agent.decide()

        # Step 4-5: Resolve + Update Energies
        self.resolver.current_tick = self.tick
        share_log = self.resolver.resolve_all(decisions, self.agents_grid)

        # Step 6: Metabolic Cost
        for agent in living_agents:
            agent.apply_metabolic_cost()

        # Step 7: Update Grid
        self.env.update()

        # Step 8: Update Memory / Clear Flags
        for agent in living_agents:
            if agent.alive:
                agent.memory.append('tick')

        # Step 8b: Update Trust
        for agent in living_agents:
            if agent.alive:
                agent.update_trust()

        # Step 8c: Cleanup alte Share-History (RGS)
        rgs_window = self.params.get('RGS_WINDOW', RGS_WINDOW)
        for agent in living_agents:
            if agent.alive:
                agent.cleanup_old_shares(self.tick, rgs_window)

        # Step 9: Death Check + Grid Update
        for agent in self.agents:
            if not agent.alive and self.agents_grid[agent.x, agent.y] == agent:
                self.agents_grid[agent.x, agent.y] = None

        # Metrics sammeln
        self.metrics.record(self.tick, self.agents, self.env, share_log=share_log)

    def get_survival_rates(self):
        """Survival Rates nach Typ"""
        survival = {TYPE_A: 0, TYPE_B: 0, TYPE_C: 0}
        for agent in self.agents:
            if agent.alive:
                survival[agent.type] += 1

        return {
            TYPE_A: survival[TYPE_A] / AGENTS_PER_TYPE,
            TYPE_B: survival[TYPE_B] / AGENTS_PER_TYPE,
            TYPE_C: survival[TYPE_C] / AGENTS_PER_TYPE,
        }
