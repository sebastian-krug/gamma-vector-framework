"""
Drei Agententypen mit unterschiedlichem Gamma-Verhalten.
"""

import numpy as np
from enum import Enum
from collections import deque
from .config import (
    TYPE_A, TYPE_B, TYPE_C,
    MEMORY_LENGTH, THETA_E, THETA_V,
    INITIAL_ENERGY, SHARE_THRESHOLD, METABOLIC_COST
)


class Action(Enum):
    IDLE = 'idle'
    HARVEST = 'harvest'
    SHARE = 'share'
    STEAL = 'steal'
    DEFEND = 'defend'


class Agent:
    # Globale Experiment-Flags (werden von Simulation gesetzt)
    selective_share = False
    policy_v2 = False

    def __init__(self, agent_id, agent_type, x, y):
        self.id = agent_id
        self.type = agent_type
        self.x = x
        self.y = y
        self.energy = INITIAL_ENERGY
        self.alive = True

        self.memory = deque(maxlen=MEMORY_LENGTH)
        self.was_attacked = False

        if agent_type == TYPE_A:
            self.gamma = 1.0
        elif agent_type == TYPE_B:
            self.gamma = 0.0
        else:
            self.gamma = 0.5

        self.expected_R = None
        self.ema_error = 0

        self.R_local = 0
        self.S_local = 0
        self.S_neighbors = []
        self.S_neighbors_var = 0
        self.neighbor_agents = []

        # Trust-System
        self.trust = {}
        self.received_share_from = set()
        self.attacked_by = set()

        # Reziprozitaets-Tracking (RGS)
        self.shared_to_history = {}

        # Movement
        self.move_cooldown = 0

    @property
    def position(self):
        return (self.x, self.y)

    def observe(self, env, agents_grid):
        """Observations sammeln (Spec §6.2)"""
        self.R_local = env.R[self.x, self.y]
        self.S_local = env.S[self.x, self.y]

        neighbors = env.get_neighbors(self.x, self.y)
        self.S_neighbors = [env.S[nx, ny] for nx, ny in neighbors]
        self.S_neighbors_var = np.var(self.S_neighbors) if self.S_neighbors else 0

        self.neighbor_agents = []
        for nx, ny in neighbors:
            if agents_grid[nx, ny] is not None and agents_grid[nx, ny].alive:
                self.neighbor_agents.append(agents_grid[nx, ny])

        if self.type == TYPE_C:
            if self.expected_R is not None:
                error = abs(self.expected_R - self.R_local)
                self.ema_error = 0.3 * error + 0.7 * self.ema_error
            self.expected_R = self.R_local

    def update_gamma(self):
        """Gamma-Controller nur fuer Type C (Spec §4)"""
        if self.type != TYPE_C:
            return

        A_t = self.was_attacked
        E_t = self.ema_error
        V_t = self.S_neighbors_var

        if A_t:
            self.gamma = 1.0
        elif E_t > THETA_E or V_t > THETA_V:
            self.gamma = 0.2
        else:
            self.gamma = 0.5

    def decide(self):
        """Aktion waehlen basierend auf Typ und Gamma (Spec §4.4, §5)"""
        if self.type == TYPE_A:
            return self._policy_tyrant()
        elif self.type == TYPE_B:
            return self._policy_martyr()
        else:
            if Agent.policy_v2:
                return self._policy_unity_v2()
            else:
                return self._policy_unity()

    def _policy_tyrant(self):
        """Type-A Policy (Spec §5.1)"""
        targets = [a for a in self.neighbor_agents if a.energy > 0]
        if targets:
            target = targets[np.random.randint(len(targets))]
            return (Action.STEAL, target)
        elif self.R_local > 0:
            return (Action.HARVEST, None)
        else:
            return (Action.IDLE, None)

    def _policy_martyr(self):
        """Type-B Policy (Spec §5.2)"""
        if self.energy > SHARE_THRESHOLD and self.neighbor_agents:
            target = self.neighbor_agents[np.random.randint(len(self.neighbor_agents))]
            return (Action.SHARE, target)
        elif self.R_local > 0:
            return (Action.HARVEST, None)
        else:
            return (Action.IDLE, None)

    def _policy_unity(self):
        """Type-C Policy (Spec §4.4)"""
        stable = (self.ema_error <= THETA_E) and (self.S_neighbors_var <= THETA_V)

        if self.was_attacked:
            return (Action.DEFEND, None)
        elif stable and self.energy > SHARE_THRESHOLD and self.neighbor_agents:
            if Agent.selective_share:
                c_neighbors = [a for a in self.neighbor_agents if a.type == TYPE_C]
                if c_neighbors:
                    target = c_neighbors[np.random.randint(len(c_neighbors))]
                    return (Action.SHARE, target)
                elif self.R_local > 0:
                    return (Action.HARVEST, None)
                else:
                    return (Action.IDLE, None)
            else:
                target = self.neighbor_agents[np.random.randint(len(self.neighbor_agents))]
                return (Action.SHARE, target)
        elif self.R_local > 0:
            return (Action.HARVEST, None)
        else:
            return (Action.IDLE, None)

    def _policy_unity_v2(self):
        """
        Unity Policy v2: Reziprozitaet + Exklusion (Exp6)

        1. Wenn angegriffen -> Defend
        2. Teile NUR mit trusted neighbors (trust >= 1)
        3. Teile NIE mit blacklisted neighbors (trust <= -2)
        4. Wenn keine trusted neighbors -> versuche C-Nachbarn (Cluster-Bildung)
        5. Wenn keine eligiblen Targets -> Harvest
        """
        if self.was_attacked:
            return (Action.DEFEND, None)

        trusted_neighbors = [
            a for a in self.neighbor_agents
            if self.is_trusted(a.id) and a.alive
        ]

        c_neighbors_neutral = [
            a for a in self.neighbor_agents
            if a.type == TYPE_C
            and a.alive
            and self.get_trust(a.id) == 0
        ]

        stable = (self.ema_error <= THETA_E) and (self.S_neighbors_var <= THETA_V)

        if stable and self.energy > SHARE_THRESHOLD:
            if trusted_neighbors:
                target = max(trusted_neighbors, key=lambda a: self.get_trust(a.id))
                return (Action.SHARE, target)

            eligible_c = [a for a in c_neighbors_neutral if not self.is_blacklisted(a.id)]
            if eligible_c:
                target = eligible_c[np.random.randint(len(eligible_c))]
                return (Action.SHARE, target)

        if self.R_local > 0:
            return (Action.HARVEST, None)
        else:
            return (Action.IDLE, None)

    def apply_metabolic_cost(self):
        """Grundverbrauch pro Tick (Spec §3.5)"""
        self.energy -= METABOLIC_COST
        if self.energy <= 0:
            self.alive = False

    def record_attack(self):
        """Markiere dass Agent angegriffen wurde"""
        self.was_attacked = True
        self.memory.append('attacked')

    def clear_attack_flag(self):
        """Reset am Ende des Ticks, aber Memory bleibt"""
        self.was_attacked = any(m == 'attacked' for m in self.memory)

    # === Trust-System ===

    def update_trust(self):
        """Trust-Update am Ende jedes Ticks."""
        for attacker_id in self.attacked_by:
            current = self.trust.get(attacker_id, 0)
            self.trust[attacker_id] = max(-3, current - 2)

        for sharer_id in self.received_share_from:
            current = self.trust.get(sharer_id, 0)
            self.trust[sharer_id] = min(+3, current + 1)

        self.attacked_by = set()
        self.received_share_from = set()

    def get_trust(self, agent_id):
        return self.trust.get(agent_id, 0)

    def is_trusted(self, agent_id):
        return self.get_trust(agent_id) >= 1

    def is_blacklisted(self, agent_id):
        return self.get_trust(agent_id) <= -2

    # === Reziprozitaets-Tracking (RGS) ===

    def record_share_to(self, target_id, current_tick):
        self.shared_to_history[target_id] = current_tick

    def has_shared_to_recently(self, target_id, current_tick, window=5):
        if target_id not in self.shared_to_history:
            return False
        last_share_tick = self.shared_to_history[target_id]
        return (current_tick - last_share_tick) <= window

    def cleanup_old_shares(self, current_tick, window=5):
        self.shared_to_history = {
            tid: tick for tid, tick in self.shared_to_history.items()
            if (current_tick - tick) <= window
        }

    # === Movement ===

    def decide_move(self, agents_by_pos, current_tick, params):
        """Entscheide ob und wohin sich der Agent bewegt."""
        from . import config

        movement_enabled = params.get('MOVEMENT_ENABLED', False)
        if not movement_enabled:
            return None

        move_cost = params.get('MOVE_COST', 2)
        move_cooldown = params.get('MOVE_COOLDOWN', 3)
        move_threshold = params.get('MOVE_ENERGY_THRESHOLD', 10)
        grid_size = config.GRID_SIZE

        if self.move_cooldown > 0:
            self.move_cooldown -= 1
            return None

        if self.energy < move_threshold + move_cost:
            return None

        empty_neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = (self.x + dx) % grid_size
                ny = (self.y + dy) % grid_size
                if (nx, ny) not in agents_by_pos:
                    empty_neighbors.append((nx, ny))

        if not empty_neighbors:
            return None

        move_scores = params.get('MOVE_SCORES', {
            'A': {'A': -2, 'B': +5, 'C': -1},
            'B': {'A': -3, 'B': +1, 'C': +1},
            'C': {'A': -10, 'B': +2, 'C': +5},
        })

        best_pos = None
        best_score = -999
        current_score = self._evaluate_position(self.x, self.y, agents_by_pos, move_scores, grid_size)

        for (nx, ny) in empty_neighbors:
            score = self._evaluate_position(nx, ny, agents_by_pos, move_scores, grid_size)
            if score > best_score:
                best_score = score
                best_pos = (nx, ny)

        if best_pos and best_score > current_score + 2:
            return best_pos

        return None

    def _evaluate_position(self, x, y, agents_by_pos, move_scores, grid_size):
        """Bewerte eine Position basierend auf Nachbarn."""
        score = 0
        my_type = self.type

        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                nx = (x + dx) % grid_size
                ny = (y + dy) % grid_size

                if (nx, ny) in agents_by_pos:
                    neighbor = agents_by_pos[(nx, ny)]
                    neighbor_type = neighbor.type

                    if my_type in move_scores and neighbor_type in move_scores[my_type]:
                        score += move_scores[my_type][neighbor_type]

        return score

    def execute_move(self, new_x, new_y, params):
        """Fuehre Bewegung aus."""
        move_cost = params.get('MOVE_COST', 2)
        move_cooldown = params.get('MOVE_COOLDOWN', 3)

        self.x = new_x
        self.y = new_y
        self.energy -= move_cost
        self.move_cooldown = move_cooldown
