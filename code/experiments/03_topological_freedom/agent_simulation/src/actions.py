"""
Simultane Aufloesung aller Aktionen.
"""

from .agents import Action
from .config import (
    HARVEST_RATE, SHARE_RATE, STEAL_RATE,
    DEFEND_BLOCK, C_STEAL, C_DEFEND,
    K0, K1, K2, K3, K_DEF,
    DYNAMIC_SYNERGY, SYNERGY_BASE, SYNERGY_FACTOR, SYNERGY_MIN, SYNERGY_MAX,
    RGS_ENABLED, RGS_WINDOW, RGS_ETA, RGS_RHO,
    DISSONANCE_PENALTY, PENALTY_FACTOR
)


class ActionResolver:
    def __init__(self, env, params=None):
        self.env = env
        self.params = params or {}
        self.k1 = self.params.get('K1', K1)
        self.c_steal = self.params.get('C_STEAL', C_STEAL)
        self.steal_rate = self.params.get('STEAL_RATE', STEAL_RATE)
        self.synergy_enabled = self.params.get('SYNERGY_ENABLED', False)
        self.synergy_eta = self.params.get('SYNERGY_ETA', 1.0)
        self.dynamic_synergy = self.params.get('DYNAMIC_SYNERGY', DYNAMIC_SYNERGY)
        self.synergy_base = self.params.get('SYNERGY_BASE', SYNERGY_BASE)
        self.synergy_factor = self.params.get('SYNERGY_FACTOR', SYNERGY_FACTOR)
        self.synergy_min = self.params.get('SYNERGY_MIN', SYNERGY_MIN)
        self.synergy_max = self.params.get('SYNERGY_MAX', SYNERGY_MAX)
        self.rgs_enabled = self.params.get('RGS_ENABLED', RGS_ENABLED)
        self.rgs_window = self.params.get('RGS_WINDOW', RGS_WINDOW)
        self.rgs_eta = self.params.get('RGS_ETA', RGS_ETA)
        self.rgs_rho = self.params.get('RGS_RHO', RGS_RHO)
        self.dissonance_penalty = self.params.get('DISSONANCE_PENALTY', DISSONANCE_PENALTY)
        self.penalty_factor = self.params.get('PENALTY_FACTOR', PENALTY_FACTOR)
        self.current_tick = 0

    def resolve_all(self, decisions, agents_grid):
        """Alle Aktionen simultan aufloesen (Spec §6.1 Step 3-4)"""
        steal_attempts = {}
        defending = set()
        share_log = []

        for agent, (action, target) in decisions.items():
            if not agent.alive:
                continue

            if action == Action.DEFEND:
                defending.add(agent.id)
                self.env.add_entropy(agent.x, agent.y, K_DEF)
                agent.energy -= C_DEFEND

            elif action == Action.STEAL and target is not None:
                if target.id not in steal_attempts:
                    steal_attempts[target.id] = []
                steal_amount = min(self.steal_rate, target.energy)
                steal_attempts[target.id].append((agent, steal_amount))
                self.env.add_entropy(agent.x, agent.y, self.k1)
                agent.energy -= self.c_steal

            elif action == Action.SHARE and target is not None:
                share_amount = min(SHARE_RATE, agent.energy)
                agent.energy -= share_amount

                trust_level = agent.get_trust(target.id)

                if self.rgs_enabled:
                    reciprocal = target.has_shared_to_recently(
                        agent.id, self.current_tick, window=self.rgs_window
                    )
                    eta = self.rgs_eta if reciprocal else self.rgs_rho
                    is_reciprocal = reciprocal
                elif self.dynamic_synergy:
                    eta = self.synergy_base + (trust_level * self.synergy_factor)
                    eta = max(self.synergy_min, min(self.synergy_max, eta))
                    is_reciprocal = False
                elif self.synergy_enabled:
                    eta = self.synergy_eta
                    is_reciprocal = False
                else:
                    eta = 1.0
                    is_reciprocal = False

                received_amount = share_amount * eta
                target.energy += received_amount

                if self.dissonance_penalty and trust_level <= 0:
                    entropy_cost = K3 * self.penalty_factor
                    if trust_level < -1:
                        entropy_cost *= 1.5
                else:
                    entropy_cost = K3

                self.env.add_entropy(agent.x, agent.y, entropy_cost)
                agent.record_share_to(target.id, self.current_tick)

                share_log.append({
                    'sender_id': agent.id,
                    'sender_type': agent.type,
                    'target_id': target.id,
                    'target_type': target.type,
                    'amount': share_amount,
                    'eta': eta,
                    'received': received_amount,
                    'trust': trust_level,
                    'reciprocal': is_reciprocal if self.rgs_enabled else False,
                    'entropy_cost': entropy_cost,
                })

                target.received_share_from.add(agent.id)

            elif action == Action.HARVEST:
                harvested = self.env.extract_resource(agent.x, agent.y, HARVEST_RATE)
                agent.energy += harvested
                self.env.add_entropy(agent.x, agent.y, K2)

            elif action == Action.IDLE:
                self.env.add_entropy(agent.x, agent.y, K0)

        # Resolve Steals
        for victim_id, attacks in steal_attempts.items():
            victim = self._find_agent_by_id(victim_id, agents_grid)
            if victim is None or not victim.alive:
                continue

            is_defending = victim_id in defending

            for attacker, base_amount in attacks:
                if is_defending:
                    effective = base_amount * (1 - DEFEND_BLOCK)
                else:
                    effective = base_amount

                actual_stolen = min(effective, victim.energy)
                victim.energy -= actual_stolen
                attacker.energy += actual_stolen

                victim.record_attack()
                victim.attacked_by.add(attacker.id)

        return share_log

    def _find_agent_by_id(self, agent_id, agents_grid):
        """Hilfsfunktion zum Finden eines Agenten"""
        for row in agents_grid:
            for agent in row:
                if agent is not None and agent.id == agent_id:
                    return agent
        return None
