"""
Logging aller Metriken pro Tick.
"""

import numpy as np
import pandas as pd
from collections import deque
from .config import TYPE_A, TYPE_B, TYPE_C


class MetricsCollector:
    def __init__(self):
        self.data = []

    def record(self, tick, agents, env, share_log=None):
        """Metriken fuer einen Tick sammeln (Spec §8.2)"""
        counts = {TYPE_A: 0, TYPE_B: 0, TYPE_C: 0}
        energy = {TYPE_A: 0, TYPE_B: 0, TYPE_C: 0}
        gamma_values = []

        for agent in agents:
            if agent.alive:
                counts[agent.type] += 1
                energy[agent.type] += agent.energy
                if agent.type == TYPE_C:
                    gamma_values.append(agent.gamma)

        mean_R = np.mean(env.R)
        mean_S = np.mean(env.S)

        gamma_mean = np.mean(gamma_values) if gamma_values else 0
        gamma_std = np.std(gamma_values) if gamma_values else 0

        record = {
            'tick': tick,
            'count_A': counts[TYPE_A],
            'count_B': counts[TYPE_B],
            'count_C': counts[TYPE_C],
            'energy_A': energy[TYPE_A],
            'energy_B': energy[TYPE_B],
            'energy_C': energy[TYPE_C],
            'mean_R': mean_R,
            'mean_S': mean_S,
            'gamma_mean': gamma_mean,
            'gamma_std': gamma_std,
        }

        if share_log:
            for agent_type in [TYPE_A, TYPE_B, TYPE_C]:
                type_shares = [s for s in share_log if s['sender_type'] == agent_type]

                if type_shares:
                    avg_eta = np.mean([s['eta'] for s in type_shares])
                    total_sent = sum(s['amount'] for s in type_shares)
                    total_received = sum(s['received'] for s in type_shares)
                    efficiency = total_received / total_sent if total_sent > 0 else 0
                    reciprocal_count = sum(1 for s in type_shares if s.get('reciprocal', False))
                    reciprocity_rate = reciprocal_count / len(type_shares)
                    avg_entropy = np.mean([s.get('entropy_cost', 2) for s in type_shares])
                    received_by_type = sum(
                        s['received'] for s in share_log if s['target_type'] == agent_type
                    )
                    net_gain = received_by_type - total_sent
                else:
                    avg_eta = 0
                    efficiency = 0
                    reciprocity_rate = 0
                    avg_entropy = 0
                    net_gain = 0

                record[f'avg_eta_{agent_type}'] = avg_eta
                record[f'share_efficiency_{agent_type}'] = efficiency
                record[f'reciprocity_rate_{agent_type}'] = reciprocity_rate
                record[f'avg_entropy_{agent_type}'] = avg_entropy
                record[f'net_share_gain_{agent_type}'] = net_gain

        self.data.append(record)

    def to_dataframe(self):
        return pd.DataFrame(self.data)

    def save(self, filepath):
        df = self.to_dataframe()
        df.to_csv(filepath, index=False)

    def measure_clusters(self, agents, grid_size=32):
        """
        Misst die durchschnittliche Cluster-Groesse pro Typ.
        Ein Cluster ist eine Gruppe von Agenten desselben Typs,
        die direkt oder indirekt benachbart sind (Moore).
        """
        living = [a for a in agents if a.alive]
        pos_to_agent = {(a.x, a.y): a for a in living}
        visited = set()

        clusters = {TYPE_A: [], TYPE_B: [], TYPE_C: []}

        for agent in living:
            if agent.id in visited:
                continue

            cluster_size = 0
            queue = deque([agent])

            while queue:
                current = queue.popleft()
                if current.id in visited:
                    continue
                if current.type != agent.type:
                    continue

                visited.add(current.id)
                cluster_size += 1

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        nx = (current.x + dx) % grid_size
                        ny = (current.y + dy) % grid_size
                        if (nx, ny) in pos_to_agent:
                            neighbor = pos_to_agent[(nx, ny)]
                            if neighbor.type == agent.type and neighbor.id not in visited:
                                queue.append(neighbor)

            clusters[agent.type].append(cluster_size)

        result = {}
        for t in [TYPE_A, TYPE_B, TYPE_C]:
            if clusters[t]:
                result[f'avg_cluster_{t}'] = np.mean(clusters[t])
                result[f'max_cluster_{t}'] = max(clusters[t])
                result[f'num_clusters_{t}'] = len(clusters[t])
            else:
                result[f'avg_cluster_{t}'] = 0
                result[f'max_cluster_{t}'] = 0
                result[f'num_clusters_{t}'] = 0

        return result
