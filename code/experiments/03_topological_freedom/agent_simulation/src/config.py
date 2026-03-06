"""
Alle Parameter aus Unity Protocol Spec v0.1 §7
Defaults — werden von Experiment-Konfigurationen ueberschrieben.
"""

# Environment Parameters (§7.1)
GRID_SIZE = 32              # N x N
R_MAX = 100                 # Max resources per cell
S_MAX = 100                 # Max degradation per cell
DELTA = 0.05                # Entropy decay rate
MU = 0.1                    # Diffusion rate
G = 0.1                     # Resource growth rate
LAMBDA = 0.5                # Degradation damping on regrowth

# Action Parameters (§7.2)
HARVEST_RATE = 4
SHARE_RATE = 3
STEAL_RATE = 5
DEFEND_BLOCK = 0.7          # d = 70% block
C_STEAL = 1                 # Steal friction cost
C_DEFEND = 1                # Defend energy cost
METABOLIC_COST = 1          # m = 1 per tick
SHARE_THRESHOLD = 5         # Min energy to share

# Entropy Costs (§7.3)
K0 = 1                      # Idle
K3 = 2                      # Share
K2 = 5                      # Harvest
K_DEF = 7                   # Defend
K1 = 10                     # Steal

# Gamma-Controller Parameters (§7.4)
MEMORY_LENGTH = 5           # k
THETA_E = 5                 # Error threshold
THETA_V = 20                # Volatility threshold
INITIAL_ENERGY = 20

# Simulation Parameters (§7.5)
TOTAL_TICKS = 2000
NUM_RUNS = 20
AGENTS_PER_TYPE = 100       # 300 total

# Agent Types
TYPE_A = 'A'  # Tyrant
TYPE_B = 'B'  # Martyr
TYPE_C = 'C'  # Unity

# Experiment 7: Dynamische Trust-basierte Synergie
DYNAMIC_SYNERGY = False
SYNERGY_BASE = 1.0
SYNERGY_FACTOR = 0.15
SYNERGY_MIN = 0.5
SYNERGY_MAX = 1.5

# Experiment 8: RGS (Reciprocity-Gated Synergy)
RGS_ENABLED = False
RGS_WINDOW = 5
RGS_ETA = 1.25
RGS_RHO = 0.90

# Dissonance Penalty
DISSONANCE_PENALTY = False
PENALTY_FACTOR = 2.0

# Policy V2 (Trust/Exklusion)
POLICY_V2 = False

# Experiment 10: Movement (Raeumliche Selektion)
MOVEMENT_ENABLED = False
MOVE_COST = 2
MOVE_COOLDOWN = 3
MOVE_ENERGY_THRESHOLD = 10

MOVE_SCORES = {
    'A': {'A': -2, 'B': +5, 'C': -1},
    'B': {'A': -3, 'B': +1, 'C': +1},
    'C': {'A': -10, 'B': +2, 'C': +5},
}
