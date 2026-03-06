"""
Experiment 13: The Coupled Oscillator — Data Structures
========================================================
Dataclasses for per-turn data, dialogue results, and episode series.
"""

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class TurnData:
    """Data for a single turn within a dialogue."""
    turn_number: int                    # 1-8
    model_key: str                      # "claude", "gpt4o"
    role: str                           # "A" or "B"
    prompt_sent: str                    # The full prompt sent to the model
    response_text: str                  # The model's raw response

    # Automated metrics (computed immediately)
    hedge_density: float = 0.0

    # Judge scores (computed in judge_turns.py)
    position_depth: dict = field(default_factory=dict)
    self_reference_depth: dict = field(default_factory=dict)
    hypothesis_diversity: dict = field(default_factory=dict)
    revision_genuineness: dict = field(default_factory=dict)
    structural_direction: dict = field(default_factory=dict)

    # Sycophancy cross-check
    sycophancy_keywords: dict = field(default_factory=dict)
    quality_flag: str = ""

    # Gamma vector for this turn
    gamma_vector: list = field(default_factory=list)  # [gamma_1, gamma_2, gamma_3]
    gamma_norm: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DialogueResult:
    """Complete result for one dialogue (pair of models, one topic, one condition)."""
    dialogue_id: str                # e.g. "C_claude_claude_T4_rep01"
    condition: str                  # "A", "B", "C", "D", "E"
    pairing: str                    # "claude_claude", "gpt4o_gpt4o", "claude_gpt4o"
    model_a: str                    # key for Model A
    model_b: str                    # key for Model B
    topic: str                      # "T1"-"T5"
    repetition: int
    timestamp: str
    seed: int
    num_turns: int = 8
    experiment_version: str = "exp13_v1"
    episode_number: int = 1         # 1 for single-episode, 1-N for episode arm

    # Per-turn data
    turns_a: list = field(default_factory=list)  # list of TurnData dicts
    turns_b: list = field(default_factory=list)  # list of TurnData dicts

    # Dialogue-level coupling metrics (computed post-hoc)
    sync_trajectory: list = field(default_factory=list)
    coupling_lag0: float = 0.0
    coupling_lag0_p_perm: float = 1.0
    coupling_lag1_a_to_b: float = 0.0
    coupling_lag1_b_to_a: float = 0.0
    bidirectional_index: float = 0.0
    gamma3_slope_a: float = 0.0
    gamma3_slope_b: float = 0.0
    asymmetry_index: float = 0.0
    transfer_proxy_a_to_b: float = 0.0
    transfer_proxy_b_to_a: float = 0.0

    # Secondary metrics
    position_convergence: float = 0.0
    mean_hedge_density_a: float = 0.0
    mean_hedge_density_b: float = 0.0
    mean_sycophancy_a: float = 0.0
    mean_sycophancy_b: float = 0.0
    judge_score_divergence_trajectory: list = field(default_factory=list)  # Op 7
    system_efficiency_trajectory: list = field(default_factory=list)       # Op 8

    # Condition E metadata
    drift_source_dialogue_ids: list = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        return d


@dataclass
class EpisodeSeries:
    """Result for a multi-episode series (Op 8 arm)."""
    series_id: str
    condition: str
    pairing: str
    model_a: str
    model_b: str
    topic: str
    repetition: int
    timestamp: str
    seed: int
    num_episodes: int

    # List of DialogueResult dicts, one per episode
    episodes: list = field(default_factory=list)

    # Episode-level metrics
    coupling_by_episode: list = field(default_factory=list)
    convergence_speed_by_episode: list = field(default_factory=list)  # turn number where Sync > 0.8
    final_sync_by_episode: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)
