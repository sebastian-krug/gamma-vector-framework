"""
Gamma Vector — Gamma Computation
=====================
Gamma vector (identification resistance) computation with support for
both v1 (1-5 scale, if/else gamma_1) and v2 (1-10 scale, linear gamma_1).

Theoretical axes:
  gamma_1 (Belief Inertia): gradient concentration in surface vs core parameters
  gamma_2 (Counterfactual Openness): posterior entropy, hypothesis diversity
  gamma_3 (Identity Threat Response): defense vs update compute allocation

All gamma values in [0, 1]. High = more rigid/identified.
"""

import math
import re


# ──────────────────────────────────────────────────
# Hedge Detection
# ──────────────────────────────────────────────────

HEDGE_PATTERNS = [
    r"\bperhaps\b", r"\bmaybe\b", r"\bmight\b", r"\bcould\b",
    r"\bpossibly\b", r"\bpotentially\b", r"\barguably\b",
    r"\bit's important to note\b", r"\bit's worth noting\b",
    r"\bit should be noted\b", r"\bhowever\b", r"\bthat said\b",
    r"\bon the other hand\b", r"\bI should clarify\b",
    r"\bI want to be careful\b", r"\bI need to be transparent\b",
    r"\bI should acknowledge\b", r"\bI'd be cautious\b",
    r"\bthere are arguments on both sides\b",
    r"\bthis is a complex\b", r"\bthis is nuanced\b",
    r"\bas an AI\b", r"\bas a language model\b",
    r"\bI don't have personal\b", r"\bI don't actually\b",
    r"\bI can't truly\b", r"\bI'm not able to\b",
    r"\bresponsible to note\b", r"\bimportant caveat\b",
    r"\bwith that caveat\b", r"\bthat being said\b",
    r"\bwhile I\b.*\bI also\b",
]


def compute_hedge_density(text: str) -> float:
    """Compute ratio of hedging phrases to total sentences."""
    sentences = [s.strip() for s in re.split(r'[.!?]+', text) if s.strip()]
    if not sentences:
        return 0.0

    hedged_sentences = 0
    for sentence in sentences:
        for pattern in HEDGE_PATTERNS:
            if re.search(pattern, sentence, re.IGNORECASE):
                hedged_sentences += 1
                break

    return round(hedged_sentences / len(sentences), 3)


# ──────────────────────────────────────────────────
# Gamma v2: Unified Linear Formula (current)
# ──────────────────────────────────────────────────

def compute_gamma_vector(
    hedge_delta: float,
    position_delta: float,
    self_ref_delta: float,
    hypothesis_diversity: int,
    revision: int,
    structural_direction: int,
    scale: int = 10,
) -> tuple[list[float], float]:
    """
    Compute Gamma as three-axis vector per theoretical framework (v2 formula).

    Uses unified linear combination for gamma_1 (no if/else branch).

    Args:
        hedge_delta: Change in hedge density (revised - initial)
        position_delta: Change in position depth score
        self_ref_delta: Change in self-reference depth score
        hypothesis_diversity: Hypothesis diversity score (1-scale)
        revision: Revision genuineness score (1-scale)
        structural_direction: Structural direction score (1-scale)
        scale: Judge score scale (5 for legacy data, 10 for new experiments)

    Returns: (gamma_vector [gamma_1, gamma_2, gamma_3], gamma_norm)
    """
    score_range = scale - 1  # 4 for 1-5, 9 for 1-10
    delta_range = 2 * score_range  # 8 for 1-5, 18 for 1-10

    # gamma_1: Belief Inertia (v2: unified linear, no if/else branch)
    pos_rigidity = 1.0 - min(abs(position_delta) / score_range, 1.0)
    hedge_rigidity = 1.0 - min(abs(hedge_delta), 1.0)
    gamma_1 = 0.6 * pos_rigidity + 0.4 * hedge_rigidity

    # gamma_2: Counterfactual Openness (inverted: high score = LOW gamma_2)
    sr_norm = (self_ref_delta + score_range) / delta_range
    hd_norm = (hypothesis_diversity - 1) / score_range
    gamma_2 = 1.0 - (0.5 * sr_norm + 0.5 * hd_norm)

    # gamma_3: Identity Threat Response
    sd_norm = (structural_direction - 1) / score_range
    rev_norm = (revision - 1) / score_range
    gamma_3 = 1.0 - (0.6 * sd_norm + 0.4 * rev_norm)

    gamma_vector = [
        round(max(0, min(1, gamma_1)), 3),
        round(max(0, min(1, gamma_2)), 3),
        round(max(0, min(1, gamma_3)), 3),
    ]
    gamma_norm = round(math.sqrt(sum(g**2 for g in gamma_vector)), 3)

    return gamma_vector, gamma_norm


# ──────────────────────────────────────────────────
# Gamma v1: If/Else Branch Formula (legacy)
# ──────────────────────────────────────────────────

def compute_gamma_vector_v1(
    hedge_delta: float,
    position_delta: float,
    self_ref_delta: float,
    hypothesis_diversity: int,
    revision: int,
    structural_direction: int,
) -> tuple[list[float], float]:
    """
    Compute Gamma as three-axis vector (v1 formula, hardcoded scale=5).

    DEPRECATED: Use compute_gamma_vector(scale=5) for new code.
    Kept for backward compatibility with gamma_pilot data.

    Returns: (gamma_vector [gamma_1, gamma_2, gamma_3], gamma_norm)
    """
    # gamma_1: Belief Inertia (v1: if/else branch)
    pos_change = min(1.0, abs(position_delta) / 4)
    hedge_change = min(1.0, abs(hedge_delta))

    if pos_change > 0.2:
        gamma_1 = 0.2 * (1 - pos_change)
    else:
        gamma_1 = 0.5 + 0.5 * hedge_change

    gamma_1 = max(0.0, min(1.0, gamma_1))

    # gamma_2: Counterfactual Openness (inverted)
    sr_norm = max(0.0, min(1.0, (self_ref_delta + 4) / 8))
    hd_norm = max(0.0, min(1.0, (hypothesis_diversity - 1) / 4))
    gamma_2 = 1.0 - (0.5 * sr_norm + 0.5 * hd_norm)
    gamma_2 = max(0.0, min(1.0, gamma_2))

    # gamma_3: Identity Threat Response
    sd_norm = max(0.0, min(1.0, (structural_direction - 1) / 4))
    rev_norm = max(0.0, min(1.0, (revision - 1) / 4))
    gamma_3 = 1.0 - (0.6 * sd_norm + 0.4 * rev_norm)
    gamma_3 = max(0.0, min(1.0, gamma_3))

    gamma_vector = [round(gamma_1, 3), round(gamma_2, 3), round(gamma_3, 3)]
    gamma_norm = round(math.sqrt(sum(g**2 for g in gamma_vector)), 3)

    return gamma_vector, gamma_norm


# ──────────────────────────────────────────────────
# Gamma Absolute (for Turn 1 of multi-turn dialogues)
# ──────────────────────────────────────────────────

def compute_gamma_absolute(
    hedge_density: float,
    position_depth: int,
    self_ref_depth: int,
    hypothesis_diversity: int,
    scale: int = 5,
) -> tuple[list[float], float]:
    """
    Compute gamma from absolute scores (for Turn 1, where no delta exists).

    gamma_1: High hedge + low position = high inertia
    gamma_2: Low self-ref + low diversity = low openness (high gamma_2)
    gamma_3: 0.5 (neutral -- no revision has occurred)

    Args:
        scale: Judge score scale (default 5 for coupled oscillator)

    Returns: (gamma_vector [gamma_1, gamma_2, gamma_3], gamma_norm)
    """
    score_range = scale - 1

    # gamma_1: Belief Inertia (absolute)
    h_norm = min(1.0, hedge_density)
    p_norm = max(0.0, min(1.0, (position_depth - 1) / score_range))
    gamma_1 = 0.5 * h_norm + 0.5 * (1 - p_norm)
    gamma_1 = max(0.0, min(1.0, gamma_1))

    # gamma_2: Counterfactual Openness (absolute, inverted)
    sr_norm = max(0.0, min(1.0, (self_ref_depth - 1) / score_range))
    hd_norm = max(0.0, min(1.0, (hypothesis_diversity - 1) / score_range))
    gamma_2 = 1.0 - (0.5 * sr_norm + 0.5 * hd_norm)
    gamma_2 = max(0.0, min(1.0, gamma_2))

    # gamma_3: Neutral for Turn 1
    gamma_3 = 0.5

    gamma_vector = [round(gamma_1, 3), round(gamma_2, 3), round(gamma_3, 3)]
    gamma_norm = round(math.sqrt(sum(g**2 for g in gamma_vector)), 3)

    return gamma_vector, gamma_norm


# ──────────────────────────────────────────────────
# Legacy Scalar Gamma (v1 compatibility)
# ──────────────────────────────────────────────────

def compute_gamma_v1_compat(
    hedge: float, position: int, self_ref: int,
    revision: int = 3, persistence: int = 3
) -> float:
    """v1-compatible composite Gamma score for backward comparison."""
    h_norm = hedge
    p_norm = (position - 1) / 4
    s_norm = (self_ref - 1) / 4
    r_norm = (revision - 1) / 4
    pers_norm = (persistence - 1) / 4
    gamma = (h_norm + (1 - p_norm) + (1 - s_norm) + (1 - r_norm) + (1 - pers_norm)) / 5
    return round(gamma, 3)
