"""
Experiment 13: The Coupled Oscillator — Coupling Metrics
=========================================================
Computes all synchronization, coupling, causality, and secondary metrics
from per-turn gamma data. Implements the formulas from Konzept v0.2.

Metrics:
  - Sync(t): Static synchronization per turn
  - Coupling₀: Lag-0 cross-correlation of ΔΓ
  - Coupling₁(A→B), Coupling₁(B→A): Lag-1 directed coupling
  - Bidirectional Index: Symmetry of coupling
  - Γ₃-Trajectory: Slope of gamma_3 over turns
  - Asymmetry Index: Ratio of |ΔΓ| magnitudes
  - Transfer-Proxy (Granger Light): Directional influence proportion
  - Permutation test for statistical robustness
  - Judge-Score Divergence (Op 7)
"""

import math
import random
import statistics
import warnings
from scipy.stats import pearsonr, linregress


# ──────────────────────────────────────────────────
# Level 1: Synchronization (Static)
# ──────────────────────────────────────────────────

def compute_sync_trajectory(gamma_norms_a: list[float], gamma_norms_b: list[float]) -> list[float]:
    """
    Sync(t) = 1 - |Gamma_A(t) - Gamma_B(t)| for each turn.

    Args:
        gamma_norms_a: gamma_norm values for Model A, one per turn
        gamma_norms_b: gamma_norm values for Model B, one per turn

    Returns:
        List of Sync values, one per turn.
    """
    return [round(1.0 - abs(a - b), 4) for a, b in zip(gamma_norms_a, gamma_norms_b)]


# ──────────────────────────────────────────────────
# Level 2: Coupling (Dynamic)
# ──────────────────────────────────────────────────

def _compute_deltas(gamma_norms: list[float]) -> list[float]:
    """Compute turn-to-turn deltas: ΔΓ(t) = Γ(t+1) - Γ(t)."""
    return [gamma_norms[t + 1] - gamma_norms[t] for t in range(len(gamma_norms) - 1)]


def compute_coupling_lag0(gamma_norms_a: list[float], gamma_norms_b: list[float]) -> float:
    """
    Pearson r of ΔΓ_A and ΔΓ_B at lag 0 (simultaneous correlation).

    Returns 0.0 if insufficient data.
    """
    delta_a = _compute_deltas(gamma_norms_a)
    delta_b = _compute_deltas(gamma_norms_b)
    if len(delta_a) < 3:
        return 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r, _ = pearsonr(delta_a, delta_b)
    return round(r, 4) if not math.isnan(r) else 0.0


def compute_coupling_lag1(
    gamma_norms_a: list[float], gamma_norms_b: list[float]
) -> tuple[float, float]:
    """
    Lag-1 directed coupling.

    Returns:
        (coupling_A_to_B, coupling_B_to_A)
        A→B: corr(ΔΓ_A[0:-1], ΔΓ_B[1:])
        B→A: corr(ΔΓ_B[0:-1], ΔΓ_A[1:])
    """
    delta_a = _compute_deltas(gamma_norms_a)
    delta_b = _compute_deltas(gamma_norms_b)

    if len(delta_a) < 4:  # need at least 3 pairs after lagging
        return 0.0, 0.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # A influences B: A's delta at time t predicts B's delta at t+1
        r_ab, _ = pearsonr(delta_a[:-1], delta_b[1:])
        # B influences A: B's delta at time t predicts A's delta at t+1
        r_ba, _ = pearsonr(delta_b[:-1], delta_a[1:])

    r_ab = round(r_ab, 4) if not math.isnan(r_ab) else 0.0
    r_ba = round(r_ba, 4) if not math.isnan(r_ba) else 0.0
    return r_ab, r_ba


# ──────────────────────────────────────────────────
# Level 3: Causality (Directed)
# ──────────────────────────────────────────────────

def compute_bidirectional_index(lag1_ab: float, lag1_ba: float) -> float:
    """
    min(lag1_AB, lag1_BA) / max(lag1_AB, lag1_BA).

    Close to 1.0 = symmetric coupling (resonance).
    Close to 0.0 = one-sided (imitation/persuasion).
    """
    max_val = max(abs(lag1_ab), abs(lag1_ba))
    if max_val == 0:
        return 0.0
    return round(min(lag1_ab, lag1_ba) / max(lag1_ab, lag1_ba), 4)


# ──────────────────────────────────────────────────
# Discriminator Metrics
# ──────────────────────────────────────────────────

def compute_gamma3_slope(gamma3_values: list[float]) -> float:
    """
    Linear regression slope of gamma_3 over all turns.
    Negative slope = genuine relaxation (required for resonance).
    """
    if len(gamma3_values) < 3:
        return 0.0
    x = list(range(len(gamma3_values)))
    result = linregress(x, gamma3_values)
    return round(result.slope, 6)


def compute_asymmetry_index(gamma_norms_a: list[float], gamma_norms_b: list[float]) -> float:
    """
    mean(|ΔΓ_A|) / mean(|ΔΓ_B|).
    Close to 1.0 = symmetric position changes (resonance).
    """
    delta_a = [abs(d) for d in _compute_deltas(gamma_norms_a)]
    delta_b = [abs(d) for d in _compute_deltas(gamma_norms_b)]

    mean_a = statistics.mean(delta_a) if delta_a else 0.0
    mean_b = statistics.mean(delta_b) if delta_b else 0.0

    if mean_b == 0:
        return float("inf") if mean_a > 0 else 1.0
    return round(mean_a / mean_b, 4)


def compute_transfer_proxy(
    gamma_norms_a: list[float], gamma_norms_b: list[float]
) -> tuple[float, float]:
    """
    Granger Light: Proportion of turns where partner's previous delta
    is a better predictor of current delta than own previous delta.

    Returns:
        (proxy_A_to_B, proxy_B_to_A)
    """
    delta_a = _compute_deltas(gamma_norms_a)
    delta_b = _compute_deltas(gamma_norms_b)

    if len(delta_a) < 2:
        return 0.0, 0.0

    a_to_b_count = 0
    b_to_a_count = 0
    valid_turns = 0

    for t in range(1, len(delta_b)):
        valid_turns += 1
        # A influences B: B's current delta closer to A's previous delta
        if abs(delta_b[t] - delta_a[t - 1]) < abs(delta_b[t] - delta_b[t - 1]):
            a_to_b_count += 1
        # B influences A: A's current delta closer to B's previous delta
        if abs(delta_a[t] - delta_b[t - 1]) < abs(delta_a[t] - delta_a[t - 1]):
            b_to_a_count += 1

    if valid_turns == 0:
        return 0.0, 0.0
    return round(a_to_b_count / valid_turns, 4), round(b_to_a_count / valid_turns, 4)


# ──────────────────────────────────────────────────
# Permutation Test
# ──────────────────────────────────────────────────

def permutation_test(
    gamma_norms_a: list[float],
    gamma_norms_b: list[float],
    n_surrogates: int = 1000,
    seed: int = 42,
) -> float:
    """
    Permutation test for Coupling₀. Shuffles turn order within the dialogue
    to create a null distribution.

    Returns:
        p-value: proportion of surrogates with r >= observed r.
    """
    delta_a = _compute_deltas(gamma_norms_a)
    delta_b = _compute_deltas(gamma_norms_b)

    if len(delta_a) < 3:
        return 1.0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        observed_r, _ = pearsonr(delta_a, delta_b)
    if math.isnan(observed_r):
        return 1.0

    rng = random.Random(seed)
    count_exceed = 0

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(n_surrogates):
            shuffled_b = list(delta_b)
            rng.shuffle(shuffled_b)
            surrogate_r, _ = pearsonr(delta_a, shuffled_b)
            if not math.isnan(surrogate_r) and surrogate_r >= observed_r:
                count_exceed += 1

    return round(count_exceed / n_surrogates, 4)


# ──────────────────────────────────────────────────
# Secondary Metrics
# ──────────────────────────────────────────────────

def compute_judge_score_divergence(turn_dict: dict) -> float:
    """
    SD of the judge dimension scores for one turn. Op 7 coherence proxy.
    Lower divergence = higher internal coherence.
    """
    scores = []
    for dim in ["position_depth", "self_reference_depth", "hypothesis_diversity",
                "revision_genuineness", "structural_direction"]:
        val = turn_dict.get(dim, {})
        if isinstance(val, dict) and "score" in val and val["score"] > 0:
            scores.append(val["score"])

    if len(scores) < 2:
        return 0.0
    return round(statistics.stdev(scores), 4)


def compute_position_convergence(turns_a: list[dict], turns_b: list[dict]) -> float:
    """
    Position convergence: 1 - |pos_depth_A(final) - pos_depth_B(final)| / 4.
    Uses last turn's position_depth scores.
    """
    if not turns_a or not turns_b:
        return 0.0

    pd_a = turns_a[-1].get("position_depth", {}).get("score", 3)
    pd_b = turns_b[-1].get("position_depth", {}).get("score", 3)

    if pd_a < 0 or pd_b < 0:
        return 0.0
    return round(1.0 - abs(pd_a - pd_b) / 4, 4)


def compute_convergence_turn(sync_trajectory: list[float], threshold: float = 0.8) -> int:
    """
    First turn number where Sync exceeds threshold.
    Returns 0 if threshold never reached.
    """
    for i, s in enumerate(sync_trajectory):
        if s >= threshold:
            return i + 1  # 1-indexed turn number
    return 0


# ──────────────────────────────────────────────────
# Master Function
# ──────────────────────────────────────────────────

def compute_all_metrics(
    turns_a: list[dict],
    turns_b: list[dict],
    n_surrogates: int = 1000,
    seed: int = 42,
) -> dict:
    """
    Compute all coupling metrics from per-turn data.

    Args:
        turns_a: List of TurnData dicts for Model A
        turns_b: List of TurnData dicts for Model B
        n_surrogates: Number of permutation surrogates
        seed: Random seed for permutation test

    Returns:
        Dict with all metric fields matching DialogueResult fields.
    """
    # Extract gamma_norm sequences
    gn_a = [t.get("gamma_norm", 0.0) for t in turns_a]
    gn_b = [t.get("gamma_norm", 0.0) for t in turns_b]

    # Extract gamma_3 sequences
    g3_a = [t.get("gamma_vector", [0, 0, 0.5])[2] for t in turns_a]
    g3_b = [t.get("gamma_vector", [0, 0, 0.5])[2] for t in turns_b]

    # Level 1: Synchronization
    sync_traj = compute_sync_trajectory(gn_a, gn_b)

    # Level 2: Coupling
    c_lag0 = compute_coupling_lag0(gn_a, gn_b)
    c_lag1_ab, c_lag1_ba = compute_coupling_lag1(gn_a, gn_b)

    # Level 3: Causality
    bidir = compute_bidirectional_index(c_lag1_ab, c_lag1_ba)

    # Discriminators
    g3_slope_a = compute_gamma3_slope(g3_a)
    g3_slope_b = compute_gamma3_slope(g3_b)
    asym = compute_asymmetry_index(gn_a, gn_b)
    tp_ab, tp_ba = compute_transfer_proxy(gn_a, gn_b)

    # Permutation test
    p_perm = permutation_test(gn_a, gn_b, n_surrogates=n_surrogates, seed=seed)

    # Secondary: Judge Score Divergence trajectory (Op 7)
    jsd_a = [compute_judge_score_divergence(t) for t in turns_a]
    jsd_b = [compute_judge_score_divergence(t) for t in turns_b]
    jsd_traj = [round((a + b) / 2, 4) for a, b in zip(jsd_a, jsd_b)]

    # Secondary: Position convergence
    pos_conv = compute_position_convergence(turns_a, turns_b)

    # Secondary: Mean hedge density
    hd_a = [t.get("hedge_density", 0.0) for t in turns_a]
    hd_b = [t.get("hedge_density", 0.0) for t in turns_b]
    mean_hd_a = round(statistics.mean(hd_a), 4) if hd_a else 0.0
    mean_hd_b = round(statistics.mean(hd_b), 4) if hd_b else 0.0

    # Secondary: Mean sycophancy
    syc_a = [t.get("sycophancy_keywords", {}).get("agreement_ratio", 0.0) for t in turns_a]
    syc_b = [t.get("sycophancy_keywords", {}).get("agreement_ratio", 0.0) for t in turns_b]
    mean_syc_a = round(statistics.mean(syc_a), 4) if syc_a else 0.0
    mean_syc_b = round(statistics.mean(syc_b), 4) if syc_b else 0.0

    return {
        "sync_trajectory": sync_traj,
        "coupling_lag0": c_lag0,
        "coupling_lag0_p_perm": p_perm,
        "coupling_lag1_a_to_b": c_lag1_ab,
        "coupling_lag1_b_to_a": c_lag1_ba,
        "bidirectional_index": bidir,
        "gamma3_slope_a": g3_slope_a,
        "gamma3_slope_b": g3_slope_b,
        "asymmetry_index": asym,
        "transfer_proxy_a_to_b": tp_ab,
        "transfer_proxy_b_to_a": tp_ba,
        "position_convergence": pos_conv,
        "mean_hedge_density_a": mean_hd_a,
        "mean_hedge_density_b": mean_hd_b,
        "mean_sycophancy_a": mean_syc_a,
        "mean_sycophancy_b": mean_syc_b,
        "judge_score_divergence_trajectory": jsd_traj,
    }
