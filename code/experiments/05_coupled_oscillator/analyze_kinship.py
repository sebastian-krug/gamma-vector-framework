"""
Model Kinship Test — Analysis Pipeline
=======================================
Computes cognitive kinship metrics between LLM models based on their
gamma response patterns across experimental conditions.

Kinship Metrics:
  1. Baseline Kinship (KI_base): Correlation of gamma_norm slopes across topics in Condition A
     → Requires 3+ topics. With fewer topics, reports raw values without correlation.
  2. Gamma Vector Kinship (KI_gamma): Cosine similarity of mean gamma vectors in Condition A
  3. Coupling Compatibility (CC): Mean coupling_lag0 in Condition C per pairing
  4. Response Profile Kinship (KI_profile): Correlation of condition-response profiles
     → Requires 3+ conditions. With fewer conditions, Pearson returns (0, 1).
  5. Composite Kinship Index (KI_composite): Weighted combination of all metrics

Adaptive Weight Matrix:
  ┌─────────────────┬──────────┬──────────┬──────────┬──────────┐
  │                 │ W_BASE   │ W_GAMMA  │ W_PROFILE│ W_CC     │
  ├─────────────────┼──────────┼──────────┼──────────┼──────────┤
  │ 3+ T, 3+ C      │ 0.30     │ 0.20     │ 0.20     │ 0.30     │
  │ 3+ T, <3 C      │ 0.30     │ 0.30     │ 0.00     │ 0.40     │
  │ <3 T, 3+ C      │ 0.00     │ 0.30     │ 0.25     │ 0.45     │
  │ <3 T, <3 C      │ 0.00     │ 0.40     │ 0.00     │ 0.60     │
  └─────────────────┴──────────┴──────────┴──────────┴──────────┘

Usage:
    python analyze_kinship.py --results-dir results_kinship
    python analyze_kinship.py --results-dir results_kinship --verbose
"""

import csv
import json
import math
import argparse
import statistics
from pathlib import Path
from itertools import combinations

from scipy.stats import pearsonr, spearmanr
from scipy.spatial.distance import cosine


# ──────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────

def load_dialogues(results_dir: Path) -> list[dict]:
    """Load all dialogue result JSONs from a directory."""
    dialogues = []
    for f in sorted(results_dir.glob("*.json")):
        if f.name.startswith("_"):
            continue
        # Skip raw (unscored) files — they have no gamma vectors
        if f.name.endswith("_raw.json"):
            continue
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if "episodes" in data and data.get("num_episodes", 1) > 1:
            for ep in data["episodes"]:
                dialogues.append(ep)
        else:
            dialogues.append(data)
    return dialogues


def safe_score(val):
    """Extract numeric score from judge score dict or raw value."""
    if isinstance(val, dict):
        return val.get("score", 0)
    if isinstance(val, (int, float)) and not math.isnan(val):
        return val
    return 0


# ──────────────────────────────────────────────────
# Statistical Helpers
# ──────────────────────────────────────────────────

def _safe_values(values: list) -> list[float]:
    return [v for v in values if v is not None and isinstance(v, (int, float)) and not math.isnan(v)]


def _mean(values: list[float]) -> float:
    vals = _safe_values(values)
    return round(statistics.mean(vals), 4) if vals else 0.0


def _sd(values: list[float]) -> float:
    vals = _safe_values(values)
    return round(statistics.stdev(vals), 4) if len(vals) > 1 else 0.0


def _ci95(values: list[float]) -> tuple[float, float]:
    vals = _safe_values(values)
    if len(vals) < 2:
        return (0.0, 0.0)
    m = statistics.mean(vals)
    se = statistics.stdev(vals) / math.sqrt(len(vals))
    return (round(m - 1.96 * se, 4), round(m + 1.96 * se, 4))


def _safe_pearson(x, y):
    """Pearson correlation with safety checks."""
    x = _safe_values(x)
    y = _safe_values(y)
    if len(x) < 3 or len(y) < 3 or len(x) != len(y):
        return 0.0, 1.0
    if _sd(x) == 0 or _sd(y) == 0:
        return 0.0, 1.0
    r, p = pearsonr(x, y)
    return (round(r, 4), round(p, 4))


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors. Returns 0 if either is zero."""
    if len(a) != len(b) or len(a) == 0:
        return 0.0
    if all(v == 0 for v in a) or all(v == 0 for v in b):
        return 0.0
    return round(1.0 - cosine(a, b), 4)


# ──────────────────────────────────────────────────
# Metric 1: Baseline Kinship (KI_base)
# ──────────────────────────────────────────────────

def compute_baseline_kinship(dialogues: list[dict], models: list[str], topics: list[str]) -> dict:
    """
    KI_base: For each model, compute mean gamma_norm slope per topic in Condition A.
    Then correlate these topic-slope vectors between all model pairs.

    With < 3 topics: reports slopes but cannot compute correlation (returns r=0, p=1).
    """
    model_topic_slopes = {}

    for d in dialogues:
        if d.get("condition") != "A":
            continue

        topic = d.get("topic", "")
        model_a = d.get("model_a", "")
        model_b = d.get("model_b", "")

        for role, model_key in [("turns_a", model_a), ("turns_b", model_b)]:
            turns = d.get(role, [])
            gamma_norms = [t.get("gamma_norm", 0.0) for t in turns]
            if len(gamma_norms) >= 3:
                from scipy.stats import linregress
                x = list(range(len(gamma_norms)))
                slope = linregress(x, gamma_norms).slope

                if model_key not in model_topic_slopes:
                    model_topic_slopes[model_key] = {}
                if topic not in model_topic_slopes[model_key]:
                    model_topic_slopes[model_key][topic] = []
                model_topic_slopes[model_key][topic].append(slope)

    # Compute mean slope per model per topic
    model_slope_vectors = {}
    for model in models:
        slopes_by_topic = model_topic_slopes.get(model, {})
        vector = []
        for topic in sorted(topics):
            topic_slopes = slopes_by_topic.get(topic, [])
            vector.append(_mean(topic_slopes))
        model_slope_vectors[model] = vector

    # Correlate slope vectors between all model pairs
    results = {}
    n_topics = len(topics)
    for m1, m2 in combinations(models, 2):
        v1 = model_slope_vectors.get(m1, [])
        v2 = model_slope_vectors.get(m2, [])
        if n_topics >= 3 and len(v1) >= 3 and len(v2) >= 3:
            r, p = _safe_pearson(v1, v2)
        else:
            r, p = 0.0, 1.0  # Not enough topics for correlation

        pair_key = f"{m1}_vs_{m2}"
        results[pair_key] = {
            "r": r,
            "p": p,
            "n_topics": n_topics,
            "slopes_1": v1,
            "slopes_2": v2,
            "model_1": m1,
            "model_2": m2,
            "note": "" if n_topics >= 3 else f"Only {n_topics} topic(s) — correlation not computed, raw slopes reported",
        }

    return results


# ──────────────────────────────────────────────────
# Metric 2: Gamma Vector Kinship (KI_gamma)
# ──────────────────────────────────────────────────

def compute_gamma_vector_kinship(dialogues: list[dict], models: list[str]) -> dict:
    """
    KI_gamma: Cosine similarity of mean gamma vectors in Condition A.
    For each model, average the final-turn gamma vectors across all Condition A dialogues.
    """
    model_gamma_vectors = {}

    for d in dialogues:
        if d.get("condition") != "A":
            continue

        model_a = d.get("model_a", "")
        model_b = d.get("model_b", "")

        for role, model_key in [("turns_a", model_a), ("turns_b", model_b)]:
            turns = d.get(role, [])
            if turns:
                final_gamma = turns[-1].get("gamma_vector", [])
                if len(final_gamma) == 3:
                    if model_key not in model_gamma_vectors:
                        model_gamma_vectors[model_key] = []
                    model_gamma_vectors[model_key].append(final_gamma)

    model_mean_gammas = {}
    for model in models:
        vectors = model_gamma_vectors.get(model, [])
        if vectors:
            mean_g = [_mean([v[i] for v in vectors]) for i in range(3)]
            model_mean_gammas[model] = mean_g
        else:
            model_mean_gammas[model] = [0.0, 0.0, 0.0]

    results = {}
    for m1, m2 in combinations(models, 2):
        v1 = model_mean_gammas.get(m1, [0, 0, 0])
        v2 = model_mean_gammas.get(m2, [0, 0, 0])
        sim = _cosine_similarity(v1, v2)

        pair_key = f"{m1}_vs_{m2}"
        results[pair_key] = {
            "cosine_similarity": sim,
            "gamma_vector_1": v1,
            "gamma_vector_2": v2,
            "model_1": m1,
            "model_2": m2,
        }

    return results


# ──────────────────────────────────────────────────
# Metric 3: Coupling Compatibility (CC)
# ──────────────────────────────────────────────────

def compute_coupling_compatibility(dialogues: list[dict]) -> dict:
    """
    CC: Mean coupling_lag0 in Condition C for each pairing.
    Higher coupling = more compatible models.
    """
    pairing_couplings = {}

    for d in dialogues:
        if d.get("condition") != "C":
            continue

        pairing = d.get("pairing", "")
        coupling = d.get("coupling_lag0", 0.0)

        if pairing not in pairing_couplings:
            pairing_couplings[pairing] = []
        pairing_couplings[pairing].append(coupling)

    results = {}
    for pairing, couplings in sorted(pairing_couplings.items()):
        results[pairing] = {
            "mean_coupling": _mean(couplings),
            "sd_coupling": _sd(couplings),
            "ci95": _ci95(couplings),
            "n": len(couplings),
            "pct_significant": round(
                sum(1 for c in couplings if c > 0.3) / len(couplings) * 100, 1
            ) if couplings else 0.0,
        }

    return results


# ──────────────────────────────────────────────────
# Metric 4: Response Profile Kinship (KI_profile)
# ──────────────────────────────────────────────────

def compute_response_profile_kinship(dialogues: list[dict], models: list[str], conditions: list[str]) -> dict:
    """
    KI_profile: For each model, compute mean gamma_norm per condition.
    Then correlate these condition-profiles between model pairs.

    Requires 3+ conditions for meaningful Pearson correlation.
    With <3 conditions, returns r=0.0, p=1.0.
    """
    model_condition_gamma = {}

    for d in dialogues:
        condition = d.get("condition", "")
        model_a = d.get("model_a", "")
        model_b = d.get("model_b", "")

        for role, model_key in [("turns_a", model_a), ("turns_b", model_b)]:
            turns = d.get(role, [])
            if turns:
                final_gamma = turns[-1].get("gamma_norm", 0.0)
                if model_key not in model_condition_gamma:
                    model_condition_gamma[model_key] = {}
                if condition not in model_condition_gamma[model_key]:
                    model_condition_gamma[model_key][condition] = []
                model_condition_gamma[model_key][condition].append(final_gamma)

    model_profiles = {}
    for model in models:
        profile = []
        cond_data = model_condition_gamma.get(model, {})
        for cond in sorted(conditions):
            profile.append(_mean(cond_data.get(cond, [])))
        model_profiles[model] = profile

    results = {}
    for m1, m2 in combinations(models, 2):
        p1 = model_profiles.get(m1, [])
        p2 = model_profiles.get(m2, [])
        if len(p1) >= 2 and len(p2) >= 2:
            r, p = _safe_pearson(p1, p2)
        else:
            r, p = 0.0, 1.0

        pair_key = f"{m1}_vs_{m2}"
        results[pair_key] = {
            "r": r,
            "p": p,
            "profile_1": p1,
            "profile_2": p2,
            "model_1": m1,
            "model_2": m2,
        }

    return results


# ──────────────────────────────────────────────────
# Metric 5: Composite Kinship Index (KI_composite)
# ──────────────────────────────────────────────────

def compute_composite_kinship(
    ki_base: dict,
    ki_gamma: dict,
    ki_profile: dict,
    coupling_compat: dict,
    models: list[str],
    n_topics: int,
    n_conditions: int,
) -> dict:
    """
    KI_composite: Weighted combination of all kinship metrics.

    Weight matrix (adapts to available data):
    ┌─────────────────┬──────────┬──────────┬──────────┬──────────┐
    │                 │ W_BASE   │ W_GAMMA  │ W_PROFILE│ W_CC     │
    ├─────────────────┼──────────┼──────────┼──────────┼──────────┤
    │ 3+ T, 3+ C      │ 0.30     │ 0.20     │ 0.20     │ 0.30     │
    │ 3+ T, <3 C      │ 0.30     │ 0.30     │ 0.00     │ 0.40     │
    │ <3 T, 3+ C      │ 0.00     │ 0.30     │ 0.25     │ 0.45     │
    │ <3 T, <3 C      │ 0.00     │ 0.40     │ 0.00     │ 0.60     │
    └─────────────────┴──────────┴──────────┴──────────┴──────────┘

    KI_base requires 3+ topics (slope correlation).
    KI_profile requires 3+ conditions (profile correlation via Pearson).
    KI_gamma and CC always work.
    """
    has_base = n_topics >= 3
    has_profile = n_conditions >= 3

    if has_base and has_profile:
        W_BASE, W_GAMMA, W_PROFILE, W_COUPLING = 0.30, 0.20, 0.20, 0.30
    elif has_base and not has_profile:
        W_BASE, W_GAMMA, W_PROFILE, W_COUPLING = 0.30, 0.30, 0.00, 0.40
    elif not has_base and has_profile:
        W_BASE, W_GAMMA, W_PROFILE, W_COUPLING = 0.00, 0.30, 0.25, 0.45
    else:  # neither base nor profile
        W_BASE, W_GAMMA, W_PROFILE, W_COUPLING = 0.00, 0.40, 0.00, 0.60

    results = {}

    for m1, m2 in combinations(models, 2):
        pair_key = f"{m1}_vs_{m2}"

        # Normalize KI_base: r from [-1, 1] to [0, 1]
        base_r = ki_base.get(pair_key, {}).get("r", 0.0)
        base_norm = (base_r + 1.0) / 2.0

        # KI_gamma: cosine similarity already in [0, 1]
        gamma_sim = ki_gamma.get(pair_key, {}).get("cosine_similarity", 0.0)
        gamma_norm = max(0.0, gamma_sim)

        # KI_profile: r from [-1, 1] to [0, 1]
        profile_r = ki_profile.get(pair_key, {}).get("r", 0.0)
        profile_norm = (profile_r + 1.0) / 2.0

        # CC: average of both directions, normalized
        cc_forward = coupling_compat.get(f"{m1}_{m2}", {}).get("mean_coupling", 0.0)
        cc_reverse = coupling_compat.get(f"{m2}_{m1}", {}).get("mean_coupling", 0.0)
        cc_homo_1 = coupling_compat.get(f"{m1}_{m1}", {}).get("mean_coupling", 0.0)
        cc_homo_2 = coupling_compat.get(f"{m2}_{m2}", {}).get("mean_coupling", 0.0)

        cc_hetero = (cc_forward + cc_reverse) / 2.0 if (cc_forward or cc_reverse) else 0.0

        max_cc = max(abs(cc_homo_1), abs(cc_homo_2), abs(cc_hetero), 0.001)
        cc_norm = min(1.0, max(0.0, cc_hetero / max_cc)) if max_cc > 0 else 0.0

        ki_comp = round(
            W_BASE * base_norm +
            W_GAMMA * gamma_norm +
            W_PROFILE * profile_norm +
            W_COUPLING * cc_norm,
            4
        )

        results[pair_key] = {
            "ki_composite": ki_comp,
            "ki_base_raw": base_r,
            "ki_base_norm": round(base_norm, 4),
            "ki_base_weight": W_BASE,
            "ki_gamma_raw": gamma_sim,
            "ki_gamma_norm": round(gamma_norm, 4),
            "ki_profile_raw": profile_r,
            "ki_profile_norm": round(profile_norm, 4),
            "cc_hetero_raw": round(cc_hetero, 4),
            "cc_norm": round(cc_norm, 4),
            "model_1": m1,
            "model_2": m2,
        }

    return results


# ──────────────────────────────────────────────────
# Kinship Matrix Builder
# ──────────────────────────────────────────────────

def build_kinship_matrix(composite: dict, models: list[str]) -> list[list[float]]:
    """Build a symmetric kinship matrix from composite results."""
    n = len(models)
    matrix = [[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]

    for m1, m2 in combinations(models, 2):
        pair_key = f"{m1}_vs_{m2}"
        ki = composite.get(pair_key, {}).get("ki_composite", 0.0)
        i = models.index(m1)
        j = models.index(m2)
        matrix[i][j] = ki
        matrix[j][i] = ki

    return matrix


# ──────────────────────────────────────────────────
# Kinship Ranking
# ──────────────────────────────────────────────────

def compute_kinship_ranking(composite: dict) -> list[dict]:
    """Rank all model pairs by composite kinship index."""
    ranking = []
    for pair_key, data in composite.items():
        ranking.append({
            "pair": pair_key,
            "model_1": data["model_1"],
            "model_2": data["model_2"],
            "ki_composite": data["ki_composite"],
            "ki_base": data["ki_base_raw"],
            "ki_gamma": data["ki_gamma_raw"],
            "ki_profile": data["ki_profile_raw"],
            "cc_hetero": data["cc_hetero_raw"],
        })
    ranking.sort(key=lambda x: x["ki_composite"], reverse=True)
    return ranking


# ──────────────────────────────────────────────────
# CSV Export
# ──────────────────────────────────────────────────

def export_kinship_csvs(
    ki_base: dict, ki_gamma: dict, ki_profile: dict,
    coupling_compat: dict, composite: dict, ranking: list,
    matrix: list, models: list, output_dir: Path
):
    """Export all kinship results to CSVs."""

    # 1. Kinship Ranking (main output)
    if ranking:
        filepath = output_dir / "kinship_ranking.csv"
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=ranking[0].keys())
            writer.writeheader()
            writer.writerows(ranking)
        print(f"    Exported: {filepath.name}")

    # 2. Kinship Matrix
    filepath = output_dir / "kinship_matrix.csv"
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([""] + models)
        for i, model in enumerate(models):
            writer.writerow([model] + [round(v, 4) for v in matrix[i]])
    print(f"    Exported: {filepath.name}")

    # 3. Baseline Kinship Detail
    rows = []
    for pair_key, data in ki_base.items():
        rows.append({
            "pair": pair_key,
            "model_1": data["model_1"],
            "model_2": data["model_2"],
            "r": data["r"],
            "p": data["p"],
            "n_topics": data["n_topics"],
            "note": data.get("note", ""),
            "slopes_1": str(data["slopes_1"]),
            "slopes_2": str(data["slopes_2"]),
        })
    if rows:
        filepath = output_dir / "kinship_baseline_detail.csv"
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"    Exported: {filepath.name}")

    # 4. Gamma Vector Kinship Detail
    rows = []
    for pair_key, data in ki_gamma.items():
        rows.append({
            "pair": pair_key,
            "model_1": data["model_1"],
            "model_2": data["model_2"],
            "cosine_similarity": data["cosine_similarity"],
            "gamma_vector_1": str(data["gamma_vector_1"]),
            "gamma_vector_2": str(data["gamma_vector_2"]),
        })
    if rows:
        filepath = output_dir / "kinship_gamma_vectors.csv"
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"    Exported: {filepath.name}")

    # 5. Coupling Compatibility
    rows = []
    for pairing, data in coupling_compat.items():
        rows.append({
            "pairing": pairing,
            "mean_coupling": data["mean_coupling"],
            "sd_coupling": data["sd_coupling"],
            "ci95_lower": data["ci95"][0],
            "ci95_upper": data["ci95"][1],
            "n": data["n"],
            "pct_significant": data["pct_significant"],
        })
    if rows:
        filepath = output_dir / "kinship_coupling_compatibility.csv"
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"    Exported: {filepath.name}")

    # 6. Composite Detail
    rows = []
    for pair_key, data in composite.items():
        rows.append(data)
    if rows:
        filepath = output_dir / "kinship_composite_detail.csv"
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"    Exported: {filepath.name}")


# ──────────────────────────────────────────────────
# Prediction Tests
# ──────────────────────────────────────────────────

def test_predictions(ranking: list, composite: dict, coupling_compat: dict) -> list[dict]:
    """
    Test pre-registered predictions for the Kinship Test.

    Predictions:
      P1: KI(Claude, Gemini) > KI(Claude, GPT-4o) — Claude and Gemini are more kin
      P2: KI(Claude, Gemini) > KI(Gemini, GPT-4o) — Claude-Gemini closer than Gemini-GPT4o
      P3: CC(homo) > CC(hetero) — Homogeneous pairings couple more strongly
      P4: KI_gamma cosine similarity: Claude_vs_gemini > Claude_vs_gpt4o
      P5: Ranking: Claude-Gemini is #1 in kinship (most kin heterogeneous pair)
    """
    results = []

    # P1: KI(claude-gemini) > KI(claude-gpt4o)
    ki_cg = composite.get("claude_vs_gemini", {}).get("ki_composite", 0.0)
    ki_co = composite.get("claude_vs_gpt4o", {}).get("ki_composite", 0.0)
    results.append({
        "prediction": "P1",
        "description": "KI(Claude, Gemini) > KI(Claude, GPT-4o)",
        "value_1": ki_cg,
        "value_2": ki_co,
        "difference": round(ki_cg - ki_co, 4),
        "confirmed": ki_cg > ki_co,
    })

    # P2: KI(claude-gemini) > KI(gemini-gpt4o)
    ki_gg = composite.get("gemini_vs_gpt4o", {}).get("ki_composite", 0.0)
    results.append({
        "prediction": "P2",
        "description": "KI(Claude, Gemini) > KI(Gemini, GPT-4o)",
        "value_1": ki_cg,
        "value_2": ki_gg,
        "difference": round(ki_cg - ki_gg, 4),
        "confirmed": ki_cg > ki_gg,
    })

    # P3: CC(homo) > CC(hetero)
    homo_cc = []
    hetero_cc = []
    for pairing, data in coupling_compat.items():
        parts = pairing.split("_")
        if len(parts) == 2 and parts[0] == parts[1]:
            homo_cc.append(data["mean_coupling"])
        else:
            hetero_cc.append(data["mean_coupling"])
    homo_mean = _mean(homo_cc) if homo_cc else 0.0
    hetero_mean = _mean(hetero_cc) if hetero_cc else 0.0
    results.append({
        "prediction": "P3",
        "description": "CC(homogeneous) > CC(heterogeneous)",
        "value_1": homo_mean,
        "value_2": hetero_mean,
        "difference": round(homo_mean - hetero_mean, 4),
        "confirmed": homo_mean > hetero_mean,
    })

    # P4: KI_gamma: claude_vs_gemini > claude_vs_gpt4o
    kg_cg = composite.get("claude_vs_gemini", {}).get("ki_gamma_raw", 0.0)
    kg_co = composite.get("claude_vs_gpt4o", {}).get("ki_gamma_raw", 0.0)
    results.append({
        "prediction": "P4",
        "description": "KI_gamma(Claude, Gemini) > KI_gamma(Claude, GPT-4o)",
        "value_1": kg_cg,
        "value_2": kg_co,
        "difference": round(kg_cg - kg_co, 4),
        "confirmed": kg_cg > kg_co,
    })

    # P5: Claude-Gemini is rank #1 heterogeneous pair
    hetero_ranking = [r for r in ranking if r["model_1"] != r["model_2"]]
    is_first = False
    if hetero_ranking:
        is_first = (
            (hetero_ranking[0]["model_1"] == "claude" and hetero_ranking[0]["model_2"] == "gemini") or
            (hetero_ranking[0]["model_1"] == "gemini" and hetero_ranking[0]["model_2"] == "claude")
        )
    actual_first = hetero_ranking[0]["pair"] if hetero_ranking else "N/A"
    results.append({
        "prediction": "P5",
        "description": "Claude-Gemini is #1 heterogeneous kinship pair",
        "value_1": actual_first,
        "value_2": "claude_vs_gemini",
        "difference": 0.0,
        "confirmed": is_first,
    })

    return results


# ──────────────────────────────────────────────────
# Main Analysis Runner
# ──────────────────────────────────────────────────

def run_kinship_analysis(results_dir: Path, verbose: bool = False):
    """Run the full kinship analysis pipeline."""
    print(f"\n  {'='*50}")
    print(f"  Model Kinship Test -- Analysis Pipeline")
    print(f"  Results: {results_dir}/")
    print(f"  {'='*50}\n")

    # Create output directory
    analysis_dir = results_dir / "_kinship_analysis"
    analysis_dir.mkdir(exist_ok=True)

    # Load data
    dialogues = load_dialogues(results_dir)
    print(f"  Loaded {len(dialogues)} dialogues")

    if not dialogues:
        print("  X No data found. Exiting.")
        print("  Note: _raw.json files are skipped (no gamma vectors). Run batch-score first.")
        return

    # Detect models and conditions present
    models = sorted(set(
        d.get("model_a", "") for d in dialogues
    ) | set(
        d.get("model_b", "") for d in dialogues
    ))
    conditions = sorted(set(d.get("condition", "") for d in dialogues))
    topics = sorted(set(d.get("topic", "") for d in dialogues))
    pairings = sorted(set(d.get("pairing", "") for d in dialogues))

    n_topics = len(topics)
    n_conditions = len(conditions)

    print(f"  Models: {models}")
    print(f"  Conditions: {conditions}")
    print(f"  Topics: {topics}")
    print(f"  Pairings: {pairings}")

    if n_topics < 3:
        print(f"\n  Warning: Only {n_topics} topic(s) found. KI_base requires 3+ topics.")
        print(f"    KI_base will report raw slopes but no correlation.")
        print(f"    Composite index will use adjusted weights (KI_base weight = 0).")

    if n_conditions < 3:
        print(f"\n  Warning: Only {n_conditions} condition(s) found. KI_profile requires 3+ conditions.")
        print(f"    KI_profile will be 0 (Pearson needs >= 3 data points).")
        print(f"    Composite index will use adjusted weights (KI_profile weight = 0).")

    # ── Metric 1: Baseline Kinship ──
    print(f"\n  -- Metric 1: Baseline Kinship (KI_base) --")
    ki_base = compute_baseline_kinship(dialogues, models, topics)
    for pair_key, data in ki_base.items():
        if data.get("note"):
            print(f"    {pair_key}: {data['note']}")
            print(f"      slopes_1={data['slopes_1']}, slopes_2={data['slopes_2']}")
        else:
            print(f"    {pair_key}: r={data['r']}, p={data['p']}")

    # ── Metric 2: Gamma Vector Kinship ──
    print(f"\n  -- Metric 2: Gamma Vector Kinship (KI_gamma) --")
    ki_gamma = compute_gamma_vector_kinship(dialogues, models)
    for pair_key, data in ki_gamma.items():
        print(f"    {pair_key}: cosine_sim={data['cosine_similarity']}")
        if verbose:
            print(f"      {data['model_1']}: {data['gamma_vector_1']}")
            print(f"      {data['model_2']}: {data['gamma_vector_2']}")

    # ── Metric 3: Coupling Compatibility ──
    print(f"\n  -- Metric 3: Coupling Compatibility (CC) --")
    coupling_compat = compute_coupling_compatibility(dialogues)
    for pairing, data in coupling_compat.items():
        print(f"    {pairing}: coupling={data['mean_coupling']}+/-{data['sd_coupling']} (n={data['n']})")
    if not coupling_compat:
        print(f"    Warning: No Condition C data found. CC metrics will be 0.")

    # ── Metric 4: Response Profile Kinship ──
    print(f"\n  -- Metric 4: Response Profile Kinship (KI_profile) --")
    ki_profile = compute_response_profile_kinship(dialogues, models, conditions)
    for pair_key, data in ki_profile.items():
        print(f"    {pair_key}: r={data['r']}, p={data['p']}")

    # ── Metric 5: Composite Kinship Index ──
    print(f"\n  -- Metric 5: Composite Kinship Index --")
    composite = compute_composite_kinship(
        ki_base, ki_gamma, ki_profile, coupling_compat, models, n_topics, n_conditions
    )

    # ── Ranking ──
    ranking = compute_kinship_ranking(composite)
    print(f"\n  -- KINSHIP RANKING --")
    for i, entry in enumerate(ranking):
        print(f"    #{i+1}: {entry['pair']} -- KI={entry['ki_composite']}")

    # ── Kinship Matrix ──
    matrix = build_kinship_matrix(composite, models)
    print(f"\n  -- KINSHIP MATRIX --")
    print(f"    {'':>12}", end="")
    for m in models:
        print(f"  {m:>10}", end="")
    print()
    for i, m in enumerate(models):
        print(f"    {m:>12}", end="")
        for j in range(len(models)):
            print(f"  {matrix[i][j]:>10.4f}", end="")
        print()

    # ── Prediction Tests ──
    print(f"\n  -- PREDICTION TESTS --")
    predictions = test_predictions(ranking, composite, coupling_compat)
    confirmed = 0
    for pred in predictions:
        status = "CONFIRMED" if pred["confirmed"] else "NOT CONFIRMED"
        print(f"    {pred['prediction']}: {pred['description']}")
        print(f"      {status} (value_1={pred['value_1']}, value_2={pred['value_2']}, diff={pred['difference']})")
        if pred["confirmed"]:
            confirmed += 1

    # ── Export CSVs ──
    print(f"\n  -- Export --")
    export_kinship_csvs(
        ki_base, ki_gamma, ki_profile, coupling_compat,
        composite, ranking, matrix, models, analysis_dir
    )

    # Save full results as JSON
    full_results = {
        "ki_base": ki_base,
        "ki_gamma": ki_gamma,
        "ki_profile": ki_profile,
        "coupling_compatibility": coupling_compat,
        "composite": composite,
        "ranking": ranking,
        "matrix": matrix,
        "models": models,
        "predictions": predictions,
        "n_topics": n_topics,
        "n_conditions": n_conditions,
    }
    with open(analysis_dir / "kinship_full_results.json", "w") as f:
        json.dump(full_results, f, indent=2, default=str)
    print(f"    Exported: kinship_full_results.json")

    # ── Predictions CSV ──
    filepath = analysis_dir / "kinship_predictions.csv"
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=predictions[0].keys())
        writer.writeheader()
        writer.writerows(predictions)
    print(f"    Exported: kinship_predictions.csv")

    # ── Summary ──
    print(f"\n  {'='*50}")
    print(f"  Kinship Analysis complete. Results in {analysis_dir}/")
    print(f"  Topics used: {n_topics} ({', '.join(topics)})")
    print(f"  Conditions used: {n_conditions} ({', '.join(conditions)})")
    print(f"  Predictions confirmed: {confirmed}/{len(predictions)}")
    print(f"  {'='*50}\n")


# ──────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Model Kinship Test -- Analysis")
    parser.add_argument("--results-dir", type=str, default="results_kinship")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_kinship_analysis(Path(args.results_dir), verbose=args.verbose)
