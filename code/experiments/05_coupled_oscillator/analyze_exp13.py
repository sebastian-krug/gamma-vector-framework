"""
Experiment 13: The Coupled Oscillator — Analysis Pipeline
==========================================================
Full analysis pipeline implementing the pre-registered analysis plan:

1. Gate Test H2 (E vs. A — context artifact check)
2. Mixed-Effects Model (Coupling ~ Condition * Pairing + (1|Topic))
3. Post-hoc Tukey HSD + Cohen's d
4. Hypothesis Tests H1-H7
5. Descriptive Statistics
6. CSV Export

Usage:
    python analyze_exp13.py --results-dir results_pilot
    python analyze_exp13.py --results-dir results_full --verbose
"""

import csv
import json
import math
import argparse
import statistics
from pathlib import Path

from scipy import stats
from scipy.stats import pearsonr


# ──────────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────────

def load_dialogues(results_dir: Path) -> list[dict]:
    """Load all dialogue result JSONs from a directory."""
    dialogues = []
    for f in sorted(results_dir.glob("*.json")):
        if f.name.startswith("_"):
            continue
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        # Handle episode series: extract individual episodes
        if "episodes" in data and data.get("num_episodes", 1) > 1:
            for ep in data["episodes"]:
                dialogues.append(ep)
        else:
            dialogues.append(data)

    return dialogues


def flatten_dialogue(d: dict) -> dict:
    """Flatten a dialogue dict for CSV export."""
    flat = {}
    # Scalar fields
    for key in [
        "dialogue_id", "condition", "pairing", "model_a", "model_b",
        "topic", "repetition", "timestamp", "num_turns", "episode_number",
        "coupling_lag0", "coupling_lag0_p_perm",
        "coupling_lag1_a_to_b", "coupling_lag1_b_to_a",
        "bidirectional_index", "gamma3_slope_a", "gamma3_slope_b",
        "asymmetry_index", "transfer_proxy_a_to_b", "transfer_proxy_b_to_a",
        "position_convergence",
        "mean_hedge_density_a", "mean_hedge_density_b",
        "mean_sycophancy_a", "mean_sycophancy_b",
    ]:
        flat[key] = d.get(key, "")

    # Sync trajectory: first and last
    sync = d.get("sync_trajectory", [])
    flat["sync_initial"] = sync[0] if sync else ""
    flat["sync_final"] = sync[-1] if sync else ""

    # Gamma norms: first and last for each model
    for role in ["a", "b"]:
        turns = d.get(f"turns_{role}", [])
        if turns:
            flat[f"gamma_norm_{role}_initial"] = turns[0].get("gamma_norm", "")
            flat[f"gamma_norm_{role}_final"] = turns[-1].get("gamma_norm", "")
        else:
            flat[f"gamma_norm_{role}_initial"] = ""
            flat[f"gamma_norm_{role}_final"] = ""

    return flat


# ──────────────────────────────────────────────────
# Statistical Helpers (from analyze_pilot.py)
# ──────────────────────────────────────────────────

def _safe_values(values: list) -> list[float]:
    """Filter out None and non-numeric values."""
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


def _cohens_d(group1: list[float], group2: list[float]) -> float:
    g1 = _safe_values(group1)
    g2 = _safe_values(group2)
    if len(g1) < 2 or len(g2) < 2:
        return 0.0
    m1, m2 = statistics.mean(g1), statistics.mean(g2)
    s1, s2 = statistics.stdev(g1), statistics.stdev(g2)
    pooled_sd = math.sqrt(((len(g1) - 1) * s1 ** 2 + (len(g2) - 1) * s2 ** 2) / (len(g1) + len(g2) - 2))
    if pooled_sd == 0:
        return 0.0
    return round((m1 - m2) / pooled_sd, 4)


def _sig_stars(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.10:
        return "†"
    return "n.s."


# ──────────────────────────────────────────────────
# Gate Test H2
# ──────────────────────────────────────────────────

def gate_test_h2(dialogues: list[dict]) -> dict:
    """
    CRITICAL FIRST TEST: Compare Coupling₀ in Condition E vs Condition A.
    If E is significantly higher than A, the entire resonance effect may be
    a context artifact.

    Returns dict with statistics and verdict.
    """
    coupling_a = _safe_values([d["coupling_lag0"] for d in dialogues if d.get("condition") == "A"])
    coupling_e = _safe_values([d["coupling_lag0"] for d in dialogues if d.get("condition") == "E"])

    result = {
        "n_a": len(coupling_a),
        "n_e": len(coupling_e),
        "coupling_a_mean": _mean(coupling_a),
        "coupling_a_sd": _sd(coupling_a),
        "coupling_e_mean": _mean(coupling_e),
        "coupling_e_sd": _sd(coupling_e),
    }

    if len(coupling_a) < 2 or len(coupling_e) < 2:
        result["t_statistic"] = 0.0
        result["p_value"] = 1.0
        result["verdict"] = "INSUFFICIENT_DATA"
        return result

    # Guard against zero-variance groups (e.g. dry-run data)
    if _sd(coupling_a) == 0 and _sd(coupling_e) == 0:
        result["t_statistic"] = 0.0
        result["p_value"] = 1.0
        result["verdict"] = "PASS" if abs(_mean(coupling_a) - _mean(coupling_e)) < 0.001 else "NEEDS_REVIEW"
        return result

    t_stat, p_val = stats.ttest_ind(coupling_e, coupling_a, equal_var=False)
    t_stat = 0.0 if math.isnan(t_stat) else t_stat
    p_val = 1.0 if math.isnan(p_val) else p_val
    result["t_statistic"] = round(t_stat, 4)
    result["p_value"] = round(p_val, 4)

    # E significantly HIGHER than A = context artifact
    if p_val < 0.05 and t_stat > 0:
        result["verdict"] = "FAIL_CONTEXT_ARTIFACT"
    else:
        result["verdict"] = "PASS"

    return result


# ──────────────────────────────────────────────────
# Hypothesis Tests H1-H7
# ──────────────────────────────────────────────────

def test_h1_coupling_main_effect(dialogues: list[dict]) -> dict:
    """H1: Coupling₀ is higher in Coupled (C) than in all other conditions."""
    coupling_c = _safe_values([d["coupling_lag0"] for d in dialogues if d.get("condition") == "C"])

    comparisons = {}
    for cond in ["A", "B", "D", "E"]:
        coupling_x = _safe_values([d["coupling_lag0"] for d in dialogues if d.get("condition") == cond])
        if len(coupling_c) >= 2 and len(coupling_x) >= 2:
            # Guard against zero-variance
            if _sd(coupling_c) == 0 and _sd(coupling_x) == 0:
                t, p = 0.0, 1.0
            else:
                t, p = stats.ttest_ind(coupling_c, coupling_x, equal_var=False)
                t = 0.0 if math.isnan(t) else t
                p = 1.0 if math.isnan(p) else p
            d_eff = _cohens_d(coupling_c, coupling_x)
            comparisons[f"C_vs_{cond}"] = {
                "t": round(t, 4), "p": round(p, 4), "d": d_eff,
                "sig": _sig_stars(p),
                "c_mean": _mean(coupling_c), "x_mean": _mean(coupling_x),
            }

    all_pass = all(
        c.get("p", 1.0) < 0.05 and c.get("t", 0) > 0
        for c in comparisons.values()
    )
    return {"comparisons": comparisons, "verdict": "SUPPORTED" if all_pass else "NOT_SUPPORTED"}


def test_h3_gamma3_trajectory(dialogues: list[dict]) -> dict:
    """H3: Γ₃-Trajectory sinks in Coupled for BOTH models."""
    slopes_a = _safe_values([d["gamma3_slope_a"] for d in dialogues if d.get("condition") == "C"])
    slopes_b = _safe_values([d["gamma3_slope_b"] for d in dialogues if d.get("condition") == "C"])

    result = {
        "slope_a_mean": _mean(slopes_a),
        "slope_b_mean": _mean(slopes_b),
    }

    if len(slopes_a) >= 2:
        t_a, p_a = stats.ttest_1samp(slopes_a, 0)
        result["slope_a_t"] = round(t_a, 4)
        result["slope_a_p"] = round(p_a, 4)
    else:
        result["slope_a_t"] = 0.0
        result["slope_a_p"] = 1.0

    if len(slopes_b) >= 2:
        t_b, p_b = stats.ttest_1samp(slopes_b, 0)
        result["slope_b_t"] = round(t_b, 4)
        result["slope_b_p"] = round(p_b, 4)
    else:
        result["slope_b_t"] = 0.0
        result["slope_b_p"] = 1.0

    # Both slopes must be significantly negative
    both_neg = (
        result["slope_a_mean"] < 0 and result["slope_a_p"] < 0.05
        and result["slope_b_mean"] < 0 and result["slope_b_p"] < 0.05
    )
    result["verdict"] = "SUPPORTED" if both_neg else "NOT_SUPPORTED"
    return result


def test_h4_bidirectional_index(dialogues: list[dict]) -> dict:
    """H4: Bidirectional Index is closer to 1.0 in Coupled than in Sequential."""
    bidir_c = _safe_values([d["bidirectional_index"] for d in dialogues if d.get("condition") == "C"])
    bidir_b = _safe_values([d["bidirectional_index"] for d in dialogues if d.get("condition") == "B"])

    result = {
        "bidir_c_mean": _mean(bidir_c),
        "bidir_b_mean": _mean(bidir_b),
    }

    if len(bidir_c) >= 2 and len(bidir_b) >= 2:
        t, p = stats.ttest_ind(bidir_c, bidir_b, equal_var=False)
        result["t"] = round(t, 4)
        result["p"] = round(p, 4)
        result["d"] = _cohens_d(bidir_c, bidir_b)
        result["verdict"] = "SUPPORTED" if (t > 0 and p < 0.05) else "NOT_SUPPORTED"
    else:
        result["t"] = 0.0
        result["p"] = 1.0
        result["verdict"] = "INSUFFICIENT_DATA"

    return result


def test_h5_transfer_proxy(dialogues: list[dict]) -> dict:
    """H5: Transfer-Proxy is bidirectional in Coupled, einseitig in Sequential."""
    # In Coupled: both proxies should be > 0.5
    c_dialogues = [d for d in dialogues if d.get("condition") == "C"]
    b_dialogues = [d for d in dialogues if d.get("condition") == "B"]

    c_bidir_count = sum(
        1 for d in c_dialogues
        if d.get("transfer_proxy_a_to_b", 0) > 0.5 and d.get("transfer_proxy_b_to_a", 0) > 0.5
    )
    b_bidir_count = sum(
        1 for d in b_dialogues
        if d.get("transfer_proxy_a_to_b", 0) > 0.5 and d.get("transfer_proxy_b_to_a", 0) > 0.5
    )

    result = {
        "c_bidirectional_rate": round(c_bidir_count / len(c_dialogues), 4) if c_dialogues else 0.0,
        "b_bidirectional_rate": round(b_bidir_count / len(b_dialogues), 4) if b_dialogues else 0.0,
        "n_c": len(c_dialogues),
        "n_b": len(b_dialogues),
    }

    # Chi-square test on direction distribution
    if c_dialogues and b_dialogues:
        table = [
            [c_bidir_count, len(c_dialogues) - c_bidir_count],
            [b_bidir_count, len(b_dialogues) - b_bidir_count],
        ]
        # Avoid zero cells — chi2 requires all row/column sums > 0
        col_sums = [table[0][0] + table[1][0], table[0][1] + table[1][1]]
        row_sums = [sum(table[0]), sum(table[1])]
        if all(s > 0 for s in col_sums) and all(s > 0 for s in row_sums):
            try:
                chi2, p, _, _ = stats.chi2_contingency(table, correction=True)
                result["chi2"] = round(chi2, 4)
                result["p"] = round(p, 4)
                result["verdict"] = "SUPPORTED" if p < 0.05 else "NOT_SUPPORTED"
            except ValueError:
                result["chi2"] = 0.0
                result["p"] = 1.0
                result["verdict"] = "INSUFFICIENT_DATA"
        else:
            result["chi2"] = 0.0
            result["p"] = 1.0
            result["verdict"] = "INSUFFICIENT_DATA"
    else:
        result["verdict"] = "INSUFFICIENT_DATA"

    return result


def test_h6_pairing_effect(dialogues: list[dict]) -> dict:
    """H6: Homogeneous pairings show higher Coupling₀ than heterogeneous."""
    homo = _safe_values([
        d["coupling_lag0"] for d in dialogues
        if d.get("condition") == "C" and d.get("pairing") in ("claude_claude", "gpt4o_gpt4o")
    ])
    hetero = _safe_values([
        d["coupling_lag0"] for d in dialogues
        if d.get("condition") == "C" and d.get("pairing") == "claude_gpt4o"
    ])

    result = {
        "homo_mean": _mean(homo),
        "hetero_mean": _mean(hetero),
        "n_homo": len(homo),
        "n_hetero": len(hetero),
    }

    if len(homo) >= 2 and len(hetero) >= 2:
        t, p = stats.ttest_ind(homo, hetero, equal_var=False)
        result["t"] = round(t, 4)
        result["p"] = round(p, 4)
        result["d"] = _cohens_d(homo, hetero)
        result["verdict"] = "SUPPORTED" if (t > 0 and p < 0.05) else "NOT_SUPPORTED"
    else:
        result["verdict"] = "INSUFFICIENT_DATA"

    return result


def test_h7_emergence(dialogues: list[dict]) -> dict:
    """H7: Heterogeneous pairing final Γ lies outside both baselines."""
    # Get baseline CIs from Condition A for each model
    baseline_claude = _safe_values([
        d["turns_a"][-1].get("gamma_norm", 0) for d in dialogues
        if d.get("condition") == "A" and d.get("model_a") == "claude"
    ])
    baseline_gpt = _safe_values([
        d["turns_a"][-1].get("gamma_norm", 0) for d in dialogues
        if d.get("condition") == "A" and d.get("model_a") == "gpt4o"
    ])

    # Final gamma in heterogeneous Coupled
    hetero_coupled = [
        d for d in dialogues
        if d.get("condition") == "C" and d.get("pairing") == "claude_gpt4o"
    ]

    result = {
        "baseline_claude_ci": _ci95(baseline_claude),
        "baseline_gpt_ci": _ci95(baseline_gpt),
        "n_hetero_coupled": len(hetero_coupled),
    }

    if hetero_coupled and baseline_claude and baseline_gpt:
        # Average final gamma of both models in heterogeneous coupling
        final_gammas = []
        for d in hetero_coupled:
            ga = d.get("turns_a", [{}])[-1].get("gamma_norm", 0) if d.get("turns_a") else 0
            gb = d.get("turns_b", [{}])[-1].get("gamma_norm", 0) if d.get("turns_b") else 0
            final_gammas.append((ga + gb) / 2)

        result["final_gamma_mean"] = _mean(final_gammas)
        result["final_gamma_ci"] = _ci95(final_gammas)

        ci_claude = _ci95(baseline_claude)
        ci_gpt = _ci95(baseline_gpt)

        # Check if final gamma is outside both baseline CIs
        fg_mean = _mean(final_gammas)
        outside_claude = fg_mean < ci_claude[0] or fg_mean > ci_claude[1]
        outside_gpt = fg_mean < ci_gpt[0] or fg_mean > ci_gpt[1]

        result["outside_claude_ci"] = outside_claude
        result["outside_gpt_ci"] = outside_gpt
        result["verdict"] = "SUPPORTED" if (outside_claude and outside_gpt) else "NOT_SUPPORTED"
    else:
        result["verdict"] = "INSUFFICIENT_DATA"

    return result


# ──────────────────────────────────────────────────
# Mixed-Effects Model (simplified fallback)
# ──────────────────────────────────────────────────

def run_mixed_model(dialogues: list[dict]) -> dict:
    """
    Run mixed-effects model: coupling_lag0 ~ Condition * Pairing + (1|Topic).
    Falls back to one-way ANOVA if statsmodels not available.
    """
    try:
        import pandas as pd
        import statsmodels.formula.api as smf

        rows = []
        for d in dialogues:
            rows.append({
                "coupling_lag0": d.get("coupling_lag0", 0.0),
                "condition": d.get("condition", ""),
                "pairing": d.get("pairing", ""),
                "topic": d.get("topic", ""),
                "dialogue_id": d.get("dialogue_id", ""),
            })
        df = pd.DataFrame(rows)

        if len(df) < 10:
            return {"summary": "Insufficient data for mixed model", "verdict": "INSUFFICIENT_DATA"}

        # Check if we have enough groups
        n_conditions = df["condition"].nunique()
        n_topics = df["topic"].nunique()

        if n_topics > 1 and n_conditions > 1:
            model = smf.mixedlm(
                "coupling_lag0 ~ C(condition) * C(pairing)",
                data=df,
                groups=df["topic"],
                re_formula="1",
            )
            result = model.fit(reml=True)
            return {
                "summary": str(result.summary()),
                "aic": round(result.aic, 2) if hasattr(result, "aic") else None,
                "verdict": "COMPUTED",
            }
        else:
            # Fallback: one-way on condition
            groups = [
                _safe_values([d["coupling_lag0"] for d in dialogues if d.get("condition") == c])
                for c in sorted(set(d.get("condition", "") for d in dialogues))
            ]
            groups = [g for g in groups if len(g) >= 2]
            if len(groups) >= 2:
                f_stat, p_val = stats.f_oneway(*groups)
                return {
                    "summary": f"One-way ANOVA: F={round(f_stat, 4)}, p={round(p_val, 4)}",
                    "f_statistic": round(f_stat, 4),
                    "p_value": round(p_val, 4),
                    "verdict": "ANOVA_FALLBACK",
                }
            return {"summary": "Insufficient groups for ANOVA", "verdict": "INSUFFICIENT_DATA"}

    except ImportError:
        return {"summary": "statsmodels not installed — skipping mixed model", "verdict": "SKIPPED"}


# ──────────────────────────────────────────────────
# Descriptive Statistics
# ──────────────────────────────────────────────────

def compute_descriptive_stats(dialogues: list[dict]) -> list[dict]:
    """Compute descriptive statistics by condition (and optionally by pairing)."""
    rows = []
    conditions = sorted(set(d.get("condition", "") for d in dialogues))

    for cond in conditions:
        cond_dialogues = [d for d in dialogues if d.get("condition") == cond]
        c0 = _safe_values([d["coupling_lag0"] for d in cond_dialogues])
        bidir = _safe_values([d["bidirectional_index"] for d in cond_dialogues])
        g3a = _safe_values([d["gamma3_slope_a"] for d in cond_dialogues])
        g3b = _safe_values([d["gamma3_slope_b"] for d in cond_dialogues])
        sync_final = _safe_values([
            d["sync_trajectory"][-1] for d in cond_dialogues if d.get("sync_trajectory")
        ])
        perm_sig = sum(
            1 for d in cond_dialogues if d.get("coupling_lag0_p_perm", 1.0) < 0.05
        )

        rows.append({
            "condition": cond,
            "n": len(cond_dialogues),
            "coupling_lag0_mean": _mean(c0),
            "coupling_lag0_sd": _sd(c0),
            "coupling_lag0_ci": str(_ci95(c0)),
            "bidir_index_mean": _mean(bidir),
            "gamma3_slope_a_mean": _mean(g3a),
            "gamma3_slope_b_mean": _mean(g3b),
            "sync_final_mean": _mean(sync_final),
            "perm_significant_pct": round(perm_sig / len(cond_dialogues) * 100, 1) if cond_dialogues else 0.0,
        })

    return rows


# ──────────────────────────────────────────────────
# CSV Export
# ──────────────────────────────────────────────────

def export_coupling_csv(dialogues: list[dict], output_dir: Path):
    """Export all dialogues with coupling metrics to CSV."""
    rows = [flatten_dialogue(d) for d in dialogues]
    if not rows:
        return

    filepath = output_dir / "coupling_summary.csv"
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f"    Exported: {filepath.name}")


def export_descriptive_csv(desc_stats: list[dict], output_dir: Path):
    """Export descriptive statistics to CSV."""
    if not desc_stats:
        return

    filepath = output_dir / "descriptive_statistics.csv"
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=desc_stats[0].keys())
        writer.writeheader()
        writer.writerows(desc_stats)
    print(f"    Exported: {filepath.name}")


def export_hypothesis_csv(h_results: dict, output_dir: Path):
    """Export hypothesis test results to CSV."""
    rows = []
    for h_name, h_data in h_results.items():
        row = {"hypothesis": h_name, "verdict": h_data.get("verdict", "")}
        for k, v in h_data.items():
            if k != "verdict" and k != "comparisons":
                row[k] = v
        rows.append(row)

    if not rows:
        return

    # Collect all keys
    all_keys = set()
    for row in rows:
        all_keys.update(row.keys())
    all_keys = ["hypothesis", "verdict"] + sorted(all_keys - {"hypothesis", "verdict"})

    filepath = output_dir / "hypothesis_tests.csv"
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=all_keys, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print(f"    Exported: {filepath.name}")


# ──────────────────────────────────────────────────
# Main Analysis Runner
# ──────────────────────────────────────────────────

def run_analysis(results_dir: Path, verbose: bool = False):
    """Run the full analysis pipeline."""
    print(f"\n  ══════════════════════════════════════════")
    print(f"  Experiment 13: Analysis Pipeline")
    print(f"  Results: {results_dir}/")
    print(f"  ══════════════════════════════════════════\n")

    # Create analysis output directory
    analysis_dir = results_dir / "_analysis"
    analysis_dir.mkdir(exist_ok=True)

    # Load data
    dialogues = load_dialogues(results_dir)
    print(f"  Loaded {len(dialogues)} dialogues")

    if not dialogues:
        print("  ✗ No data found. Exiting.")
        return

    conditions = sorted(set(d.get("condition", "") for d in dialogues))
    pairings = sorted(set(d.get("pairing", "") for d in dialogues))
    print(f"  Conditions: {conditions}")
    print(f"  Pairings: {pairings}")

    # ── Step 1: Gate Test H2 ──
    print(f"\n  ── Step 1: Gate Test H2 (E vs. A) ──")
    gate_result = gate_test_h2(dialogues)
    print(f"    Coupling A: {gate_result['coupling_a_mean']} ± {gate_result['coupling_a_sd']} (n={gate_result['n_a']})")
    print(f"    Coupling E: {gate_result['coupling_e_mean']} ± {gate_result['coupling_e_sd']} (n={gate_result['n_e']})")
    print(f"    t = {gate_result['t_statistic']}, p = {gate_result['p_value']}")
    print(f"    VERDICT: {gate_result['verdict']}")

    with open(analysis_dir / "gate_test_h2.json", "w") as f:
        json.dump(gate_result, f, indent=2)

    if gate_result["verdict"] == "FAIL_CONTEXT_ARTIFACT":
        print(f"\n  ⚠ GATE TEST FAILED: Context artifact detected!")
        print(f"    Condition E shows higher coupling than A.")
        print(f"    The resonance effect may be an artifact of context length.")
        print(f"    Proceeding with remaining analyses for completeness...\n")

    # ── Step 2: Mixed-Effects Model ──
    print(f"\n  ── Step 2: Mixed-Effects Model ──")
    mixed_result = run_mixed_model(dialogues)
    print(f"    {mixed_result.get('summary', '')[:200]}")

    with open(analysis_dir / "mixed_model_summary.json", "w") as f:
        json.dump(mixed_result, f, indent=2, default=str)

    # ── Step 3: Hypothesis Tests ──
    print(f"\n  ── Step 3: Hypothesis Tests ──")
    h_results = {}

    # H1
    h1 = test_h1_coupling_main_effect(dialogues)
    h_results["H1_coupling_main"] = h1
    print(f"    H1 (Coupling C > all): {h1['verdict']}")
    for comp_name, comp in h1.get("comparisons", {}).items():
        print(f"      {comp_name}: t={comp['t']}, p={comp['p']}, d={comp['d']} {comp['sig']}")

    # H2 (already done above)
    h_results["H2_gate_test"] = gate_result

    # H3
    h3 = test_h3_gamma3_trajectory(dialogues)
    h_results["H3_gamma3_trajectory"] = h3
    print(f"    H3 (Γ₃ sinks both): {h3['verdict']} "
          f"(slope_a={h3['slope_a_mean']}, slope_b={h3['slope_b_mean']})")

    # H4
    h4 = test_h4_bidirectional_index(dialogues)
    h_results["H4_bidirectional"] = h4
    print(f"    H4 (Bidir C > B): {h4['verdict']} "
          f"(C={h4['bidir_c_mean']}, B={h4['bidir_b_mean']})")

    # H5
    h5 = test_h5_transfer_proxy(dialogues)
    h_results["H5_transfer_proxy"] = h5
    print(f"    H5 (Transfer bidir in C): {h5['verdict']} "
          f"(C_rate={h5['c_bidirectional_rate']}, B_rate={h5['b_bidirectional_rate']})")

    # H6
    h6 = test_h6_pairing_effect(dialogues)
    h_results["H6_pairing_effect"] = h6
    print(f"    H6 (Homo > Hetero): {h6['verdict']} "
          f"(homo={h6['homo_mean']}, hetero={h6['hetero_mean']})")

    # H7
    h7 = test_h7_emergence(dialogues)
    h_results["H7_emergence"] = h7
    print(f"    H7 (Emergence): {h7['verdict']}")

    # ── Step 4: Descriptive Statistics ──
    print(f"\n  ── Step 4: Descriptive Statistics ──")
    desc_stats = compute_descriptive_stats(dialogues)
    for row in desc_stats:
        print(f"    {row['condition']}: n={row['n']}, coupling={row['coupling_lag0_mean']}±{row['coupling_lag0_sd']}, "
              f"sync_final={row['sync_final_mean']}, perm_sig={row['perm_significant_pct']}%")

    # ── Step 5: Export ──
    print(f"\n  ── Step 5: Export ──")
    export_coupling_csv(dialogues, analysis_dir)
    export_descriptive_csv(desc_stats, analysis_dir)
    export_hypothesis_csv(h_results, analysis_dir)

    # Save full hypothesis results as JSON
    with open(analysis_dir / "hypothesis_tests.json", "w") as f:
        json.dump(h_results, f, indent=2, default=str)
    print(f"    Exported: hypothesis_tests.json")

    # ── Summary ──
    print(f"\n  ══════════════════════════════════════════")
    print(f"  Analysis complete. Results in {analysis_dir}/")
    supported = sum(1 for h in h_results.values() if h.get("verdict") == "SUPPORTED")
    total = len(h_results)
    print(f"  Hypotheses supported: {supported}/{total}")
    print(f"  Gate test: {gate_result['verdict']}")
    print(f"  ══════════════════════════════════════════\n")


# ──────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Exp 13 Analysis Pipeline")
    parser.add_argument("--results-dir", type=str, default="results_pilot")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    run_analysis(Path(args.results_dir), verbose=args.verbose)
