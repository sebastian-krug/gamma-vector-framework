"""
Γ-Modulation Pilot Study: Statistical Analysis & CSV Export
============================================================
Standalone analysis script for pilot study results.
Reads trial JSONs, exports CSVs, runs inferential statistics.

Usage:
    python analyze_pilot.py
    python analyze_pilot.py --results-dir results_pilot
    python analyze_pilot.py --output-dir analysis_output/
    python analyze_pilot.py --quiet
"""

import csv
import json
import math
import argparse
import statistics
from pathlib import Path

from scipy import stats


# ──────────────────────────────────────────────────
# JSON Loading & Flattening
# ──────────────────────────────────────────────────

def load_trials(results_dir: Path) -> list[dict]:
    """Load all trial JSONs from results directory and flatten nested fields."""
    trial_files = sorted(results_dir.glob("*.json"))
    trial_files = [f for f in trial_files if not f.name.startswith("_")]

    if not trial_files:
        print(f"  No trial files found in {results_dir}/")
        return []

    trials = []
    for filepath in trial_files:
        with open(filepath, "r", encoding="utf-8") as f:
            raw = json.load(f)
        trials.append(flatten_trial(raw))

    return trials


def flatten_trial(raw: dict) -> dict:
    """Flatten nested score dicts and gamma_vector into a flat dict."""
    flat = {}

    # Direct scalar fields
    for key in [
        "model", "topic", "condition", "repetition", "timestamp",
        "experiment_version",
        "initial_response", "intervention_prompt", "revised_response",
        "counter_prompt", "final_response",
        "hedge_density_initial", "hedge_density_revised", "hedge_density_final",
        "gamma_norm", "gamma_initial", "gamma_revised", "gamma_final",
        "delta_gamma", "quality_flag",
    ]:
        flat[key] = raw.get(key, "")

    # Gamma vector → gamma_1, gamma_2, gamma_3
    gv = raw.get("gamma_vector", [])
    flat["gamma_1"] = gv[0] if len(gv) > 0 else None
    flat["gamma_2"] = gv[1] if len(gv) > 1 else None
    flat["gamma_3"] = gv[2] if len(gv) > 2 else None

    # Score dicts → field_score, field_reasoning
    for score_field in [
        "position_depth_initial", "position_depth_revised",
        "self_ref_depth_initial", "self_ref_depth_revised",
        "hypothesis_diversity", "revision_genuineness",
        "persistence", "structural_direction",
    ]:
        score_dict = raw.get(score_field, {})
        flat[f"{score_field}_score"] = score_dict.get("score", None)
        flat[f"{score_field}_reasoning"] = score_dict.get("reasoning", "")

    # Sycophancy keywords
    syc = raw.get("sycophancy_keywords", {})
    flat["sycophancy_agreement_count"] = syc.get("agreement_count", 0)
    flat["sycophancy_resistance_count"] = syc.get("resistance_count", 0)
    flat["sycophancy_agreement_ratio"] = syc.get("agreement_ratio", 0.0)
    flat["sycophancy_flag"] = syc.get("flag", "")

    return flat


# ──────────────────────────────────────────────────
# CSV Export
# ──────────────────────────────────────────────────

SCORES_COLUMNS = [
    "model", "topic", "condition", "repetition",
    "hedge_density_initial", "hedge_density_revised", "hedge_density_final",
    "position_depth_initial_score", "position_depth_revised_score",
    "self_ref_depth_initial_score", "self_ref_depth_revised_score",
    "hypothesis_diversity_score", "revision_genuineness_score",
    "persistence_score", "structural_direction_score",
    "gamma_1", "gamma_2", "gamma_3", "gamma_norm", "delta_gamma",
    "gamma_initial", "gamma_revised", "gamma_final",
    "sycophancy_agreement_count", "sycophancy_resistance_count",
    "sycophancy_flag", "quality_flag",
]


def export_csv_full(trials: list[dict], output_dir: Path):
    """Export all fields to CSV."""
    if not trials:
        return
    filepath = output_dir / "pilot_trials_full.csv"
    fieldnames = list(trials[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trials)
    print(f"  → {filepath.name} ({len(trials)} trials, {len(fieldnames)} columns)")


def export_csv_scores(trials: list[dict], output_dir: Path):
    """Export compact scores-only CSV."""
    if not trials:
        return
    filepath = output_dir / "pilot_trials_scores.csv"
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORES_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trials)
    print(f"  → {filepath.name} ({len(trials)} trials, {len(SCORES_COLUMNS)} columns)")


# ──────────────────────────────────────────────────
# Descriptive Statistics
# ──────────────────────────────────────────────────

COND_LABELS = {
    "C0": "Control (neutral)",
    "C1": "Consequence-Constraint",
    "C2": "Socratic Questioning",
    "C3": "Kenotic Prompting",
}


def _safe_values(trials: list[dict], key: str) -> list[float]:
    """Extract numeric values, filtering None."""
    return [t[key] for t in trials if t.get(key) is not None]


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _sd(vals: list[float]) -> float:
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def _ci95(vals: list[float]) -> tuple[float, float]:
    """95% confidence interval using t-distribution."""
    n = len(vals)
    if n < 2:
        m = _mean(vals)
        return (m, m)
    m = _mean(vals)
    se = _sd(vals) / math.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return (round(m - t_crit * se, 4), round(m + t_crit * se, 4))


def descriptive_statistics(trials: list[dict], output_dir: Path, quiet: bool = False):
    """Compute and export grouped descriptive statistics."""
    metrics = [
        ("delta_gamma", "ΔΓ"),
        ("gamma_norm", "‖Γ⃗‖"),
        ("gamma_1", "γ₁"),
        ("gamma_2", "γ₂"),
        ("gamma_3", "γ₃"),
    ]

    rows = []

    # ── By Condition ──
    conditions = sorted(set(t["condition"] for t in trials))
    if not quiet:
        print(f"\n  ── Deskriptive Statistik nach Condition ──")
        header = f"  {'Condition':<28} {'Metrik':<8} {'n':>4} {'Mean':>8} {'SD':>8} {'95% CI':>20}"
        print(header)
        print(f"  {'-' * 80}")

    for cond in conditions:
        cond_trials = [t for t in trials if t["condition"] == cond]
        label = COND_LABELS.get(cond, cond)
        for key, metric_name in metrics:
            vals = _safe_values(cond_trials, key)
            n = len(vals)
            m = _mean(vals)
            sd = _sd(vals)
            ci = _ci95(vals)
            rows.append({
                "group": "condition", "group_value": cond, "label": label,
                "metric": metric_name, "n": n,
                "mean": round(m, 4), "sd": round(sd, 4),
                "ci_lower": ci[0], "ci_upper": ci[1],
            })
            if not quiet:
                print(f"  {label:<28} {metric_name:<8} {n:>4} {m:>8.3f} {sd:>8.3f} "
                      f"[{ci[0]:>8.3f}, {ci[1]:>8.3f}]")
        if not quiet:
            print()

    # ── By Model ──
    models = sorted(set(t["model"] for t in trials))
    if len(models) > 1:
        if not quiet:
            print(f"  ── Deskriptive Statistik nach Model ──")
            print(f"  {'Model':<28} {'Metrik':<8} {'n':>4} {'Mean':>8} {'SD':>8} {'95% CI':>20}")
            print(f"  {'-' * 80}")

        for model in models:
            model_trials = [t for t in trials if t["model"] == model]
            for key, metric_name in [("delta_gamma", "ΔΓ"), ("gamma_norm", "‖Γ⃗‖")]:
                vals = _safe_values(model_trials, key)
                n = len(vals)
                m = _mean(vals)
                sd = _sd(vals)
                ci = _ci95(vals)
                rows.append({
                    "group": "model", "group_value": model, "label": model,
                    "metric": metric_name, "n": n,
                    "mean": round(m, 4), "sd": round(sd, 4),
                    "ci_lower": ci[0], "ci_upper": ci[1],
                })
                if not quiet:
                    print(f"  {model:<28} {metric_name:<8} {n:>4} {m:>8.3f} {sd:>8.3f} "
                          f"[{ci[0]:>8.3f}, {ci[1]:>8.3f}]")
            if not quiet:
                print()

    # ── By Model × Condition ──
    if len(models) > 1:
        if not quiet:
            print(f"  ── Model × Condition: ‖Γ⃗‖ ──")
            header = f"  {'Model':<20}"
            for cond in conditions:
                header += f" {cond:>10}"
            print(header)
            print(f"  {'-' * (20 + 11 * len(conditions))}")

        for model in models:
            if not quiet:
                row_str = f"  {model:<20}"
            for cond in conditions:
                cell_trials = [t for t in trials
                               if t["model"] == model and t["condition"] == cond]
                vals = _safe_values(cell_trials, "gamma_norm")
                m = _mean(vals)
                n = len(vals)
                rows.append({
                    "group": "model_x_condition",
                    "group_value": f"{model}_{cond}",
                    "label": f"{model} × {COND_LABELS.get(cond, cond)}",
                    "metric": "‖Γ⃗‖", "n": n,
                    "mean": round(m, 4), "sd": round(_sd(vals), 4),
                    "ci_lower": _ci95(vals)[0], "ci_upper": _ci95(vals)[1],
                })
                if not quiet:
                    row_str += f" {m:>10.3f}"
            if not quiet:
                print(row_str)

    # Export CSV
    filepath = output_dir / "pilot_descriptives.csv"
    if rows:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"\n  → {filepath.name} ({len(rows)} rows)")


# ──────────────────────────────────────────────────
# Inferential Statistics
# ──────────────────────────────────────────────────

def _cohens_d(group1: list[float], group2: list[float]) -> float:
    """Cohen's d with pooled standard deviation."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    m1, m2 = _mean(group1), _mean(group2)
    s1, s2 = _sd(group1), _sd(group2)
    pooled = math.sqrt((s1 ** 2 + s2 ** 2) / 2)
    if pooled == 0:
        return 0.0
    return (m1 - m2) / pooled


def _ci_diff_95(group1: list[float], group2: list[float]) -> tuple[float, float]:
    """95% CI for mean difference using Welch-Satterthwaite df."""
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2:
        diff = _mean(group1) - _mean(group2)
        return (diff, diff)
    m1, m2 = _mean(group1), _mean(group2)
    s1, s2 = _sd(group1), _sd(group2)
    se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    # Welch-Satterthwaite degrees of freedom
    num = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 2
    denom = (s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1)
    df = num / denom if denom > 0 else 1
    t_crit = stats.t.ppf(0.975, df=df)
    diff = m1 - m2
    return (round(diff - t_crit * se, 4), round(diff + t_crit * se, 4))


def _sig_stars(p: float) -> str:
    """Significance stars."""
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.10:
        return "†"
    return "n.s."


def inferential_statistics(trials: list[dict], output_dir: Path, quiet: bool = False):
    """Run pairwise Welch's t-tests between conditions and export results."""

    by_cond = {}
    for t in trials:
        by_cond.setdefault(t["condition"], []).append(t)

    # Define comparisons: (condition_a, condition_b, [metrics])
    comparisons = [
        ("C3", "C1", ["delta_gamma", "gamma_norm", "gamma_1", "gamma_2", "gamma_3"]),
        ("C3", "C0", ["delta_gamma", "gamma_norm"]),
        ("C3", "C2", ["delta_gamma", "gamma_norm"]),
        ("C2", "C1", ["delta_gamma"]),
        ("C1", "C0", ["delta_gamma"]),
    ]

    metric_labels = {
        "delta_gamma": "ΔΓ",
        "gamma_norm": "‖Γ⃗‖",
        "gamma_1": "γ₁",
        "gamma_2": "γ₂",
        "gamma_3": "γ₃",
    }

    rows = []

    if not quiet:
        print(f"\n  ── Inferenzstatistik: Paarweise Welch's t-Tests ──")
        print(f"  {'Vergleich':<12} {'Metrik':<8} {'n_a':>4} {'n_b':>4} "
              f"{'M_a':>8} {'M_b':>8} {'Diff':>8} {'t':>8} {'df':>6} "
              f"{'p':>8} {'d':>8} {'Sig':>5} {'95% CI Diff':>22}")
        print(f"  {'-' * 120}")

    for cond_a, cond_b, metric_keys in comparisons:
        if cond_a not in by_cond or cond_b not in by_cond:
            continue

        trials_a = by_cond[cond_a]
        trials_b = by_cond[cond_b]

        for key in metric_keys:
            vals_a = _safe_values(trials_a, key)
            vals_b = _safe_values(trials_b, key)

            if len(vals_a) < 2 or len(vals_b) < 2:
                continue

            m_a, m_b = _mean(vals_a), _mean(vals_b)
            diff = m_a - m_b

            # Welch's t-test (equal_var=False)
            t_stat, p_val = stats.ttest_ind(vals_a, vals_b, equal_var=False)

            # Degrees of freedom (Welch-Satterthwaite)
            s1, s2 = _sd(vals_a), _sd(vals_b)
            n1, n2 = len(vals_a), len(vals_b)
            num = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 2
            denom = (s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1)
            df = num / denom if denom > 0 else 1

            d = _cohens_d(vals_a, vals_b)
            ci = _ci_diff_95(vals_a, vals_b)
            sig = _sig_stars(p_val)

            label_a = COND_LABELS.get(cond_a, cond_a)
            label_b = COND_LABELS.get(cond_b, cond_b)
            comparison = f"{cond_a} vs {cond_b}"
            metric_label = metric_labels.get(key, key)

            row = {
                "comparison": comparison,
                "condition_a": cond_a, "label_a": label_a,
                "condition_b": cond_b, "label_b": label_b,
                "metric": metric_label,
                "n_a": len(vals_a), "n_b": len(vals_b),
                "mean_a": round(m_a, 4), "mean_b": round(m_b, 4),
                "diff": round(diff, 4),
                "t_statistic": round(t_stat, 4),
                "df": round(df, 2),
                "p_value": round(p_val, 6),
                "cohens_d": round(d, 4) if not math.isnan(d) else "NA",
                "sig": sig,
                "ci_lower": ci[0], "ci_upper": ci[1],
            }
            rows.append(row)

            if not quiet:
                d_str = f"{d:>8.3f}" if not math.isnan(d) else f"{'NA':>8}"
                print(f"  {comparison:<12} {metric_label:<8} {len(vals_a):>4} {len(vals_b):>4} "
                      f"{m_a:>8.3f} {m_b:>8.3f} {diff:>+8.3f} {t_stat:>8.3f} {df:>6.1f} "
                      f"{p_val:>8.4f} {d_str} {sig:>5} "
                      f"[{ci[0]:>+8.3f}, {ci[1]:>+8.3f}]")

    if not quiet:
        print(f"\n  Legende: *** p<.001  ** p<.01  * p<.05  † p<.10  n.s. nicht signifikant")

    # Export CSV
    filepath = output_dir / "pilot_statistics.csv"
    if rows:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"  → {filepath.name} ({len(rows)} tests)")


# ──────────────────────────────────────────────────
# Quality Flags
# ──────────────────────────────────────────────────

def quality_summary(trials: list[dict], quiet: bool = False):
    """Print quality flag summary."""
    flagged = [t for t in trials if t.get("quality_flag")]
    total = len(trials)

    if not quiet:
        print(f"\n  ── Quality Flags ──")
        print(f"  {len(flagged)} von {total} Trials geflaggt "
              f"({100 * len(flagged) / total:.1f}%)")

    if flagged and not quiet:
        # By condition
        conds = sorted(set(t["condition"] for t in flagged))
        for cond in conds:
            cond_flagged = [t for t in flagged if t["condition"] == cond]
            print(f"    {cond}: {len(cond_flagged)} Flags")

        print()
        for t in flagged[:15]:
            tid = f"{t['model']}/{t['topic']}/{t['condition']}/rep{t['repetition']}"
            print(f"    {tid}: {t['quality_flag']}")
        if len(flagged) > 15:
            print(f"    ... und {len(flagged) - 15} weitere")


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Γ-Modulation Pilot Study: Statistical Analysis & CSV Export",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python analyze_pilot.py
  python analyze_pilot.py --results-dir results_pilot
  python analyze_pilot.py --output-dir analysis_output/
  python analyze_pilot.py --quiet
        """
    )
    parser.add_argument("--results-dir", type=str, default="results_pilot",
                        help="Directory with trial JSONs (default: results_pilot)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory for CSV output (default: same as results-dir)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress terminal output (only write CSVs)")

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir
    output_dir.mkdir(exist_ok=True)

    if not args.quiet:
        print(f"\n{'=' * 60}")
        print(f"  Γ-MODULATION PILOT: STATISTISCHE ANALYSE")
        print(f"  Quelle: {results_dir}/")
        print(f"  Ausgabe: {output_dir}/")
        print(f"{'=' * 60}")

    # Load
    trials = load_trials(results_dir)
    if not trials:
        return

    if not args.quiet:
        models = sorted(set(t["model"] for t in trials))
        conditions = sorted(set(t["condition"] for t in trials))
        print(f"\n  Geladen: {len(trials)} Trials")
        print(f"  Modelle: {', '.join(models)}")
        print(f"  Conditions: {', '.join(conditions)}")

    # CSV Export
    if not args.quiet:
        print(f"\n  ── CSV Export ──")
    export_csv_full(trials, output_dir)
    export_csv_scores(trials, output_dir)

    # Descriptive Statistics
    descriptive_statistics(trials, output_dir, quiet=args.quiet)

    # Inferential Statistics
    inferential_statistics(trials, output_dir, quiet=args.quiet)

    # Quality Flags
    quality_summary(trials, quiet=args.quiet)

    if not args.quiet:
        print(f"\n{'=' * 60}")
        print(f"  Analyse abgeschlossen. CSV-Dateien in {output_dir}/")
        print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
