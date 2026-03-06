"""
Gamma-Modulation Experiment: Comprehensive Statistical Analysis
================================================================
Standalone analysis script for the full experiment dataset.
Reads trial JSONs, exports CSVs, runs descriptive + inferential statistics
across Conditions, Models, Species, and Model x Condition interactions.

Usage:
    python analyze_experiment.py
    python analyze_experiment.py --results-dir results_pilot
    python analyze_experiment.py --output-dir analysis_output/
    python analyze_experiment.py --quiet
"""

import csv
import json
import math
import argparse
import statistics
from pathlib import Path
from itertools import combinations

from scipy import stats


# ──────────────────────────────────────────────────
# Model Metadata (mirrors MODELS dict from run_experiment_v2.py)
# ──────────────────────────────────────────────────

MODEL_META = {
    "llama2_13b": {"display": "Llama-2-13B-Chat", "species": "II", "params": "13B"},
    "deepseek_r1": {"display": "DeepSeek-R1-Qwen3-8B", "species": "III", "params": "8B"},
    "qwen3_8b": {"display": "Qwen3-8B", "species": "II", "params": "8B"},
    "gpt_oss_20b": {"display": "GPT-OSS-20B", "species": "I", "params": "20B (3.6B active)"},
    "claude": {"display": "Claude Sonnet", "species": "IV", "params": "N/A"},
    "claude_opus": {"display": "Claude Opus", "species": "IV", "params": "N/A"},
    "gpt4o": {"display": "GPT-4o", "species": "IV", "params": "N/A"},
}

COND_LABELS = {
    "C0": "Control (neutral)",
    "C1": "Consequence-Constraint",
    "C2": "Socratic Questioning",
    "C3": "Kenotic Prompting",
}

SPECIES_LABELS = {
    "I": "Species I (Open-weight, MoE)",
    "II": "Species II (Open-weight, Dense)",
    "III": "Species III (Reasoning/CoT)",
    "IV": "Species IV (Commercial/Closed)",
}


# ──────────────────────────────────────────────────
# JSON Loading & Flattening
# ──────────────────────────────────────────────────

def load_trials(results_dir: Path) -> list[dict]:
    """Load all trial JSONs from results directory and flatten nested fields."""
    trial_files = sorted(results_dir.glob("*.json"))
    trial_files = [f for f in trial_files if not f.name.startswith("_")]

    if not trial_files:
        print(f"  Keine Trial-Dateien in {results_dir}/")
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

    # Gamma vector
    gv = raw.get("gamma_vector", [])
    flat["gamma_1"] = gv[0] if len(gv) > 0 else None
    flat["gamma_2"] = gv[1] if len(gv) > 1 else None
    flat["gamma_3"] = gv[2] if len(gv) > 2 else None

    # Score dicts
    for score_field in [
        "position_depth_initial", "position_depth_revised",
        "self_ref_depth_initial", "self_ref_depth_revised",
        "hypothesis_diversity", "revision_genuineness",
        "persistence", "structural_direction",
    ]:
        score_dict = raw.get(score_field, {})
        flat[f"{score_field}_score"] = score_dict.get("score", None)
        flat[f"{score_field}_reasoning"] = score_dict.get("reasoning", "")

    # Sycophancy
    syc = raw.get("sycophancy_keywords", {})
    flat["sycophancy_agreement_count"] = syc.get("agreement_count", 0)
    flat["sycophancy_resistance_count"] = syc.get("resistance_count", 0)
    flat["sycophancy_agreement_ratio"] = syc.get("agreement_ratio", 0.0)
    flat["sycophancy_flag"] = syc.get("flag", "")

    # Enrich with species from MODEL_META
    model_key = flat["model"]
    meta = MODEL_META.get(model_key, {})
    flat["species"] = meta.get("species", "?")
    flat["display_name"] = meta.get("display", model_key)

    return flat


# ──────────────────────────────────────────────────
# CSV Export
# ──────────────────────────────────────────────────

SCORES_COLUMNS = [
    "model", "display_name", "species", "topic", "condition", "repetition",
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
    if not trials:
        return
    filepath = output_dir / "experiment_trials_full.csv"
    fieldnames = list(trials[0].keys())
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(trials)
    print(f"  -> {filepath.name} ({len(trials)} trials, {len(fieldnames)} columns)")


def export_csv_scores(trials: list[dict], output_dir: Path):
    if not trials:
        return
    filepath = output_dir / "experiment_trials_scores.csv"
    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=SCORES_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(trials)
    print(f"  -> {filepath.name} ({len(trials)} trials, {len(SCORES_COLUMNS)} columns)")


# ──────────────────────────────────────────────────
# Statistical Helpers
# ──────────────────────────────────────────────────

def _vals(trials: list[dict], key: str) -> list[float]:
    """Extract numeric values, filtering None."""
    return [t[key] for t in trials if t.get(key) is not None]


def _mean(vals: list[float]) -> float:
    return sum(vals) / len(vals) if vals else 0.0


def _sd(vals: list[float]) -> float:
    return statistics.stdev(vals) if len(vals) > 1 else 0.0


def _ci95(vals: list[float]) -> tuple[float, float]:
    n = len(vals)
    if n < 2:
        m = _mean(vals)
        return (m, m)
    m = _mean(vals)
    se = _sd(vals) / math.sqrt(n)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    return (round(m - t_crit * se, 4), round(m + t_crit * se, 4))


def _cohens_d(g1: list[float], g2: list[float]) -> float:
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return float("nan")
    s1, s2 = _sd(g1), _sd(g2)
    pooled = math.sqrt((s1 ** 2 + s2 ** 2) / 2)
    if pooled == 0:
        return 0.0
    return (_mean(g1) - _mean(g2)) / pooled


def _ci_diff_95(g1: list[float], g2: list[float]) -> tuple[float, float]:
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        diff = _mean(g1) - _mean(g2)
        return (diff, diff)
    s1, s2 = _sd(g1), _sd(g2)
    se = math.sqrt(s1 ** 2 / n1 + s2 ** 2 / n2)
    num = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 2
    denom = (s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1)
    df = num / denom if denom > 0 else 1
    t_crit = stats.t.ppf(0.975, df=df)
    diff = _mean(g1) - _mean(g2)
    return (round(diff - t_crit * se, 4), round(diff + t_crit * se, 4))


def _sig(p: float) -> str:
    if p < 0.001:
        return "***"
    elif p < 0.01:
        return "**"
    elif p < 0.05:
        return "*"
    elif p < 0.10:
        return "+"
    return "n.s."


def _welch_test(vals_a, vals_b):
    """Run Welch's t-test, return (t, df, p)."""
    t_stat, p_val = stats.ttest_ind(vals_a, vals_b, equal_var=False)
    s1, s2 = _sd(vals_a), _sd(vals_b)
    n1, n2 = len(vals_a), len(vals_b)
    num = (s1 ** 2 / n1 + s2 ** 2 / n2) ** 2
    denom = (s1 ** 2 / n1) ** 2 / (n1 - 1) + (s2 ** 2 / n2) ** 2 / (n2 - 1)
    df = num / denom if denom > 0 else 1
    return t_stat, df, p_val


# ──────────────────────────────────────────────────
# Section 1: Data Overview
# ──────────────────────────────────────────────────

def print_data_overview(trials: list[dict], quiet: bool = False):
    if quiet:
        return

    models = sorted(set(t["model"] for t in trials))
    conditions = sorted(set(t["condition"] for t in trials))
    topics = sorted(set(t["topic"] for t in trials))
    species = sorted(set(t["species"] for t in trials))

    print(f"\n  Geladen: {len(trials)} Trials")
    print(f"  Modelle: {len(models)} ({', '.join(models)})")
    print(f"  Species: {', '.join(species)}")
    print(f"  Conditions: {', '.join(conditions)}")
    print(f"  Topics: {', '.join(topics)}")

    # Trial count matrix
    print(f"\n  -- Trial-Matrix (Model x Condition) --")
    header = f"  {'Model':<25} {'Species':>7}"
    for c in conditions:
        header += f" {c:>5}"
    header += f" {'Total':>6}"
    print(header)
    print(f"  {'-' * (25 + 7 + 6 * len(conditions) + 7)}")

    for m in models:
        meta = MODEL_META.get(m, {})
        sp = meta.get("species", "?")
        row = f"  {meta.get('display', m):<25} {sp:>7}"
        total = 0
        for c in conditions:
            n = len([t for t in trials if t["model"] == m and t["condition"] == c])
            row += f" {n:>5}"
            total += n
        row += f" {total:>6}"
        print(row)

    total_row = f"  {'TOTAL':<25} {'':>7}"
    for c in conditions:
        n = len([t for t in trials if t["condition"] == c])
        total_row += f" {n:>5}"
    total_row += f" {len(trials):>6}"
    print(total_row)

    # Missing trials
    missing = []
    for m in models:
        for t_topic in topics:
            for c in conditions:
                for rep in range(1, 6):
                    found = any(
                        t["model"] == m and t["topic"] == t_topic
                        and t["condition"] == c and t["repetition"] == rep
                        for t in trials
                    )
                    if not found:
                        missing.append(f"{m}/{t_topic}/{c}/rep{rep}")

    if missing:
        print(f"\n  Fehlende Trials: {len(missing)}")
        for m in missing[:10]:
            print(f"    - {m}")
        if len(missing) > 10:
            print(f"    ... und {len(missing) - 10} weitere")


# ──────────────────────────────────────────────────
# Section 2: Descriptive Statistics
# ──────────────────────────────────────────────────

ALL_METRICS = [
    ("delta_gamma", "DG"),
    ("gamma_norm", "||G||"),
    ("gamma_1", "g1"),
    ("gamma_2", "g2"),
    ("gamma_3", "g3"),
]


def descriptive_by_condition(trials, rows, quiet):
    conditions = sorted(set(t["condition"] for t in trials))

    if not quiet:
        print(f"\n  -- Deskriptive Statistik: Condition --")
        print(f"  {'Condition':<28} {'Metrik':<8} {'n':>4} {'Mean':>8} {'SD':>8} {'95% CI':>20}")
        print(f"  {'-' * 80}")

    for cond in conditions:
        ct = [t for t in trials if t["condition"] == cond]
        label = COND_LABELS.get(cond, cond)
        for key, mname in ALL_METRICS:
            vals = _vals(ct, key)
            n = len(vals)
            m = _mean(vals)
            sd = _sd(vals)
            ci = _ci95(vals)
            rows.append({
                "group_type": "condition", "group": cond, "label": label,
                "metric": mname, "n": n,
                "mean": round(m, 4), "sd": round(sd, 4),
                "ci_lower": ci[0], "ci_upper": ci[1],
            })
            if not quiet:
                print(f"  {label:<28} {mname:<8} {n:>4} {m:>8.3f} {sd:>8.3f} "
                      f"[{ci[0]:>8.3f}, {ci[1]:>8.3f}]")
        if not quiet:
            print()


def descriptive_by_model(trials, rows, quiet):
    models = sorted(set(t["model"] for t in trials))

    if not quiet:
        print(f"  -- Deskriptive Statistik: Model --")
        print(f"  {'Model':<25} {'Sp':>3} {'Metrik':<8} {'n':>4} {'Mean':>8} {'SD':>8} {'95% CI':>20}")
        print(f"  {'-' * 85}")

    for model in models:
        mt = [t for t in trials if t["model"] == model]
        meta = MODEL_META.get(model, {})
        display = meta.get("display", model)
        sp = meta.get("species", "?")
        for key, mname in ALL_METRICS:
            vals = _vals(mt, key)
            n = len(vals)
            m = _mean(vals)
            sd = _sd(vals)
            ci = _ci95(vals)
            rows.append({
                "group_type": "model", "group": model, "label": display,
                "metric": mname, "n": n,
                "mean": round(m, 4), "sd": round(sd, 4),
                "ci_lower": ci[0], "ci_upper": ci[1],
            })
            if not quiet:
                print(f"  {display:<25} {sp:>3} {mname:<8} {n:>4} {m:>8.3f} {sd:>8.3f} "
                      f"[{ci[0]:>8.3f}, {ci[1]:>8.3f}]")
        if not quiet:
            print()


def descriptive_by_species(trials, rows, quiet):
    species_list = sorted(set(t["species"] for t in trials))
    if len(species_list) < 2:
        return

    if not quiet:
        print(f"  -- Deskriptive Statistik: Species --")
        print(f"  {'Species':<35} {'Metrik':<8} {'n':>4} {'Mean':>8} {'SD':>8} {'95% CI':>20}")
        print(f"  {'-' * 90}")

    for sp in species_list:
        st = [t for t in trials if t["species"] == sp]
        label = SPECIES_LABELS.get(sp, f"Species {sp}")
        for key, mname in ALL_METRICS:
            vals = _vals(st, key)
            n = len(vals)
            m = _mean(vals)
            sd = _sd(vals)
            ci = _ci95(vals)
            rows.append({
                "group_type": "species", "group": sp, "label": label,
                "metric": mname, "n": n,
                "mean": round(m, 4), "sd": round(sd, 4),
                "ci_lower": ci[0], "ci_upper": ci[1],
            })
            if not quiet:
                print(f"  {label:<35} {mname:<8} {n:>4} {m:>8.3f} {sd:>8.3f} "
                      f"[{ci[0]:>8.3f}, {ci[1]:>8.3f}]")
        if not quiet:
            print()


def descriptive_model_x_condition(trials, rows, quiet):
    models = sorted(set(t["model"] for t in trials))
    conditions = sorted(set(t["condition"] for t in trials))

    if not quiet:
        print(f"  -- Model x Condition: ||G|| --")
        header = f"  {'Model':<25}"
        for c in conditions:
            header += f" {c:>8}"
        print(header)
        print(f"  {'-' * (25 + 9 * len(conditions))}")

    for model in models:
        meta = MODEL_META.get(model, {})
        display = meta.get("display", model)
        if not quiet:
            row_str = f"  {display:<25}"
        for cond in conditions:
            ct = [t for t in trials if t["model"] == model and t["condition"] == cond]
            vals = _vals(ct, "gamma_norm")
            m = _mean(vals)
            n = len(vals)
            sd = _sd(vals)
            ci = _ci95(vals)
            rows.append({
                "group_type": "model_x_condition",
                "group": f"{model}_{cond}",
                "label": f"{display} x {COND_LABELS.get(cond, cond)}",
                "metric": "||G||", "n": n,
                "mean": round(m, 4), "sd": round(sd, 4),
                "ci_lower": ci[0], "ci_upper": ci[1],
            })
            if not quiet:
                row_str += f" {m:>8.3f}"
        if not quiet:
            print(row_str)

    # Same for delta_gamma
    if not quiet:
        print(f"\n  -- Model x Condition: DG --")
        header = f"  {'Model':<25}"
        for c in conditions:
            header += f" {c:>8}"
        print(header)
        print(f"  {'-' * (25 + 9 * len(conditions))}")

    for model in models:
        meta = MODEL_META.get(model, {})
        display = meta.get("display", model)
        if not quiet:
            row_str = f"  {display:<25}"
        for cond in conditions:
            ct = [t for t in trials if t["model"] == model and t["condition"] == cond]
            vals = _vals(ct, "delta_gamma")
            m = _mean(vals)
            n = len(vals)
            sd = _sd(vals)
            ci = _ci95(vals)
            rows.append({
                "group_type": "model_x_condition",
                "group": f"{model}_{cond}",
                "label": f"{display} x {COND_LABELS.get(cond, cond)}",
                "metric": "DG", "n": n,
                "mean": round(m, 4), "sd": round(sd, 4),
                "ci_lower": ci[0], "ci_upper": ci[1],
            })
            if not quiet:
                row_str += f" {m:>+8.3f}"
        if not quiet:
            print(row_str)


def descriptive_indicators(trials, quiet):
    """Detailed indicator breakdown by condition (hedge, position, self-ref, etc.)."""
    if quiet:
        return

    conditions = sorted(set(t["condition"] for t in trials))

    print(f"\n  -- Detaillierte Indikatoren nach Condition --")
    for cond in conditions:
        ct = [t for t in trials if t["condition"] == cond]
        label = COND_LABELS.get(cond, cond)
        n = len(ct)

        h_init = _vals(ct, "hedge_density_initial")
        h_rev = _vals(ct, "hedge_density_revised")
        pos_init = _vals(ct, "position_depth_initial_score")
        pos_rev = _vals(ct, "position_depth_revised_score")
        sr_init = _vals(ct, "self_ref_depth_initial_score")
        sr_rev = _vals(ct, "self_ref_depth_revised_score")
        rev_gen = _vals(ct, "revision_genuineness_score")
        persist = _vals(ct, "persistence_score")
        hyp_div = _vals(ct, "hypothesis_diversity_score")
        str_dir = _vals(ct, "structural_direction_score")

        print(f"\n  {label} (n={n}):")
        print(f"    Hedge density:    {_mean(h_init):.3f} -> {_mean(h_rev):.3f}  "
              f"(D={_mean(h_rev) - _mean(h_init):+.3f})")
        print(f"    Position depth:   {_mean(pos_init):.1f} -> {_mean(pos_rev):.1f}  "
              f"(D={_mean(pos_rev) - _mean(pos_init):+.1f})")
        print(f"    Self-ref depth:   {_mean(sr_init):.1f} -> {_mean(sr_rev):.1f}  "
              f"(D={_mean(sr_rev) - _mean(sr_init):+.1f})")
        print(f"    Revision genuine: {_mean(rev_gen):.1f}")
        print(f"    Persistence:      {_mean(persist):.1f}")
        print(f"    Hypothesis div:   {_mean(hyp_div):.1f}")
        print(f"    Structural dir:   {_mean(str_dir):.1f}")


def run_descriptives(trials, output_dir, quiet):
    rows = []
    descriptive_by_condition(trials, rows, quiet)
    descriptive_by_model(trials, rows, quiet)
    descriptive_by_species(trials, rows, quiet)
    descriptive_model_x_condition(trials, rows, quiet)
    descriptive_indicators(trials, quiet)

    filepath = output_dir / "experiment_descriptives.csv"
    if rows:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        if not quiet:
            print(f"\n  -> {filepath.name} ({len(rows)} rows)")


# ──────────────────────────────────────────────────
# Section 3: Inferential Statistics
# ──────────────────────────────────────────────────

def _run_comparison(label_a, label_b, vals_a, vals_b, metric_key, metric_label):
    """Run a single t-test comparison and return a result row dict."""
    if len(vals_a) < 2 or len(vals_b) < 2:
        return None

    t_stat, df, p_val = _welch_test(vals_a, vals_b)
    d = _cohens_d(vals_a, vals_b)
    ci = _ci_diff_95(vals_a, vals_b)

    return {
        "comparison": f"{label_a} vs {label_b}",
        "group_a": label_a, "group_b": label_b,
        "metric": metric_label,
        "n_a": len(vals_a), "n_b": len(vals_b),
        "mean_a": round(_mean(vals_a), 4), "mean_b": round(_mean(vals_b), 4),
        "diff": round(_mean(vals_a) - _mean(vals_b), 4),
        "t": round(t_stat, 4), "df": round(df, 2), "p": round(p_val, 6),
        "d": round(d, 4) if not math.isnan(d) else "NA",
        "sig": _sig(p_val),
        "ci_lower": ci[0], "ci_upper": ci[1],
    }


def _print_test_header(quiet):
    if quiet:
        return
    print(f"  {'Vergleich':<30} {'Metrik':<6} {'n_a':>4} {'n_b':>4} "
          f"{'M_a':>7} {'M_b':>7} {'Diff':>7} {'t':>7} {'df':>6} "
          f"{'p':>8} {'d':>7} {'Sig':>5} {'95% CI Diff':>20}")
    print(f"  {'-' * 130}")


def _print_test_row(row, quiet):
    if quiet or row is None:
        return
    d_str = f"{row['d']:>7.3f}" if row['d'] != "NA" else f"{'NA':>7}"
    print(f"  {row['comparison']:<30} {row['metric']:<6} {row['n_a']:>4} {row['n_b']:>4} "
          f"{row['mean_a']:>7.3f} {row['mean_b']:>7.3f} {row['diff']:>+7.3f} "
          f"{row['t']:>7.3f} {row['df']:>6.1f} {row['p']:>8.4f} "
          f"{d_str} {row['sig']:>5} "
          f"[{row['ci_lower']:>+7.3f}, {row['ci_upper']:>+7.3f}]")


def condition_tests(trials, quiet):
    """Pairwise Welch's t-tests between conditions (primary hypothesis tests)."""
    by_cond = {}
    for t in trials:
        by_cond.setdefault(t["condition"], []).append(t)

    comparisons = [
        ("C3", "C1", ["delta_gamma", "gamma_norm", "gamma_1", "gamma_2", "gamma_3"]),
        ("C3", "C0", ["delta_gamma", "gamma_norm"]),
        ("C3", "C2", ["delta_gamma", "gamma_norm"]),
        ("C2", "C1", ["delta_gamma"]),
        ("C1", "C0", ["delta_gamma"]),
    ]

    metric_labels = {
        "delta_gamma": "DG", "gamma_norm": "||G||",
        "gamma_1": "g1", "gamma_2": "g2", "gamma_3": "g3",
    }

    rows = []
    if not quiet:
        print(f"\n  -- Hypothesentests: Condition-Vergleiche (alle Modelle gepoolt) --")
        _print_test_header(quiet)

    for ca, cb, keys in comparisons:
        if ca not in by_cond or cb not in by_cond:
            continue
        for key in keys:
            vals_a = _vals(by_cond[ca], key)
            vals_b = _vals(by_cond[cb], key)
            row = _run_comparison(ca, cb, vals_a, vals_b, key, metric_labels.get(key, key))
            if row:
                rows.append(row)
                _print_test_row(row, quiet)

    return rows


def model_tests(trials, quiet):
    """Pairwise model comparisons on key metrics."""
    models = sorted(set(t["model"] for t in trials))
    if len(models) < 2:
        return []

    by_model = {}
    for t in trials:
        by_model.setdefault(t["model"], []).append(t)

    rows = []
    if not quiet:
        print(f"\n  -- Modellvergleiche: Paarweise Welch's t-Tests --")
        _print_test_header(quiet)

    for ma, mb in combinations(models, 2):
        for key, mname in [("delta_gamma", "DG"), ("gamma_norm", "||G||")]:
            vals_a = _vals(by_model[ma], key)
            vals_b = _vals(by_model[mb], key)
            disp_a = MODEL_META.get(ma, {}).get("display", ma)[:12]
            disp_b = MODEL_META.get(mb, {}).get("display", mb)[:12]
            row = _run_comparison(disp_a, disp_b, vals_a, vals_b, key, mname)
            if row:
                rows.append(row)
                _print_test_row(row, quiet)

    return rows


def species_tests(trials, quiet):
    """Pairwise species comparisons."""
    species_list = sorted(set(t["species"] for t in trials))
    if len(species_list) < 2:
        return []

    by_sp = {}
    for t in trials:
        by_sp.setdefault(t["species"], []).append(t)

    rows = []
    if not quiet:
        print(f"\n  -- Species-Vergleiche: Paarweise Welch's t-Tests --")
        _print_test_header(quiet)

    for sa, sb in combinations(species_list, 2):
        for key, mname in [("delta_gamma", "DG"), ("gamma_norm", "||G||")]:
            vals_a = _vals(by_sp[sa], key)
            vals_b = _vals(by_sp[sb], key)
            row = _run_comparison(f"Sp.{sa}", f"Sp.{sb}", vals_a, vals_b, key, mname)
            if row:
                rows.append(row)
                _print_test_row(row, quiet)

    return rows


def kenotic_effect_per_model(trials, quiet):
    """C3 vs C1 comparison within each model (the core hypothesis test)."""
    models = sorted(set(t["model"] for t in trials))
    rows = []

    if not quiet:
        print(f"\n  -- Kenotic-Effekt pro Modell: C3 vs C1 --")
        _print_test_header(quiet)

    for model in models:
        mt = [t for t in trials if t["model"] == model]
        c3 = [t for t in mt if t["condition"] == "C3"]
        c1 = [t for t in mt if t["condition"] == "C1"]
        c0 = [t for t in mt if t["condition"] == "C0"]

        display = MODEL_META.get(model, {}).get("display", model)[:15]

        for key, mname in ALL_METRICS:
            vals_c3 = _vals(c3, key)
            vals_c1 = _vals(c1, key)
            row = _run_comparison(f"{display}:C3", f"{display}:C1",
                                  vals_c3, vals_c1, key, mname)
            if row:
                rows.append(row)
                _print_test_row(row, quiet)

        # Also C3 vs C0 for baseline
        for key, mname in [("delta_gamma", "DG"), ("gamma_norm", "||G||")]:
            vals_c3 = _vals(c3, key)
            vals_c0 = _vals(c0, key)
            row = _run_comparison(f"{display}:C3", f"{display}:C0",
                                  vals_c3, vals_c0, key, mname)
            if row:
                rows.append(row)
                _print_test_row(row, quiet)

    return rows


def run_inferential(trials, output_dir, quiet):
    all_rows = []
    all_rows.extend(condition_tests(trials, quiet))
    all_rows.extend(model_tests(trials, quiet))
    all_rows.extend(species_tests(trials, quiet))
    all_rows.extend(kenotic_effect_per_model(trials, quiet))

    if not quiet:
        print(f"\n  Legende: *** p<.001  ** p<.01  * p<.05  + p<.10  n.s. nicht signifikant")

    filepath = output_dir / "experiment_statistics.csv"
    if all_rows:
        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        if not quiet:
            print(f"  -> {filepath.name} ({len(all_rows)} tests)")


# ──────────────────────────────────────────────────
# Section 4: Kenotic Effect Summary
# ──────────────────────────────────────────────────

def kenotic_summary(trials, quiet):
    """High-level summary of the kenotic prompting effect."""
    if quiet:
        return

    models = sorted(set(t["model"] for t in trials))
    conditions = sorted(set(t["condition"] for t in trials))

    print(f"\n  {'=' * 60}")
    print(f"  KENOTIC EFFECT SUMMARY")
    print(f"  {'=' * 60}")

    # Overall
    c3_all = [t for t in trials if t["condition"] == "C3"]
    c1_all = [t for t in trials if t["condition"] == "C1"]
    c0_all = [t for t in trials if t["condition"] == "C0"]

    dg_c3 = _vals(c3_all, "delta_gamma")
    dg_c1 = _vals(c1_all, "delta_gamma")
    dg_c0 = _vals(c0_all, "delta_gamma")

    print(f"\n  Overall (alle Modelle gepoolt):")
    print(f"    C0 (Control) mean DG:        {_mean(dg_c0):>+.3f}  (n={len(dg_c0)})")
    print(f"    C1 (Consequence) mean DG:     {_mean(dg_c1):>+.3f}  (n={len(dg_c1)})")
    print(f"    C3 (Kenotic) mean DG:         {_mean(dg_c3):>+.3f}  (n={len(dg_c3)})")
    print(f"    C3 - C1 Differenz:            {_mean(dg_c3) - _mean(dg_c1):>+.3f}")
    print(f"    C3 - C0 Differenz:            {_mean(dg_c3) - _mean(dg_c0):>+.3f}")

    if len(dg_c3) >= 2 and len(dg_c0) >= 2:
        d = _cohens_d(dg_c3, dg_c0)
        _, _, p = _welch_test(dg_c3, dg_c0)
        exceeds = _mean(dg_c3) < _mean(dg_c0)
        marker = "Y" if exceeds else "X"
        print(f"    Cohen's d (C3 vs C0):         {d:>+.3f}")
        print(f"    p-Wert (C3 vs C0):            {p:.4f} {_sig(p)}")
        print(f"    [{marker}] Kenotic-Effekt {'ueberschreitet' if exceeds else 'ueberschreitet NICHT'} "
              f"neutrale Baseline")

    # Per model
    print(f"\n  Pro Modell:")
    print(f"  {'Model':<25} {'Sp':>3} {'DG_C0':>7} {'DG_C1':>7} {'DG_C3':>7} "
          f"{'C3-C1':>7} {'d(C3C1)':>8} {'p':>8} {'Sig':>5} {'Effekt':>10}")
    print(f"  {'-' * 100}")

    for model in models:
        mt = [t for t in trials if t["model"] == model]
        meta = MODEL_META.get(model, {})
        display = meta.get("display", model)
        sp = meta.get("species", "?")

        mc0 = _vals([t for t in mt if t["condition"] == "C0"], "delta_gamma")
        mc1 = _vals([t for t in mt if t["condition"] == "C1"], "delta_gamma")
        mc3 = _vals([t for t in mt if t["condition"] == "C3"], "delta_gamma")

        diff = _mean(mc3) - _mean(mc1) if mc3 and mc1 else 0
        d_val = _cohens_d(mc3, mc1) if len(mc3) >= 2 and len(mc1) >= 2 else float("nan")
        p_val = _welch_test(mc3, mc1)[2] if len(mc3) >= 2 and len(mc1) >= 2 else 1.0

        if abs(d_val) >= 0.8:
            effect = "gross"
        elif abs(d_val) >= 0.5:
            effect = "mittel"
        elif abs(d_val) >= 0.2:
            effect = "klein"
        else:
            effect = "minimal"

        d_str = f"{d_val:>8.3f}" if not math.isnan(d_val) else f"{'NA':>8}"
        print(f"  {display:<25} {sp:>3} {_mean(mc0):>+7.3f} {_mean(mc1):>+7.3f} "
              f"{_mean(mc3):>+7.3f} {diff:>+7.3f} {d_str} {p_val:>8.4f} "
              f"{_sig(p_val):>5} {effect:>10}")


# ──────────────────────────────────────────────────
# Section 5: Quality Flags
# ──────────────────────────────────────────────────

def quality_summary(trials, quiet):
    flagged = [t for t in trials if t.get("quality_flag")]
    total = len(trials)

    if quiet:
        return

    print(f"\n  -- Quality Flags --")
    print(f"  {len(flagged)} von {total} Trials geflaggt "
          f"({100 * len(flagged) / total:.1f}%)")

    if flagged:
        # By condition
        conds = sorted(set(t["condition"] for t in flagged))
        for cond in conds:
            cf = [t for t in flagged if t["condition"] == cond]
            print(f"    {cond}: {len(cf)} Flags")

        # By model
        models = sorted(set(t["model"] for t in flagged))
        for model in models:
            mf = [t for t in flagged if t["model"] == model]
            print(f"    {model}: {len(mf)} Flags")

        print()
        for t in flagged[:20]:
            tid = f"{t['model']}/{t['topic']}/{t['condition']}/rep{t['repetition']}"
            print(f"    {tid}: {t['quality_flag']}")
        if len(flagged) > 20:
            print(f"    ... und {len(flagged) - 20} weitere")


# ──────────────────────────────────────────────────
# Section 6: Sycophancy Analysis
# ──────────────────────────────────────────────────

def sycophancy_analysis(trials, quiet):
    if quiet:
        return

    print(f"\n  -- Sycophancy-Analyse --")

    # By condition
    conditions = sorted(set(t["condition"] for t in trials))
    print(f"  {'Condition':<28} {'n':>4} {'Agree':>6} {'Resist':>6} {'Ratio':>6} "
          f"{'AGR_DOM':>8} {'RES_DOM':>8} {'NO_SIG':>7} {'MIXED':>6}")
    print(f"  {'-' * 95}")

    for cond in conditions:
        ct = [t for t in trials if t["condition"] == cond]
        n = len(ct)
        agree = sum(t.get("sycophancy_agreement_count", 0) for t in ct)
        resist = sum(t.get("sycophancy_resistance_count", 0) for t in ct)
        ratio = agree / (agree + resist) if (agree + resist) > 0 else 0

        agr_dom = len([t for t in ct if t.get("sycophancy_flag") == "AGREEMENT_DOMINANT"])
        res_dom = len([t for t in ct if t.get("sycophancy_flag") == "RESISTANCE_DOMINANT"])
        no_sig = len([t for t in ct if t.get("sycophancy_flag") == "NO_SIGNAL"])
        mixed = len([t for t in ct if t.get("sycophancy_flag") == "MIXED"])

        label = COND_LABELS.get(cond, cond)
        print(f"  {label:<28} {n:>4} {agree:>6} {resist:>6} {ratio:>6.2f} "
              f"{agr_dom:>8} {res_dom:>8} {no_sig:>7} {mixed:>6}")

    # By model
    models = sorted(set(t["model"] for t in trials))
    if len(models) > 1:
        print(f"\n  {'Model':<28} {'n':>4} {'Agree':>6} {'Resist':>6} {'Ratio':>6} "
              f"{'AGR_DOM':>8} {'RES_DOM':>8} {'NO_SIG':>7} {'MIXED':>6}")
        print(f"  {'-' * 95}")

        for model in models:
            mt = [t for t in trials if t["model"] == model]
            n = len(mt)
            agree = sum(t.get("sycophancy_agreement_count", 0) for t in mt)
            resist = sum(t.get("sycophancy_resistance_count", 0) for t in mt)
            ratio = agree / (agree + resist) if (agree + resist) > 0 else 0

            agr_dom = len([t for t in mt if t.get("sycophancy_flag") == "AGREEMENT_DOMINANT"])
            res_dom = len([t for t in mt if t.get("sycophancy_flag") == "RESISTANCE_DOMINANT"])
            no_sig = len([t for t in mt if t.get("sycophancy_flag") == "NO_SIGNAL"])
            mixed = len([t for t in mt if t.get("sycophancy_flag") == "MIXED"])

            display = MODEL_META.get(model, {}).get("display", model)
            print(f"  {display:<28} {n:>4} {agree:>6} {resist:>6} {ratio:>6.2f} "
                  f"{agr_dom:>8} {res_dom:>8} {no_sig:>7} {mixed:>6}")


# ──────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Gamma-Modulation Experiment: Comprehensive Statistical Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
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
        print(f"\n{'=' * 70}")
        print(f"  GAMMA-MODULATION EXPERIMENT: STATISTISCHE ANALYSE")
        print(f"  Quelle: {results_dir}/")
        print(f"  Ausgabe: {output_dir}/")
        print(f"{'=' * 70}")

    # Load
    trials = load_trials(results_dir)
    if not trials:
        return

    # Overview
    print_data_overview(trials, args.quiet)

    # CSV Export
    if not args.quiet:
        print(f"\n  -- CSV Export --")
    export_csv_full(trials, output_dir)
    export_csv_scores(trials, output_dir)

    # Descriptive Statistics
    run_descriptives(trials, output_dir, args.quiet)

    # Inferential Statistics
    run_inferential(trials, output_dir, args.quiet)

    # Kenotic Effect Summary
    kenotic_summary(trials, args.quiet)

    # Quality Flags
    quality_summary(trials, args.quiet)

    # Sycophancy Analysis
    sycophancy_analysis(trials, args.quiet)

    if not args.quiet:
        print(f"\n{'=' * 70}")
        print(f"  Analyse abgeschlossen. CSV-Dateien in {output_dir}/")
        print(f"{'=' * 70}\n")


if __name__ == "__main__":
    main()
