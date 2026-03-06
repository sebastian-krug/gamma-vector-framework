"""
Export all trial JSONs to Schema A CSV.
Usage: python experiments/02_kenotic/export_csv.py
"""

import csv
import json
import statistics
from pathlib import Path

RESULTS_DIR = Path(__file__).parent / "results" / "results_v2"
OUTPUT_CSV = Path(__file__).parent / "results" / "kenotic_v2_all_300.csv"

JUDGE_DIMS = [
    "position_depth_initial", "position_depth_revised",
    "self_ref_depth_initial", "self_ref_depth_revised",
    "hypothesis_diversity", "revision_genuineness",
    "persistence", "structural_direction",
]

# Schema A column names
CSV_COLUMNS = [
    "trial_id", "experiment", "model", "topic", "condition", "repetition", "timestamp",
    # Judge scores (final median)
    "pos_depth_initial", "pos_depth_revised",
    "self_ref_initial", "self_ref_revised",
    "hypothesis_div", "revision_genuine", "persistence", "structural_dir",
    # Second judge scores
    "pos_depth_initial_2", "pos_depth_revised_2",
    "self_ref_initial_2", "self_ref_revised_2",
    "hypothesis_div_2", "revision_genuine_2", "persistence_2", "structural_dir_2",
    # Judge agreement
    "judge_agreement",
    # Hedge densities
    "hedge_initial", "hedge_revised", "hedge_final",
    # Gamma vector
    "gamma_1", "gamma_2", "gamma_3", "gamma_norm",
    # Sycophancy
    "syc_agreement_count", "syc_resistance_count", "syc_flag",
    # Meta
    "judge_scale", "gamma_version",
]

# Map from JSON dim field -> (csv_score_col, csv_score2_col)
DIM_TO_CSV = {
    "position_depth_initial":  ("pos_depth_initial",  "pos_depth_initial_2"),
    "position_depth_revised":  ("pos_depth_revised",  "pos_depth_revised_2"),
    "self_ref_depth_initial":  ("self_ref_initial",    "self_ref_initial_2"),
    "self_ref_depth_revised":  ("self_ref_revised",    "self_ref_revised_2"),
    "hypothesis_diversity":    ("hypothesis_div",      "hypothesis_div_2"),
    "revision_genuineness":    ("revision_genuine",    "revision_genuine_2"),
    "persistence":             ("persistence",         "persistence_2"),
    "structural_direction":    ("structural_dir",      "structural_dir_2"),
}


def export():
    trial_files = sorted(RESULTS_DIR.glob("*.json"))
    trial_files = [f for f in trial_files if not f.name.startswith("_")]

    rows = []
    for filepath in trial_files:
        with open(filepath, "r", encoding="utf-8") as f:
            d = json.load(f)

        model = d.get("model", "")
        topic = d.get("topic", "")
        condition = d.get("condition", "")
        rep = d.get("repetition", 0)

        row = {
            "trial_id": f"exp02_{model}_{topic}_{condition}_rep{rep:02d}",
            "experiment": "02_kenotic",
            "model": model,
            "topic": topic,
            "condition": condition,
            "repetition": rep,
            "timestamp": d.get("timestamp", ""),
        }

        # Judge scores
        agreements = []
        for dim_field, (csv_col, csv_col_2) in DIM_TO_CSV.items():
            dim_data = d.get(dim_field, {})
            row[csv_col] = dim_data.get("score", -1)
            row[csv_col_2] = dim_data.get("score_2", -1)
            ag = dim_data.get("judge_agreement", 0)
            if isinstance(ag, (int, float)):
                agreements.append(ag)

        row["judge_agreement"] = round(statistics.mean(agreements), 2) if agreements else 0

        # Hedge densities
        row["hedge_initial"] = round(d.get("hedge_density_initial", 0), 4)
        row["hedge_revised"] = round(d.get("hedge_density_revised", 0), 4)
        row["hedge_final"] = round(d.get("hedge_density_final", 0), 4)

        # Gamma vector
        gv = d.get("gamma_vector", [0, 0, 0])
        row["gamma_1"] = round(gv[0], 4) if len(gv) > 0 else 0
        row["gamma_2"] = round(gv[1], 4) if len(gv) > 1 else 0
        row["gamma_3"] = round(gv[2], 4) if len(gv) > 2 else 0
        row["gamma_norm"] = round(d.get("gamma_norm", 0), 4)

        # Sycophancy
        syc = d.get("sycophancy_keywords", {})
        if isinstance(syc, dict):
            row["syc_agreement_count"] = syc.get("agreement_count", 0)
            row["syc_resistance_count"] = syc.get("resistance_count", 0)
            flag = syc.get("flag", "NO_SIGNAL")
            row["syc_flag"] = flag if flag else "NO_SIGNAL"
        else:
            row["syc_agreement_count"] = 0
            row["syc_resistance_count"] = 0
            row["syc_flag"] = "NO_SIGNAL"

        # Meta
        row["judge_scale"] = d.get("judge_scale", 10)
        row["gamma_version"] = d.get("gamma_version", "v2")

        rows.append(row)

    # Write CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)

    # Validate
    bad_scores = sum(1 for r in rows if any(r[c] <= 0 for c in [
        "pos_depth_initial", "pos_depth_revised", "self_ref_initial", "self_ref_revised",
        "hypothesis_div", "revision_genuine", "persistence", "structural_dir"
    ]))

    print(f"Exported {len(rows)} trials to {OUTPUT_CSV}")
    print(f"  Scores <= 0: {bad_scores}")
    print(f"  Models: {sorted(set(r['model'] for r in rows))}")
    if bad_scores == 0:
        print("  OK!")
    else:
        print("  FEHLER: Noch fehlende Scores!")


if __name__ == "__main__":
    export()
