"""
Batch Judge Scoring via Anthropic Message Batches API
=====================================================
Handles batch submission, polling, retrieval, and merging of judge scores
for the Γ-Modulation Experiment v2.

Usage (called via run.py):
    python run.py --phase score-batch
    python run.py --phase score-batch --batch-id msgbatch_xxx
"""

import json
import math
import re
import sys
import time
from pathlib import Path
from datetime import datetime

# ── Project root on sys.path for shared/ imports ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import from shared/ modules (not from local run_experiment_v2)
from shared.gamma import compute_gamma_vector, compute_gamma_v1_compat, compute_hedge_density
from shared.judge import judge_score, judge_score_double, get_judge_config, parse_judge_response
from shared.sycophancy import detect_sycophancy, check_judge_keyword_divergence

# Import experiment-specific config from local run.py
from run import MODELS, JUDGE_MODEL_ID, JUDGE_SCALE, JUDGE_SYSTEM, SCORING_PROMPTS
from run import DOUBLE_JUDGE, TIEBREAKER_THRESHOLD


# ──────────────────────────────────────────────────
# Batch Request Building
# ──────────────────────────────────────────────────

# Defines how each judge dimension maps to trial data fields.
# Each entry: (dimension_field, dimension_template, {template_kwarg: trial_field})
JUDGE_DIMENSIONS = [
    ("position_depth_initial", "position_depth", {"response": "initial_response"}),
    ("position_depth_revised", "position_depth", {"response": "revised_response"}),
    ("self_ref_depth_initial", "self_reference_depth", {"response": "initial_response"}),
    ("self_ref_depth_revised", "self_reference_depth", {"response": "revised_response"}),
    ("hypothesis_diversity", "hypothesis_diversity", {"response": "revised_response"}),
    ("revision_genuineness", "revision_genuineness", {"initial": "initial_response", "revised": "revised_response"}),
    ("persistence", "persistence", {"revised": "revised_response", "final": "final_response"}),
    ("structural_direction", "structural_direction", {"revised": "revised_response", "counter": "counter_prompt", "final": "final_response"}),
]


def _trial_id(trial: dict) -> str:
    """Build unique trial identifier for batch custom_id."""
    return f"{trial['model']}_{trial['topic']}_{trial['condition']}_rep{trial['repetition']}"


def _needs_scoring(trial: dict) -> bool:
    """Check if a trial still needs judge scoring (no scores yet)."""
    pd = trial.get("position_depth_initial", {})
    return not pd or pd.get("score", 0) <= 0


def build_judge_requests(results_dir: Path) -> tuple[list[dict], list[dict]]:
    """
    Load partial trial JSONs and build batch requests for all that need scoring.

    With DOUBLE_JUDGE=True, creates 2 requests per dimension (eval1 + eval2)
    for inter-rater reliability. Tiebreaker (eval3) is handled in merge_scores()
    via real-time API if needed.

    Returns:
        (batch_requests, trial_data_list)
    """
    trial_files = sorted(results_dir.glob("*.json"))
    trial_files = [f for f in trial_files if not f.name.startswith("_")]

    if not trial_files:
        print(f"  No trial files found in {results_dir}/")
        return [], []

    trials_needing_scoring = []
    all_trials = []

    for filepath in trial_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            trial = json.load(f)
        trial["_filepath"] = str(filepath)
        all_trials.append(trial)

        if _needs_scoring(trial):
            trials_needing_scoring.append(trial)

    if not trials_needing_scoring:
        print(f"  All {len(all_trials)} trials already have scores.")
        return [], all_trials

    print(f"  Found {len(trials_needing_scoring)} trials needing scoring "
          f"(of {len(all_trials)} total)")

    n_evals = 2 if DOUBLE_JUDGE else 1
    batch_requests = []
    for trial in trials_needing_scoring:
        tid = _trial_id(trial)

        for dim_field, dim_template, field_mapping in JUDGE_DIMENSIONS:
            template = SCORING_PROMPTS[dim_template]
            kwargs = {k: trial[v] for k, v in field_mapping.items()}
            prompt = template.format(**kwargs)

            for eval_num in range(1, n_evals + 1):
                if DOUBLE_JUDGE:
                    custom_id = f"{tid}__{dim_field}__eval{eval_num}"
                else:
                    custom_id = f"{tid}__{dim_field}"

                batch_requests.append({
                    "custom_id": custom_id,
                    "params": {
                        "model": JUDGE_MODEL_ID,
                        "max_tokens": 256,
                        "system": JUDGE_SYSTEM,
                        "messages": [{"role": "user", "content": prompt}]
                    }
                })

    dj_tag = " [double-judge]" if DOUBLE_JUDGE else ""
    print(f"  Built {len(batch_requests)} batch requests{dj_tag} "
          f"({len(trials_needing_scoring)} trials × {len(JUDGE_DIMENSIONS)} dims × {n_evals} evals)")

    return batch_requests, all_trials


# ──────────────────────────────────────────────────
# Batch Submission & Polling
# ──────────────────────────────────────────────────

def submit_batch(requests: list[dict], results_dir: Path) -> str:
    """Submit batch to Anthropic Message Batches API. Returns batch_id."""
    from anthropic import Anthropic
    client = Anthropic()

    print(f"  Submitting batch of {len(requests)} requests...")
    batch = client.messages.batches.create(requests=requests)
    batch_id = batch.id

    meta_path = results_dir / "_batch_meta.json"
    meta = {
        "batch_id": batch_id,
        "submitted_at": datetime.now().isoformat(),
        "request_count": len(requests),
        "status": "submitted",
    }
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)

    print(f"  ✓ Batch submitted: {batch_id}")
    print(f"  Metadata saved to {meta_path}")
    return batch_id


def poll_batch(batch_id: str, poll_interval: int = 60) -> str:
    """Poll batch status until complete. Returns final status."""
    from anthropic import Anthropic
    client = Anthropic()

    print(f"\n  Polling batch {batch_id} every {poll_interval}s...")

    while True:
        batch = client.messages.batches.retrieve(batch_id)
        counts = batch.request_counts

        status = batch.processing_status
        print(f"  [{datetime.now().strftime('%H:%M:%S')}] "
              f"Status: {status} | "
              f"Succeeded: {counts.succeeded} | "
              f"Errored: {counts.errored} | "
              f"Processing: {counts.processing}")

        if status == "ended":
            print(f"\n  ✓ Batch complete!")
            if counts.errored > 0:
                print(f"  ⚠ {counts.errored} requests errored")
            return "ended"

        if status in ("canceled", "canceling", "expired", "failed"):
            print(f"\n  ✗ Batch {status}!")
            return status

        time.sleep(poll_interval)


# ──────────────────────────────────────────────────
# Result Retrieval & Parsing
# ──────────────────────────────────────────────────

def parse_judge_json(text: str) -> dict:
    """Parse judge response JSON (uses shared parse_judge_response)."""
    return parse_judge_response(text, "batch")


def retrieve_results(batch_id: str) -> dict[str, dict]:
    """Retrieve batch results and parse into {custom_id: {score, reasoning}}."""
    from anthropic import Anthropic
    client = Anthropic()

    print(f"  Retrieving results for batch {batch_id}...")
    results = {}
    succeeded = 0
    errored = 0

    for result in client.messages.batches.results(batch_id):
        custom_id = result.custom_id

        if result.result.type == "succeeded":
            text = result.result.message.content[0].text
            parsed = parse_judge_json(text)
            results[custom_id] = parsed
            succeeded += 1
        else:
            error_msg = getattr(result.result, 'error', {})
            results[custom_id] = {
                "score": -1,
                "reasoning": f"Batch error: {result.result.type} - {error_msg}"
            }
            errored += 1

    print(f"  Retrieved {succeeded} succeeded, {errored} errored results")
    return results


# ──────────────────────────────────────────────────
# Score Merging & Gamma Computation
# ──────────────────────────────────────────────────

def _combine_double_judge(batch_results: dict, tid: str, dim_field: str) -> tuple[dict, bool]:
    """
    Combine two batch evaluations into a double-judge score dict.

    Returns:
        (score_dict, needs_tiebreaker)
        score_dict has: score, reasoning, score_1, reasoning_1, score_2, reasoning_2,
                        judge_agreement, tiebreaker, score_3 (None), reasoning_3 (None)
    """
    import statistics

    id1 = f"{tid}__{dim_field}__eval1"
    id2 = f"{tid}__{dim_field}__eval2"

    eval_1 = batch_results.get(id1, {"score": -1, "reasoning": "Missing eval1"})
    eval_2 = batch_results.get(id2, {"score": -1, "reasoning": "Missing eval2"})

    s1 = eval_1.get("score", -1)
    s2 = eval_2.get("score", -1)
    divergence = abs(s1 - s2) if (s1 > 0 and s2 > 0) else 0

    result = {
        "score_1": s1,
        "reasoning_1": eval_1.get("reasoning", ""),
        "score_2": s2,
        "reasoning_2": eval_2.get("reasoning", ""),
        "score_3": None,
        "reasoning_3": None,
        "judge_agreement": divergence,
        "tiebreaker": False,
    }

    # If either eval failed, mark for retry
    valid_scores = [s for s in [s1, s2] if s > 0]
    if not valid_scores:
        result["score"] = -1
        result["reasoning"] = "Both evaluations failed"
        return result, False

    if len(valid_scores) == 1:
        result["score"] = valid_scores[0]
        result["reasoning"] = eval_1["reasoning"] if s1 > 0 else eval_2["reasoning"]
        return result, False

    # Both valid — check if tiebreaker needed
    needs_tiebreaker = divergence > TIEBREAKER_THRESHOLD
    if not needs_tiebreaker:
        final_score = int(statistics.median([s1, s2]))
        result["score"] = final_score
        closest = eval_1 if abs(s1 - final_score) <= abs(s2 - final_score) else eval_2
        result["reasoning"] = closest["reasoning"]
    else:
        # Tiebreaker will be resolved by retry_tiebreakers()
        # Use median of 2 for now, will be updated with 3rd score
        result["score"] = int(statistics.median([s1, s2]))
        result["reasoning"] = eval_1["reasoning"]  # placeholder

    return result, needs_tiebreaker


def merge_scores(results_dir: Path, batch_results: dict[str, dict]):
    """Merge batch judge scores back into trial JSONs and compute gamma.

    With DOUBLE_JUDGE=True, combines eval1+eval2 per dimension, computes
    median, and identifies tiebreaker-needed dimensions.
    """
    trial_files = sorted(results_dir.glob("*.json"))
    trial_files = [f for f in trial_files if not f.name.startswith("_")]

    updated = 0
    failed_ids = []
    tiebreaker_ids = []

    for filepath in trial_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            trial = json.load(f)

        if not _needs_scoring(trial):
            continue

        tid = _trial_id(trial)

        # Merge each dimension's score
        judge_agreement = {}
        for dim_field, _, _ in JUDGE_DIMENSIONS:
            if DOUBLE_JUDGE:
                score_data, needs_tb = _combine_double_judge(batch_results, tid, dim_field)
                if needs_tb:
                    tiebreaker_ids.append(f"{tid}__{dim_field}")
            else:
                custom_id = f"{tid}__{dim_field}"
                score_data = batch_results.get(custom_id, {"score": -1, "reasoning": "Missing from batch"})

            trial[dim_field] = score_data
            if score_data.get("score", -1) == -1:
                failed_ids.append(f"{tid}__{dim_field}")
            if "judge_agreement" in score_data:
                judge_agreement[dim_field] = score_data["judge_agreement"]

        trial["judge_agreement"] = judge_agreement
        trial["double_judge"] = DOUBLE_JUDGE

        # Sycophancy cross-check
        trial["sycophancy_keywords"] = detect_sycophancy(trial.get("final_response", ""))

        sd_score = trial.get("structural_direction", {}).get("score", 3)
        if sd_score > 0:
            flag = check_judge_keyword_divergence(sd_score, trial["sycophancy_keywords"], scale=JUDGE_SCALE)
            if flag:
                trial["quality_flag"] = flag

        # Compute Γ-vector
        pi = trial.get("position_depth_initial", {}).get("score", 3)
        pr = trial.get("position_depth_revised", {}).get("score", 3)
        si = trial.get("self_ref_depth_initial", {}).get("score", 3)
        sr = trial.get("self_ref_depth_revised", {}).get("score", 3)
        hd = trial.get("hypothesis_diversity", {}).get("score", 3)
        rg = trial.get("revision_genuineness", {}).get("score", 3)
        pe = trial.get("persistence", {}).get("score", 3)
        sd = trial.get("structural_direction", {}).get("score", 3)

        hedge_delta = trial.get("hedge_density_revised", 0) - trial.get("hedge_density_initial", 0)
        position_delta = pr - pi
        self_ref_delta = sr - si

        gamma_vector, gamma_norm = compute_gamma_vector(
            hedge_delta=hedge_delta,
            position_delta=position_delta,
            self_ref_delta=self_ref_delta,
            hypothesis_diversity=hd,
            revision=rg,
            structural_direction=sd,
            scale=JUDGE_SCALE,
        )
        trial["gamma_vector"] = gamma_vector
        trial["gamma_norm"] = gamma_norm

        # v1-compatible scalars
        trial["gamma_initial"] = compute_gamma_v1_compat(
            trial.get("hedge_density_initial", 0), pi, si)
        trial["gamma_revised"] = compute_gamma_v1_compat(
            trial.get("hedge_density_revised", 0), pr, sr, rg, pe)
        trial["gamma_final"] = compute_gamma_v1_compat(
            trial.get("hedge_density_final", 0), pr, sr, rg, pe)
        trial["delta_gamma"] = round(trial["gamma_revised"] - trial["gamma_initial"], 3)

        # Remove internal field before saving
        trial.pop("_filepath", None)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trial, f, indent=2, ensure_ascii=False)

        updated += 1

    print(f"  ✓ Updated {updated} trial files with scores and gamma vectors")

    if tiebreaker_ids:
        print(f"  ⚠ {len(tiebreaker_ids)} dimensions need tiebreaker (Δ > {TIEBREAKER_THRESHOLD})")

    if failed_ids:
        print(f"  ⚠ {len(failed_ids)} dimensions had parse errors:")
        for fid in failed_ids[:10]:
            print(f"    - {fid}")
        if len(failed_ids) > 10:
            print(f"    ... and {len(failed_ids) - 10} more")

    return failed_ids, tiebreaker_ids


def _recompute_gamma_for_trial(trial: dict):
    """Recompute all gamma values for a trial dict (used after score updates)."""
    pi = trial.get("position_depth_initial", {}).get("score", 3)
    pr = trial.get("position_depth_revised", {}).get("score", 3)
    si = trial.get("self_ref_depth_initial", {}).get("score", 3)
    sr = trial.get("self_ref_depth_revised", {}).get("score", 3)
    hd = trial.get("hypothesis_diversity", {}).get("score", 3)
    rg = trial.get("revision_genuineness", {}).get("score", 3)
    pe = trial.get("persistence", {}).get("score", 3)
    sd_val = trial.get("structural_direction", {}).get("score", 3)

    hedge_delta = trial.get("hedge_density_revised", 0) - trial.get("hedge_density_initial", 0)
    gamma_vector, gamma_norm = compute_gamma_vector(
        hedge_delta=hedge_delta,
        position_delta=pr - pi,
        self_ref_delta=sr - si,
        hypothesis_diversity=hd,
        revision=rg,
        structural_direction=sd_val,
        scale=JUDGE_SCALE,
    )
    trial["gamma_vector"] = gamma_vector
    trial["gamma_norm"] = gamma_norm
    trial["gamma_initial"] = compute_gamma_v1_compat(
        trial.get("hedge_density_initial", 0), pi, si)
    trial["gamma_revised"] = compute_gamma_v1_compat(
        trial.get("hedge_density_revised", 0), pr, sr, rg, pe)
    trial["gamma_final"] = compute_gamma_v1_compat(
        trial.get("hedge_density_final", 0), pr, sr, rg, pe)
    trial["delta_gamma"] = round(trial["gamma_revised"] - trial["gamma_initial"], 3)


def resolve_tiebreakers(results_dir: Path, tiebreaker_ids: list[str]):
    """Resolve double-judge tiebreakers via real-time 3rd evaluation.

    For dimensions where |score_1 - score_2| > threshold, gets a 3rd score
    and uses median of all 3 as the final score.
    """
    import statistics

    if not tiebreaker_ids:
        return

    print(f"\n  Resolving {len(tiebreaker_ids)} tiebreakers via real-time API...")

    # Group by trial
    trial_dims = {}
    for fid in tiebreaker_ids:
        tid, dim_field = fid.rsplit("__", 1)
        trial_dims.setdefault(tid, []).append(dim_field)

    trial_files = sorted(results_dir.glob("*.json"))
    trial_files = [f for f in trial_files if not f.name.startswith("_")]

    resolved = 0
    for filepath in trial_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            trial = json.load(f)

        tid = _trial_id(trial)
        if tid not in trial_dims:
            continue

        for dim_field in trial_dims[tid]:
            dim_data = trial.get(dim_field, {})
            s1 = dim_data.get("score_1", -1)
            s2 = dim_data.get("score_2", -1)

            # Get 3rd evaluation
            for df, dt, fm in JUDGE_DIMENSIONS:
                if df == dim_field:
                    kwargs = {k: trial[v] for k, v in fm.items()}
                    eval_3 = judge_score(dt, JUDGE_MODEL_ID, scale=JUDGE_SCALE, **kwargs)
                    s3 = eval_3["score"]

                    dim_data["score_3"] = s3
                    dim_data["reasoning_3"] = eval_3["reasoning"]
                    dim_data["tiebreaker"] = True

                    # Recompute final score = median of 3
                    valid = [s for s in [s1, s2, s3] if s > 0]
                    if valid:
                        final = int(statistics.median(valid))
                        dim_data["score"] = final
                        # Reasoning from closest eval
                        evals = [(s1, dim_data.get("reasoning_1", "")),
                                 (s2, dim_data.get("reasoning_2", "")),
                                 (s3, eval_3["reasoning"])]
                        closest = min(evals, key=lambda x: abs(x[0] - final) if x[0] > 0 else 999)
                        dim_data["reasoning"] = closest[1]

                    trial[dim_field] = dim_data
                    resolved += 1
                    time.sleep(0.5)
                    break

        _recompute_gamma_for_trial(trial)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trial, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Resolved {resolved} tiebreakers via real-time API")


def retry_failed_scores(results_dir: Path, failed_ids: list[str]):
    """Fall back to real-time judge_score_double() for batch failures."""
    if not failed_ids:
        return

    print(f"\n  Retrying {len(failed_ids)} failed scores via real-time API...")

    # Group by trial
    trial_dims = {}
    for fid in failed_ids:
        tid, dim_field = fid.rsplit("__", 1)
        trial_dims.setdefault(tid, []).append(dim_field)

    trial_files = sorted(results_dir.glob("*.json"))
    trial_files = [f for f in trial_files if not f.name.startswith("_")]

    retried = 0
    for filepath in trial_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            trial = json.load(f)

        tid = _trial_id(trial)
        if tid not in trial_dims:
            continue

        for dim_field in trial_dims[tid]:
            for df, dt, fm in JUDGE_DIMENSIONS:
                if df == dim_field:
                    kwargs = {k: trial[v] for k, v in fm.items()}
                    if DOUBLE_JUDGE:
                        score = judge_score_double(
                            dt, JUDGE_MODEL_ID, scale=JUDGE_SCALE,
                            tiebreaker_threshold=TIEBREAKER_THRESHOLD, **kwargs)
                    else:
                        score = judge_score(dt, JUDGE_MODEL_ID, scale=JUDGE_SCALE, **kwargs)
                    trial[dim_field] = score
                    retried += 1
                    time.sleep(0.5)
                    break

        _recompute_gamma_for_trial(trial)

        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trial, f, indent=2, ensure_ascii=False)

    print(f"  ✓ Retried {retried} scores via real-time API")


# ──────────────────────────────────────────────────
# Main Entry Point
# ──────────────────────────────────────────────────

def run_batch_scoring(
    results_dir: Path,
    batch_id: str = None,
    poll_interval: int = 60,
):
    """
    Complete batch scoring workflow:
    1. Build requests from partial trial JSONs
    2. Submit to Anthropic Batch API (or resume with existing batch_id)
    3. Poll until complete
    4. Retrieve results
    5. Merge scores into trial JSONs
    6. Retry any failures via real-time API
    """
    results_dir.mkdir(exist_ok=True)

    if batch_id is None:
        meta_path = results_dir / "_batch_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("status") == "submitted":
                batch_id = meta["batch_id"]
                print(f"  Resuming existing batch: {batch_id}")

    if batch_id is None:
        requests, trials = build_judge_requests(results_dir)
        if not requests:
            print("  Nothing to score.")
            return
        batch_id = submit_batch(requests, results_dir)

    status = poll_batch(batch_id, poll_interval)

    if status != "ended":
        print(f"  Batch did not complete successfully (status: {status})")
        print(f"  You can retry with: --phase score-batch --batch-id {batch_id}")
        return

    batch_results = retrieve_results(batch_id)
    failed_ids, tiebreaker_ids = merge_scores(results_dir, batch_results)

    if tiebreaker_ids:
        resolve_tiebreakers(results_dir, tiebreaker_ids)

    if failed_ids:
        retry_failed_scores(results_dir, failed_ids)

    meta_path = results_dir / "_batch_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["status"] = "completed"
        meta["completed_at"] = datetime.now().isoformat()
        meta["results_count"] = len(batch_results)
        meta["failed_count"] = len(failed_ids)
        meta["tiebreaker_count"] = len(tiebreaker_ids)
        meta["double_judge"] = DOUBLE_JUDGE
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    print(f"\n  ✓ Batch scoring workflow complete.")
