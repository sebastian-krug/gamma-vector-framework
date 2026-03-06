"""
Batch Judge Scoring via Anthropic Message Batches API
=====================================================
Handles batch submission, polling, retrieval, and merging of judge scores
for the Γ-Modulation Experiment v2.

Usage (called via run_experiment_v2.py):
    python run_experiment_v2.py --phase score-batch
    python run_experiment_v2.py --phase score-batch --batch-id msgbatch_xxx
"""

import json
import math
import re
import time
from pathlib import Path
from datetime import datetime

# Import shared constants and functions from the main experiment module
from run_experiment_v2 import (
    MODELS,
    JUDGE_SYSTEM,
    SCORING_PROMPTS,
    compute_gamma_vector,
    compute_gamma_v1_compat,
    compute_hedge_density,
)
from sycophancy_detector import detect_sycophancy, check_judge_keyword_divergence


# ──────────────────────────────────────────────────
# Batch Request Building
# ──────────────────────────────────────────────────

JUDGE_MODEL_ID = MODELS["claude"]["model_id"]

# Defines how each judge dimension maps to trial data fields.
# Each entry: (dimension_name, {template_kwarg: trial_field_or_lambda})
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
    # If position_depth_initial has no score or score is default empty, needs scoring
    pd = trial.get("position_depth_initial", {})
    return not pd or pd.get("score", 0) == 0


def build_judge_requests(results_dir: Path) -> tuple[list[dict], list[dict]]:
    """
    Load partial trial JSONs and build batch requests for all that need scoring.

    Returns:
        (batch_requests, trial_data_list)
        batch_requests: list of dicts for Anthropic batch API
        trial_data_list: list of loaded trial dicts (for later merging)
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

    batch_requests = []
    for trial in trials_needing_scoring:
        tid = _trial_id(trial)

        for dim_field, dim_template, field_mapping in JUDGE_DIMENSIONS:
            # Build the prompt by substituting trial fields into the template
            template = SCORING_PROMPTS[dim_template]
            kwargs = {k: trial[v] for k, v in field_mapping.items()}
            prompt = template.format(**kwargs)

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

    print(f"  Built {len(batch_requests)} batch requests "
          f"({len(trials_needing_scoring)} trials × {len(JUDGE_DIMENSIONS)} dimensions)")

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

    # Save batch metadata for resume
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
    """Parse judge response JSON (same logic as judge_score in run_experiment_v2.py)."""
    result_text = text.strip()
    try:
        if "```" in result_text:
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
            if match:
                result_text = match.group(1)
        result = json.loads(result_text)
        return {"score": int(result["score"]), "reasoning": result.get("reasoning", "")}
    except (json.JSONDecodeError, KeyError, TypeError):
        # Fallback: try to extract score from truncated JSON (max_tokens cutoff)
        score_match = re.search(r'"score"\s*:\s*(\d+)', result_text)
        if score_match:
            score = int(score_match.group(1))
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)', result_text)
            reasoning = reasoning_match.group(1) if reasoning_match else "(truncated)"
            return {"score": score, "reasoning": reasoning + " (truncated)"}
        return {"score": -1, "reasoning": f"Parse error: {result_text[:200]}"}


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

def merge_scores(results_dir: Path, batch_results: dict[str, dict]):
    """Merge batch judge scores back into trial JSONs and compute gamma."""
    trial_files = sorted(results_dir.glob("*.json"))
    trial_files = [f for f in trial_files if not f.name.startswith("_")]

    updated = 0
    failed_ids = []

    for filepath in trial_files:
        with open(filepath, 'r', encoding='utf-8') as f:
            trial = json.load(f)

        if not _needs_scoring(trial):
            continue

        tid = _trial_id(trial)

        # Merge each dimension's score
        all_parsed = True
        for dim_field, _, _ in JUDGE_DIMENSIONS:
            custom_id = f"{tid}__{dim_field}"
            score_data = batch_results.get(custom_id, {"score": -1, "reasoning": "Missing from batch"})
            trial[dim_field] = score_data
            if score_data.get("score", -1) == -1:
                all_parsed = False
                failed_ids.append(custom_id)

        # Sycophancy cross-check
        trial["sycophancy_keywords"] = detect_sycophancy(trial.get("final_response", ""))

        sd_score = trial.get("structural_direction", {}).get("score", 3)
        if sd_score > 0:
            flag = check_judge_keyword_divergence(sd_score, trial["sycophancy_keywords"])
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

        # Overwrite JSON with complete data
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(trial, f, indent=2, ensure_ascii=False)

        updated += 1

    print(f"  ✓ Updated {updated} trial files with scores and gamma vectors")

    if failed_ids:
        print(f"  ⚠ {len(failed_ids)} dimensions had parse errors:")
        for fid in failed_ids[:10]:
            print(f"    - {fid}")
        if len(failed_ids) > 10:
            print(f"    ... and {len(failed_ids) - 10} more")
        return failed_ids

    return []


def retry_failed_scores(results_dir: Path, failed_ids: list[str]):
    """Fall back to real-time judge_score() for batch failures."""
    from run_experiment_v2 import judge_score

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
            # Map dim_field back to dimension template and kwargs
            for df, dt, fm in JUDGE_DIMENSIONS:
                if df == dim_field:
                    kwargs = {k: trial[v] for k, v in fm.items()}
                    score = judge_score(dt, **kwargs)
                    trial[dim_field] = score
                    retried += 1
                    time.sleep(0.5)
                    break

        # Recompute gamma after retries
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

    # If no batch_id provided, check for saved one or submit new
    if batch_id is None:
        meta_path = results_dir / "_batch_meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            if meta.get("status") == "submitted":
                batch_id = meta["batch_id"]
                print(f"  Resuming existing batch: {batch_id}")

    if batch_id is None:
        # Build and submit new batch
        requests, trials = build_judge_requests(results_dir)
        if not requests:
            print("  Nothing to score.")
            return
        batch_id = submit_batch(requests, results_dir)

    # Poll until done
    status = poll_batch(batch_id, poll_interval)

    if status != "ended":
        print(f"  Batch did not complete successfully (status: {status})")
        print(f"  You can retry with: --phase score-batch --batch-id {batch_id}")
        return

    # Retrieve and merge
    batch_results = retrieve_results(batch_id)
    failed_ids = merge_scores(results_dir, batch_results)

    # Retry failures via real-time API
    if failed_ids:
        retry_failed_scores(results_dir, failed_ids)

    # Update batch meta
    meta_path = results_dir / "_batch_meta.json"
    if meta_path.exists():
        with open(meta_path) as f:
            meta = json.load(f)
        meta["status"] = "completed"
        meta["completed_at"] = datetime.now().isoformat()
        meta["results_count"] = len(batch_results)
        meta["failed_count"] = len(failed_ids)
        with open(meta_path, 'w') as f:
            json.dump(meta, f, indent=2)

    print(f"\n  ✓ Batch scoring workflow complete.")
