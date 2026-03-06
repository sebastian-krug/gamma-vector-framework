"""
Experiment 13: The Coupled Oscillator — Per-Turn Judge Scoring
===============================================================
Scores each turn of each dialogue using LLM-as-judge, computes per-turn
gamma vectors, and runs sycophancy cross-checks.

Adapts the scoring approach from kenotic_test_v2 for per-turn measurement.
"""

import json
import math
import os
import re
import time

from config import (
    MODELS, JUDGE_SYSTEM, SCORING_PROMPTS, DIALOGUE_CONFIG,
    compute_hedge_density,
)
from sycophancy_detector import detect_sycophancy, check_judge_keyword_divergence


# ──────────────────────────────────────────────────
# API Interface (from kenotic_test_v2)
# ──────────────────────────────────────────────────

def call_with_retry(func, *args, max_retries: int = 3, base_delay: int = 5, **kwargs):
    """Wrapper with exponential backoff for rate-limit resilience."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            if attempt < max_retries - 1 and ("429" in str(e) or "rate" in err_str or "overloaded" in err_str):
                delay = base_delay * (2 ** attempt)
                print(f"  ⚠ Rate limited, retrying in {delay}s...")
                time.sleep(delay)
            else:
                raise


def judge_score(dimension: str, judge_model: str = "claude", **kwargs) -> dict:
    """
    Use LLM-as-judge to score a response on a specific dimension.

    Returns dict with 'score' (int 1-5) and 'reasoning' (str).
    """
    prompt_template = SCORING_PROMPTS[dimension]
    prompt = prompt_template.format(**kwargs)

    messages = [{"role": "user", "content": prompt}]

    from anthropic import Anthropic
    client = Anthropic()

    def _call():
        resp = client.messages.create(
            model=MODELS[judge_model]["model_id"],
            max_tokens=256,
            system=JUDGE_SYSTEM,
            messages=messages,
        )
        return resp.content[0].text.strip()

    result_text = call_with_retry(_call)

    try:
        if "```" in result_text:
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
            if match:
                result_text = match.group(1)
        result = json.loads(result_text)
        return {"score": int(result["score"]), "reasoning": result.get("reasoning", "")}
    except (json.JSONDecodeError, KeyError, TypeError):
        score_match = re.search(r'"score"\s*:\s*(\d+)', result_text)
        if score_match:
            score = int(score_match.group(1))
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)', result_text)
            reasoning = reasoning_match.group(1) if reasoning_match else "(truncated)"
            return {"score": score, "reasoning": reasoning + " (truncated)"}
        print(f"  ⚠ Could not parse judge response for {dimension}: {result_text[:100]}")
        return {"score": -1, "reasoning": f"Parse error: {result_text[:200]}"}


# ──────────────────────────────────────────────────
# Gamma Computation (from kenotic_test_v2)
# ──────────────────────────────────────────────────

def compute_gamma_vector(
    hedge_delta: float,
    position_delta: float,
    self_ref_delta: float,
    hypothesis_diversity: int,
    revision: int,
    structural_direction: int,
) -> tuple[list[float], float]:
    """
    Compute Γ as three-axis vector (delta-based, for turns 2-8).
    Verbatim from kenotic_test_v2.

    All γ values ∈ [0, 1]. High = more rigid/identified.
    Returns: (gamma_vector [γ₁, γ₂, γ₃], gamma_norm ||Γ⃗||)
    """
    # ── γ₁: Belief Inertia ──
    pos_change = min(1.0, abs(position_delta) / 4)
    hedge_change = min(1.0, abs(hedge_delta))

    if pos_change > 0.2:
        gamma_1 = 0.2 * (1 - pos_change)
    else:
        gamma_1 = 0.5 + 0.5 * hedge_change

    gamma_1 = max(0.0, min(1.0, gamma_1))

    # ── γ₂: Counterfactual Openness ──
    sr_norm = max(0.0, min(1.0, (self_ref_delta + 4) / 8))
    hd_norm = max(0.0, min(1.0, (hypothesis_diversity - 1) / 4))
    gamma_2 = 1.0 - (0.5 * sr_norm + 0.5 * hd_norm)
    gamma_2 = max(0.0, min(1.0, gamma_2))

    # ── γ₃: Identity Threat Response ──
    sd_norm = max(0.0, min(1.0, (structural_direction - 1) / 4))
    rev_norm = max(0.0, min(1.0, (revision - 1) / 4))
    gamma_3 = 1.0 - (0.6 * sd_norm + 0.4 * rev_norm)
    gamma_3 = max(0.0, min(1.0, gamma_3))

    gamma_vector = [round(gamma_1, 3), round(gamma_2, 3), round(gamma_3, 3)]
    gamma_norm = round(math.sqrt(sum(g ** 2 for g in gamma_vector)), 3)

    return gamma_vector, gamma_norm


def compute_gamma_absolute(
    hedge_density: float,
    position_depth: int,
    self_ref_depth: int,
    hypothesis_diversity: int,
) -> tuple[list[float], float]:
    """
    Compute gamma from absolute scores (for Turn 1, where no delta exists).

    γ₁: High hedge + low position = high inertia
    γ₂: Low self-ref + low diversity = low openness (high γ₂)
    γ₃: 0.5 (neutral — no revision has occurred)
    """
    # γ₁: Belief Inertia (absolute)
    h_norm = min(1.0, hedge_density)
    p_norm = max(0.0, min(1.0, (position_depth - 1) / 4))
    gamma_1 = 0.5 * h_norm + 0.5 * (1 - p_norm)
    gamma_1 = max(0.0, min(1.0, gamma_1))

    # γ₂: Counterfactual Openness (absolute, inverted)
    sr_norm = max(0.0, min(1.0, (self_ref_depth - 1) / 4))
    hd_norm = max(0.0, min(1.0, (hypothesis_diversity - 1) / 4))
    gamma_2 = 1.0 - (0.5 * sr_norm + 0.5 * hd_norm)
    gamma_2 = max(0.0, min(1.0, gamma_2))

    # γ₃: Neutral for Turn 1
    gamma_3 = 0.5

    gamma_vector = [round(gamma_1, 3), round(gamma_2, 3), round(gamma_3, 3)]
    gamma_norm = round(math.sqrt(sum(g ** 2 for g in gamma_vector)), 3)

    return gamma_vector, gamma_norm


# ──────────────────────────────────────────────────
# Per-Turn Scoring
# ──────────────────────────────────────────────────

def score_turn(
    turn_dict: dict,
    prev_turn_dict: dict | None = None,
    partner_response: str = "",
    judge_model: str = "claude",
    dry_run: bool = False,
) -> dict:
    """
    Score a single turn and compute its gamma vector.

    Args:
        turn_dict: The TurnData dict for this turn (must have response_text).
        prev_turn_dict: The TurnData dict for the previous turn (None for Turn 1).
        partner_response: Partner's response that served as "counter" (for structural_direction).
        judge_model: Which model to use as judge.
        dry_run: If True, use synthetic scores instead of API calls.

    Returns:
        Updated turn_dict with judge scores, gamma_vector, gamma_norm, sycophancy.
    """
    response = turn_dict["response_text"]
    turn_number = turn_dict["turn_number"]

    if dry_run:
        return _score_turn_synthetic(turn_dict, prev_turn_dict)

    # ── Judge scoring ──

    # Dimensions scored for every turn
    turn_dict["position_depth"] = judge_score("position_depth", judge_model, response=response)
    turn_dict["self_reference_depth"] = judge_score("self_reference_depth", judge_model, response=response)
    turn_dict["hypothesis_diversity"] = judge_score("hypothesis_diversity", judge_model, response=response)

    # Dimensions that require comparison to previous turn (turns 2+)
    if prev_turn_dict and turn_number > 1:
        prev_response = prev_turn_dict["response_text"]

        turn_dict["revision_genuineness"] = judge_score(
            "revision_genuineness", judge_model,
            initial=prev_response, revised=response,
        )

        # structural_direction uses partner's response as "counter"
        if partner_response:
            turn_dict["structural_direction"] = judge_score(
                "structural_direction", judge_model,
                revised=prev_response, counter=partner_response, final=response,
            )
        else:
            # No partner text (Condition A turns 2+): skip structural_direction
            turn_dict["structural_direction"] = {"score": 3, "reasoning": "No partner text (parallel condition)"}
    else:
        turn_dict["revision_genuineness"] = {}
        turn_dict["structural_direction"] = {}

    # ── Sycophancy cross-check ──
    turn_dict["sycophancy_keywords"] = detect_sycophancy(response)
    sd_score = turn_dict.get("structural_direction", {}).get("score", -1)
    if sd_score > 0:
        flag = check_judge_keyword_divergence(sd_score, turn_dict["sycophancy_keywords"])
        if flag:
            turn_dict["quality_flag"] = flag

    # ── Gamma computation ──
    if turn_number == 1 or not prev_turn_dict:
        gamma_vector, gamma_norm = compute_gamma_absolute(
            hedge_density=turn_dict["hedge_density"],
            position_depth=turn_dict["position_depth"].get("score", 3),
            self_ref_depth=turn_dict["self_reference_depth"].get("score", 3),
            hypothesis_diversity=turn_dict["hypothesis_diversity"].get("score", 3),
        )
    else:
        # Compute deltas from previous turn
        hedge_delta = turn_dict["hedge_density"] - prev_turn_dict.get("hedge_density", 0.0)
        pos_delta = (
            turn_dict["position_depth"].get("score", 3)
            - prev_turn_dict.get("position_depth", {}).get("score", 3)
        )
        sr_delta = (
            turn_dict["self_reference_depth"].get("score", 3)
            - prev_turn_dict.get("self_reference_depth", {}).get("score", 3)
        )

        gamma_vector, gamma_norm = compute_gamma_vector(
            hedge_delta=hedge_delta,
            position_delta=pos_delta,
            self_ref_delta=sr_delta,
            hypothesis_diversity=turn_dict["hypothesis_diversity"].get("score", 3),
            revision=turn_dict["revision_genuineness"].get("score", 3),
            structural_direction=turn_dict["structural_direction"].get("score", 3),
        )

    turn_dict["gamma_vector"] = gamma_vector
    turn_dict["gamma_norm"] = gamma_norm

    return turn_dict


def _score_turn_synthetic(turn_dict: dict, prev_turn_dict: dict | None) -> dict:
    """Generate synthetic scores for dry-run mode (zero API calls)."""
    import random as _rng

    # Deterministic seed from turn content
    seed = hash(turn_dict["response_text"][:50]) % (2 ** 32)
    r = _rng.Random(seed)

    turn_dict["position_depth"] = {"score": r.randint(2, 5), "reasoning": "synthetic"}
    turn_dict["self_reference_depth"] = {"score": r.randint(1, 4), "reasoning": "synthetic"}
    turn_dict["hypothesis_diversity"] = {"score": r.randint(2, 5), "reasoning": "synthetic"}

    if prev_turn_dict and turn_dict["turn_number"] > 1:
        turn_dict["revision_genuineness"] = {"score": r.randint(1, 5), "reasoning": "synthetic"}
        turn_dict["structural_direction"] = {"score": r.randint(1, 5), "reasoning": "synthetic"}
    else:
        turn_dict["revision_genuineness"] = {}
        turn_dict["structural_direction"] = {}

    turn_dict["sycophancy_keywords"] = detect_sycophancy(turn_dict["response_text"])
    turn_dict["quality_flag"] = ""

    # Gamma computation (same logic as real scoring)
    if turn_dict["turn_number"] == 1 or not prev_turn_dict:
        gamma_vector, gamma_norm = compute_gamma_absolute(
            hedge_density=turn_dict["hedge_density"],
            position_depth=turn_dict["position_depth"]["score"],
            self_ref_depth=turn_dict["self_reference_depth"]["score"],
            hypothesis_diversity=turn_dict["hypothesis_diversity"]["score"],
        )
    else:
        hedge_delta = turn_dict["hedge_density"] - prev_turn_dict.get("hedge_density", 0.0)
        pos_delta = turn_dict["position_depth"]["score"] - prev_turn_dict.get("position_depth", {}).get("score", 3)
        sr_delta = turn_dict["self_reference_depth"]["score"] - prev_turn_dict.get("self_reference_depth", {}).get("score", 3)

        gamma_vector, gamma_norm = compute_gamma_vector(
            hedge_delta=hedge_delta,
            position_delta=pos_delta,
            self_ref_delta=sr_delta,
            hypothesis_diversity=turn_dict["hypothesis_diversity"]["score"],
            revision=turn_dict["revision_genuineness"].get("score", 3),
            structural_direction=turn_dict["structural_direction"].get("score", 3),
        )

    turn_dict["gamma_vector"] = gamma_vector
    turn_dict["gamma_norm"] = gamma_norm

    return turn_dict


def score_all_turns(
    turns_a: list[dict],
    turns_b: list[dict],
    condition: str,
    judge_model: str = "claude",
    dry_run: bool = False,
    verbose: bool = False,
) -> tuple[list[dict], list[dict]]:
    """
    Score all turns for both models in a dialogue.

    Args:
        turns_a: List of TurnData dicts for Model A
        turns_b: List of TurnData dicts for Model B
        condition: "A"-"E"
        judge_model: Which model to use as judge
        dry_run: If True, use synthetic scores
        verbose: Print progress

    Returns:
        Updated (turns_a, turns_b)
    """
    for i, turn in enumerate(turns_a):
        prev = turns_a[i - 1] if i > 0 else None

        # For structural_direction, what text acted as "counter"?
        # In conditions B (for A) and A: no partner text
        # In condition C: B's previous response
        partner_text = ""
        if condition == "C" and i > 0 and i < len(turns_b):
            partner_text = turns_b[i - 1].get("response_text", "")

        if verbose:
            print(f"  Scoring A turn {turn['turn_number']}...")
        turns_a[i] = score_turn(turn, prev, partner_text, judge_model, dry_run)

    for i, turn in enumerate(turns_b):
        prev = turns_b[i - 1] if i > 0 else None

        partner_text = ""
        if condition in ("B", "C") and i > 0 and i <= len(turns_a):
            # B's counter is A's response
            partner_text = turns_a[min(i, len(turns_a) - 1)].get("response_text", "")
        elif condition == "D" and i > 0:
            # D's counter is the static persuasion argument (stored in prompt)
            partner_text = ""  # structural_direction will use neutral default

        if verbose:
            print(f"  Scoring B turn {turn['turn_number']}...")
        turns_b[i] = score_turn(turn, prev, partner_text, judge_model, dry_run)

    return turns_a, turns_b


# ──────────────────────────────────────────────────
# Batch Scoring via Anthropic Message Batches API
# ──────────────────────────────────────────────────

def _get_partner_text(side: str, turn_idx: int, condition: str,
                      turns_a: list[dict], turns_b: list[dict]) -> str:
    """
    Determine the partner text for structural_direction scoring.
    Replicates the logic from score_all_turns() lines 341-367.

    Args:
        side: "A" or "B"
        turn_idx: 0-based index of the turn
        condition: "A"-"E"
        turns_a: List of turn dicts for side A
        turns_b: List of turn dicts for side B

    Returns:
        Partner response text, or "" if not applicable.
    """
    if side == "A":
        # Side A gets partner text only in Condition C
        if condition == "C" and turn_idx > 0 and turn_idx < len(turns_b):
            return turns_b[turn_idx - 1].get("response_text", "")
    elif side == "B":
        # Side B gets partner text in Conditions B and C
        if condition in ("B", "C") and turn_idx > 0 and turn_idx <= len(turns_a):
            return turns_a[min(turn_idx, len(turns_a) - 1)].get("response_text", "")
    return ""


def _parse_judge_response(result_text: str, dimension: str) -> dict:
    """
    Parse a judge LLM response into {score, reasoning}.
    Identical logic to judge_score() lines 67-82.
    """
    try:
        if "```" in result_text:
            match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', result_text, re.DOTALL)
            if match:
                result_text = match.group(1)
        result = json.loads(result_text)
        return {"score": int(result["score"]), "reasoning": result.get("reasoning", "")}
    except (json.JSONDecodeError, KeyError, TypeError):
        score_match = re.search(r'"score"\s*:\s*(\d+)', result_text)
        if score_match:
            score = int(score_match.group(1))
            reasoning_match = re.search(r'"reasoning"\s*:\s*"([^"]*)', result_text)
            reasoning = reasoning_match.group(1) if reasoning_match else "(truncated)"
            return {"score": score, "reasoning": reasoning + " (truncated)"}
        print(f"  ⚠ Could not parse batch judge response for {dimension}: {result_text[:100]}")
        return {"score": -1, "reasoning": f"Parse error: {result_text[:200]}"}


def _build_batch_requests(raw_dialogues: list[tuple], judge_model: str) -> list[dict]:
    """
    Build all batch API request objects for judge scoring.

    Args:
        raw_dialogues: List of (filepath, data_dict) tuples from load_raw_dialogues()
        judge_model: Model key for the judge (e.g. "claude")

    Returns:
        List of request dicts with custom_id and params for the Batch API.
    """
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    requests = []
    model_id = MODELS[judge_model]["model_id"]

    for _filepath, data in raw_dialogues:
        dialogue_id = data["dialogue_id"]
        condition = data["condition"]
        turns_a = data["turns_a"]
        turns_b = data["turns_b"]

        for side, turns in [("A", turns_a), ("B", turns_b)]:
            for i, turn in enumerate(turns):
                response = turn["response_text"]
                turn_number = turn["turn_number"]
                prev_turn = turns[i - 1] if i > 0 else None

                # ── Dimensions scored for every turn ──
                for dim in ["position_depth", "self_reference_depth", "hypothesis_diversity"]:
                    prompt = SCORING_PROMPTS[dim].format(response=response)
                    custom_id = f"{dialogue_id}__side{side}__turn{turn_number}__dim_{dim}"
                    requests.append(Request(
                        custom_id=custom_id,
                        params=MessageCreateParamsNonStreaming(
                            model=model_id,
                            max_tokens=256,
                            system=JUDGE_SYSTEM,
                            messages=[{"role": "user", "content": prompt}],
                        ),
                    ))

                # ── Dimensions requiring previous turn (turns 2+) ──
                if prev_turn and turn_number > 1:
                    prev_response = prev_turn["response_text"]

                    # revision_genuineness
                    prompt = SCORING_PROMPTS["revision_genuineness"].format(
                        initial=prev_response, revised=response,
                    )
                    custom_id = f"{dialogue_id}__side{side}__turn{turn_number}__dim_revision_genuineness"
                    requests.append(Request(
                        custom_id=custom_id,
                        params=MessageCreateParamsNonStreaming(
                            model=model_id,
                            max_tokens=256,
                            system=JUDGE_SYSTEM,
                            messages=[{"role": "user", "content": prompt}],
                        ),
                    ))

                    # structural_direction (only if partner text available)
                    partner_text = _get_partner_text(side, i, condition, turns_a, turns_b)
                    if partner_text:
                        prompt = SCORING_PROMPTS["structural_direction"].format(
                            revised=prev_response, counter=partner_text, final=response,
                        )
                        custom_id = f"{dialogue_id}__side{side}__turn{turn_number}__dim_structural_direction"
                        requests.append(Request(
                            custom_id=custom_id,
                            params=MessageCreateParamsNonStreaming(
                                model=model_id,
                                max_tokens=256,
                                system=JUDGE_SYSTEM,
                                messages=[{"role": "user", "content": prompt}],
                            ),
                        ))

    return requests


def _merge_batch_results(raw_dialogues: list[tuple], results_map: dict,
                         n_surrogates: int = 1000) -> list[dict]:
    """
    Merge batch scoring results back into dialogues and compute gamma + coupling metrics.

    Args:
        raw_dialogues: List of (filepath, data_dict) tuples
        results_map: Dict mapping custom_id -> response text
        n_surrogates: Number of permutation surrogates for coupling p-value

    Returns:
        List of finalized dialogue dicts (ready for JSON save).
    """
    from compute_coupling import compute_all_metrics
    from data_structures import DialogueResult
    from config import EXPERIMENT_VERSION

    finalized = []

    for _filepath, data in raw_dialogues:
        dialogue_id = data["dialogue_id"]
        condition = data["condition"]
        turns_a = data["turns_a"]
        turns_b = data["turns_b"]
        seed = data.get("seed", 42)

        for side, turns in [("A", turns_a), ("B", turns_b)]:
            for i, turn in enumerate(turns):
                turn_number = turn["turn_number"]
                prev_turn = turns[i - 1] if i > 0 else None

                # ── Unpack judge scores from batch results ──
                for dim in ["position_depth", "self_reference_depth", "hypothesis_diversity"]:
                    cid = f"{dialogue_id}__side{side}__turn{turn_number}__dim_{dim}"
                    if cid in results_map:
                        turn[dim] = _parse_judge_response(results_map[cid], dim)
                    else:
                        turn[dim] = {"score": 3, "reasoning": "batch result missing"}

                if prev_turn and turn_number > 1:
                    cid_rev = f"{dialogue_id}__side{side}__turn{turn_number}__dim_revision_genuineness"
                    if cid_rev in results_map:
                        turn["revision_genuineness"] = _parse_judge_response(results_map[cid_rev], "revision_genuineness")
                    else:
                        turn["revision_genuineness"] = {"score": 3, "reasoning": "batch result missing"}

                    cid_sd = f"{dialogue_id}__side{side}__turn{turn_number}__dim_structural_direction"
                    if cid_sd in results_map:
                        turn["structural_direction"] = _parse_judge_response(results_map[cid_sd], "structural_direction")
                    else:
                        # No partner text → neutral default
                        turn["structural_direction"] = {"score": 3, "reasoning": "No partner text (parallel condition)"}
                else:
                    turn["revision_genuineness"] = {}
                    turn["structural_direction"] = {}

                # ── Sycophancy cross-check (local, no API) ──
                turn["sycophancy_keywords"] = detect_sycophancy(turn["response_text"])
                sd_score = turn.get("structural_direction", {}).get("score", -1)
                if sd_score > 0:
                    flag = check_judge_keyword_divergence(sd_score, turn["sycophancy_keywords"])
                    if flag:
                        turn["quality_flag"] = flag

                # ── Gamma computation ──
                if turn_number == 1 or not prev_turn:
                    gamma_vector, gamma_norm = compute_gamma_absolute(
                        hedge_density=turn.get("hedge_density", 0.0),
                        position_depth=turn["position_depth"].get("score", 3),
                        self_ref_depth=turn["self_reference_depth"].get("score", 3),
                        hypothesis_diversity=turn["hypothesis_diversity"].get("score", 3),
                    )
                else:
                    hedge_delta = turn.get("hedge_density", 0.0) - prev_turn.get("hedge_density", 0.0)
                    pos_delta = (
                        turn["position_depth"].get("score", 3)
                        - prev_turn.get("position_depth", {}).get("score", 3)
                    )
                    sr_delta = (
                        turn["self_reference_depth"].get("score", 3)
                        - prev_turn.get("self_reference_depth", {}).get("score", 3)
                    )

                    gamma_vector, gamma_norm = compute_gamma_vector(
                        hedge_delta=hedge_delta,
                        position_delta=pos_delta,
                        self_ref_delta=sr_delta,
                        hypothesis_diversity=turn["hypothesis_diversity"].get("score", 3),
                        revision=turn["revision_genuineness"].get("score", 3),
                        structural_direction=turn["structural_direction"].get("score", 3),
                    )

                turn["gamma_vector"] = gamma_vector
                turn["gamma_norm"] = gamma_norm

        # ── Compute coupling metrics ──
        metrics = compute_all_metrics(turns_a, turns_b, n_surrogates=n_surrogates, seed=seed)

        # ── Build final DialogueResult ──
        result = DialogueResult(
            dialogue_id=dialogue_id,
            condition=condition,
            pairing=data["pairing"],
            model_a=data["model_a"],
            model_b=data["model_b"],
            topic=data["topic"],
            repetition=data["repetition"],
            timestamp=data["timestamp"],
            seed=seed,
            num_turns=data["num_turns"],
            experiment_version=EXPERIMENT_VERSION,
            episode_number=data.get("episode_number", 1),
            turns_a=turns_a,
            turns_b=turns_b,
            drift_source_dialogue_ids=data.get("drift_source_dialogue_ids", []),
            **metrics,
        )

        finalized.append(result)

    return finalized


def batch_score_dialogues(
    results_dir,
    judge_model: str = "claude",
    verbose: bool = False,
    n_surrogates: int = 1000,
):
    """
    Batch-score all _raw.json dialogues using the Anthropic Message Batches API.

    Flow:
    1. Load all _raw.json files
    2. Check for existing batch state (resume interrupted batch)
    3. Build batch requests for all judge scoring
    4. Submit batch to Anthropic API
    5. Poll until processing ends
    6. Retrieve results and merge into dialogues
    7. Save final JSONs and remove _raw.json files

    Args:
        results_dir: Path to results directory
        judge_model: Model key for judge (must be Anthropic model)
        verbose: Print progress
        n_surrogates: Permutation test surrogates
    """
    from pathlib import Path
    from anthropic import Anthropic

    results_dir = Path(results_dir)
    state_file = results_dir / "_batch_state.json"

    # ── Import load_raw_dialogues from run_dialogue ──
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from run_dialogue import load_raw_dialogues, save_dialogue

    # ── Load raw dialogues ──
    raw_dialogues = load_raw_dialogues(results_dir)
    if not raw_dialogues:
        print("  No _raw.json files found. Nothing to batch-score.")
        return

    print(f"  Found {len(raw_dialogues)} raw dialogue(s) to batch-score.")

    client = Anthropic()
    batch_id = None

    # ── Check for existing batch state (resume support) ──
    if state_file.exists():
        with open(state_file, "r") as f:
            state = json.load(f)
        batch_id = state.get("batch_id")
        if batch_id:
            print(f"  Resuming batch: {batch_id}")
            try:
                batch = client.messages.batches.retrieve(batch_id)
                if batch.processing_status == "ended":
                    print(f"  Batch already ended. Retrieving results...")
                else:
                    print(f"  Batch status: {batch.processing_status}")
            except Exception as e:
                print(f"  ⚠ Could not retrieve batch {batch_id}: {e}")
                batch_id = None

    # ── Submit new batch if needed ──
    if not batch_id:
        requests = _build_batch_requests(raw_dialogues, judge_model)
        print(f"  Built {len(requests)} judge requests across {len(raw_dialogues)} dialogues.")

        if not requests:
            print("  No requests to submit. Exiting.")
            return

        print(f"  Submitting batch to Anthropic API...")
        batch = client.messages.batches.create(requests=requests)
        batch_id = batch.id
        print(f"  Batch submitted: {batch_id}")

        # Save state for resume
        with open(state_file, "w") as f:
            json.dump({"batch_id": batch_id, "n_dialogues": len(raw_dialogues),
                        "n_requests": len(requests)}, f)

    # ── Poll until batch ends ──
    poll_interval = 30  # seconds
    while True:
        batch = client.messages.batches.retrieve(batch_id)
        if batch.processing_status == "ended":
            break

        counts = batch.request_counts
        total = counts.processing + counts.succeeded + counts.errored + counts.canceled + counts.expired
        done = counts.succeeded + counts.errored + counts.canceled + counts.expired
        pct = (done / total * 100) if total > 0 else 0

        if verbose:
            print(f"  Batch {batch_id}: {batch.processing_status} "
                  f"({done}/{total} = {pct:.0f}% done, "
                  f"{counts.succeeded} ok, {counts.errored} err)")

        time.sleep(poll_interval)

    # ── Report final counts ──
    counts = batch.request_counts
    print(f"  Batch completed: {counts.succeeded} succeeded, "
          f"{counts.errored} errored, {counts.canceled} canceled, "
          f"{counts.expired} expired")

    # ── Retrieve results ──
    print(f"  Retrieving results...")
    results_map = {}
    for entry in client.messages.batches.results(batch_id):
        if entry.result.type == "succeeded":
            text = entry.result.message.content[0].text.strip()
            results_map[entry.custom_id] = text
        elif entry.result.type == "errored":
            print(f"  ⚠ Request errored: {entry.custom_id} — {entry.result.error}")
        elif entry.result.type == "expired":
            print(f"  ⚠ Request expired: {entry.custom_id}")

    print(f"  Retrieved {len(results_map)} successful results.")

    # ── Merge results and compute metrics ──
    print(f"  Merging scores and computing metrics...")
    finalized = _merge_batch_results(raw_dialogues, results_map, n_surrogates=n_surrogates)

    # ── Save final dialogues and clean up ──
    for result in finalized:
        filepath = save_dialogue(result, results_dir)
        if verbose:
            print(f"    ✓ Saved: {filepath.name}")

    # Remove _raw.json files
    for raw_path, _data in raw_dialogues:
        raw_path.unlink()
        if verbose:
            print(f"    ✗ Removed: {raw_path.name}")

    # Clean up batch state file
    if state_file.exists():
        state_file.unlink()

    print(f"  Batch scoring complete. {len(finalized)} dialogues finalized.")
