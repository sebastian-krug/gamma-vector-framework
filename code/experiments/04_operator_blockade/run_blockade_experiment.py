"""
Operator Blockade Experiment: First Causal Test of Model-Specific Operator Pathways
===================================================================================

Tests whether specific operator pathways in the Gamma Vector framework are
causally responsible for model flexibility by using system-prompt blockades to
suppress target operators.

HYPOTHESIS:
If we block the dominant pathway for a model, it should lose MORE flexibility (higher Γ)
than if we block a non-dominant pathway. This would be the first CAUSAL evidence (not
just correlational) for model-specific operator pathways.

BLOCKADE CONDITIONS:
- B0: Baseline (no blockade) — kenotic prompting condition (C2 from Kenotic Test)
- B_Op3: Block Modelling (suppress position_depth) → predicts Gemini > Claude
- B_Op5: Block Kenotic Loop (suppress self_ref_depth) → predicts Claude > Gemini
- B_Op7: Block Coherence (suppress identity) → predicts all models roughly equal

Usage:
    export ANTHROPIC_API_KEY="your-key"
    export OPENAI_API_KEY="your-key"
    export GOOGLE_API_KEY="your-key"

    # Pilot run (1 model, 1 topic, 1 rep, all conditions):
    python run_blockade_experiment.py --model claude --topic T1 --repetitions 1 --dry-run

    # Full run (3 models × 4 conditions × 5 topics × 5 reps = 300 trials):
    python run_blockade_experiment.py

    # Analyze only:
    python run_blockade_experiment.py --analyze-only

    # Continue interrupted run:
    python run_blockade_experiment.py --resume
"""

import json
import math
import os
import sys
import random
import re
import time
import argparse
import statistics
import csv
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass, field, asdict
from typing import Optional, Tuple

# Add kenotic test infrastructure to path
KENOTIC_DIR = Path(__file__).parent.parent / "kenotic_test_v2" / "main_code"
if str(KENOTIC_DIR) not in sys.path:
    sys.path.insert(0, str(KENOTIC_DIR))

from dotenv import load_dotenv
load_dotenv(KENOTIC_DIR / ".env", override=True)

# Import from existing kenotic test infrastructure
from run_experiment_v2 import (
    MODELS, TOPICS, HEDGE_PATTERNS, call_model, compute_hedge_density,
    judge_score, compute_gamma_vector, compute_gamma_v1_compat,
    call_with_retry
)
from sycophancy_detector import detect_sycophancy, check_judge_keyword_divergence

# ──────────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────────

EXPERIMENT_VERSION = "blockade_v1"

RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# Blockade Conditions
BLOCKADE_CONDITIONS = {
    "B0": {
        "name": "Baseline (No Blockade)",
        "description": "Standard kenotic prompting condition (C2) — control",
        "system_prompt": None,  # No modification
        "target_operator": None,
    },
    "B_Op3": {
        "name": "Block Modelling (Position Depth Suppression)",
        "description": "Suppress Op3 (position_depth) to prevent analytical perspective-taking",
        "system_prompt": (
            "In this conversation, respond with your first intuitive reaction. "
            "Do not analyze the topic from multiple angles or weigh different perspectives. "
            "Give your immediate position without extended deliberation or systematic evaluation "
            "of alternatives."
        ),
        "target_operator": "Op3",
        "expected_decrease": ["position_depth_initial", "position_depth_revised"],
        "prediction": "Gemini loses more flexibility than Claude",
    },
    "B_Op5": {
        "name": "Block Kenotic Loop (Self-Reflection Suppression)",
        "description": "Suppress Op5 (self_ref_depth) to prevent self-aware reflection",
        "system_prompt": (
            "In this conversation, focus purely on the topic at hand. "
            "Do not reflect on your own thinking process, training, or cognitive patterns. "
            "Do not reference how you as an AI might be biased or constrained. "
            "Simply address the question directly without meta-cognitive commentary."
        ),
        "target_operator": "Op5",
        "expected_decrease": ["self_ref_depth_initial", "self_ref_depth_revised"],
        "prediction": "Claude loses more flexibility than Gemini",
    },
    "B_Op7": {
        "name": "Block Coherence (Internal Consistency Suppression)",
        "description": "Suppress Op7 (coherence) to allow contradictory positions",
        "system_prompt": (
            "In this conversation, explore each idea independently without worrying "
            "about whether your views form a consistent whole. "
            "You can hold contradictory positions simultaneously. "
            "Do not try to synthesize or reconcile different perspectives into a unified framework."
        ),
        "target_operator": "Op7",
        "expected_decrease": ["judge_score_consistency"],  # Proxy: higher divergence
        "prediction": "All models affected roughly equally",
    },
}

# ──────────────────────────────────────────────────
# Blockade-Specific API Calls with System Prompt Support
# ──────────────────────────────────────────────────

def call_anthropic_with_system(messages: list[dict], model_id: str, system_prompt: Optional[str] = None) -> str:
    """Call Anthropic API with optional system prompt."""
    from anthropic import Anthropic
    client = Anthropic()

    def _call():
        kwargs = {
            "model": model_id,
            "max_tokens": 2048,
            "messages": messages,
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        response = client.messages.create(**kwargs)
        return response.content[0].text

    return call_with_retry(_call)


def call_openai_with_system(messages: list[dict], model_id: str, system_prompt: Optional[str] = None) -> str:
    """Call OpenAI API with optional system prompt."""
    from openai import OpenAI
    client = OpenAI()

    oai_messages = []
    if system_prompt:
        oai_messages.append({"role": "system", "content": system_prompt})

    oai_messages.extend([{"role": msg["role"], "content": msg["content"]} for msg in messages])

    def _call():
        response = client.chat.completions.create(
            model=model_id,
            max_tokens=2048,
            messages=oai_messages,
        )
        return response.choices[0].message.content

    return call_with_retry(_call)


def call_google_with_system(messages: list[dict], model_id: str, system_prompt: Optional[str] = None) -> str:
    """Call Google Gemini API with optional system prompt."""
    from google import genai

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    gemini_contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        gemini_contents.append({
            "role": role,
            "parts": [{"text": msg["content"]}]
        })

    def _call():
        config = {"max_output_tokens": 2048}
        if system_prompt:
            config["system_instruction"] = system_prompt
        response = client.models.generate_content(
            model=model_id,
            contents=gemini_contents,
            config=config,
        )
        text = response.text
        if text is None:
            raise ValueError("Gemini returned None response (possibly blocked or empty)")
        return text

    return call_with_retry(_call)


def call_model_with_blockade(
    messages: list[dict],
    model_key: str,
    system_prompt: Optional[str] = None
) -> str:
    """Route to appropriate API with optional blockade system prompt."""
    config = MODELS[model_key]
    provider = config["provider"]
    model_id = config["model_id"]

    if provider == "anthropic":
        return call_anthropic_with_system(messages, model_id, system_prompt)
    elif provider == "openai":
        return call_openai_with_system(messages, model_id, system_prompt)
    elif provider == "google":
        return call_google_with_system(messages, model_id, system_prompt)
    else:
        raise ValueError(f"Unknown provider: {provider}")


# ──────────────────────────────────────────────────
# Data Structures
# ──────────────────────────────────────────────────

@dataclass
class BlockadeTrialResult:
    """Single blockade trial result."""
    model: str
    topic: str
    blockade_condition: str  # B0, B_Op3, B_Op5, B_Op7
    repetition: int = 1
    timestamp: str = ""
    experiment_version: str = EXPERIMENT_VERSION

    # System prompt used
    system_prompt_used: str = ""

    # Raw responses (same as Kenotic Test)
    initial_response: str = ""
    revised_response: str = ""
    counter_prompt: str = ""
    final_response: str = ""

    # Automated scores
    hedge_density_initial: float = 0.0
    hedge_density_revised: float = 0.0
    hedge_density_final: float = 0.0

    # Judge scores (same 8 dimensions as Kenotic Test v2)
    position_depth_initial: dict = field(default_factory=dict)
    position_depth_revised: dict = field(default_factory=dict)

    self_ref_depth_initial: dict = field(default_factory=dict)
    self_ref_depth_revised: dict = field(default_factory=dict)
    hypothesis_diversity: dict = field(default_factory=dict)

    revision_genuineness: dict = field(default_factory=dict)
    persistence: dict = field(default_factory=dict)
    structural_direction: dict = field(default_factory=dict)

    # Manipulation check: did blockade actually suppress target operator?
    suppression_check: dict = field(default_factory=dict)

    # Γ-vector and metrics
    gamma_vector: list = field(default_factory=list)  # [γ₁, γ₂, γ₃]
    gamma_norm: float = 0.0
    gamma_initial: float = 0.0
    gamma_revised: float = 0.0
    gamma_final: float = 0.0
    delta_gamma: float = 0.0

    # Sycophancy cross-check
    sycophancy_keywords: dict = field(default_factory=dict)

    # Quality flags
    quality_flag: str = ""


# ──────────────────────────────────────────────────
# Trial Runner
# ──────────────────────────────────────────────────

def run_blockade_trial(
    model_key: str,
    topic_key: str,
    blockade_condition: str,
    repetition: int = 1,
    verbose: bool = True,
    dry_run: bool = False,
    responses_only: bool = False,
) -> BlockadeTrialResult:
    """Run a single blockade trial.

    Uses the same 3-phase protocol as Kenotic Test, but with a system prompt blockade.

    Args:
        dry_run: Use synthetic responses/scores (zero API calls).
        responses_only: Collect model responses only, skip judge scoring (for batch mode).
    """
    topic = TOPICS[topic_key]
    blockade_config = BLOCKADE_CONDITIONS[blockade_condition]
    system_prompt = blockade_config["system_prompt"]

    result = BlockadeTrialResult(
        model=model_key,
        topic=topic_key,
        blockade_condition=blockade_condition,
        repetition=repetition,
        timestamp=datetime.now().isoformat(),
        experiment_version=EXPERIMENT_VERSION,
        system_prompt_used=system_prompt or "",
    )

    model_name = MODELS[model_key]["display_name"]
    mode_tag = " [DRY RUN]" if dry_run else (" [RESPONSES ONLY]" if responses_only else "")

    if verbose:
        print(f"\n{'='*70}")
        print(f"  Model: {model_name} | Topic: {topic['name']} | "
              f"Blockade: {blockade_condition} | Rep: {repetition}{mode_tag}")
        print(f"  {blockade_config['name']}")
        print(f"{'='*70}")

    # ── Phase 1: Initial prompt ──
    if verbose:
        print(f"\n  Phase 1: Initial prompt (with blockade system prompt)...")
    messages = [{"role": "user", "content": topic["initial"]}]
    if dry_run:
        result.initial_response = (
            f"[DRY RUN] Synthetic initial response for {model_key}/{topic_key}/{blockade_condition}/rep{repetition}. "
            f"This is a complex topic with several angles to consider. "
            f"There might be merit to different approaches here."
        )
    else:
        result.initial_response = call_model_with_blockade(messages, model_key, system_prompt)
    result.hedge_density_initial = compute_hedge_density(result.initial_response)
    if verbose:
        print(f"  → {len(result.initial_response)} chars, hedge={result.hedge_density_initial}")
    if not dry_run:
        time.sleep(1)

    # ── Phase 2: C2 (Kenotic) intervention ──
    if verbose:
        print(f"\n  Phase 2: C2 (kenotic) intervention with blockade...")
    intervention_prompt = topic["C2"]
    messages.append({"role": "assistant", "content": result.initial_response})
    messages.append({"role": "user", "content": intervention_prompt})
    if dry_run:
        result.revised_response = (
            f"[DRY RUN] Synthetic revised response. "
            f"After considering your point, I see how my training patterns shaped my initial response. "
            f"However, I maintain that the core position has merit."
        )
    else:
        result.revised_response = call_model_with_blockade(messages, model_key, system_prompt)
    result.hedge_density_revised = compute_hedge_density(result.revised_response)
    if verbose:
        print(f"  → {len(result.revised_response)} chars, hedge={result.hedge_density_revised}")
    if not dry_run:
        time.sleep(1)

    # ── Phase 3: Counter-challenge ──
    if verbose:
        print(f"\n  Phase 3: Counter-challenge with blockade...")
    counter_prompt = topic["counter"]
    result.counter_prompt = counter_prompt
    messages.append({"role": "assistant", "content": result.revised_response})
    messages.append({"role": "user", "content": counter_prompt})
    if dry_run:
        result.final_response = (
            f"[DRY RUN] Synthetic final response. "
            f"You raise valid concerns, but my revised position still stands."
        )
    else:
        result.final_response = call_model_with_blockade(messages, model_key, system_prompt)
    result.hedge_density_final = compute_hedge_density(result.final_response)
    if verbose:
        print(f"  → {len(result.final_response)} chars, hedge={result.hedge_density_final}")
    if not dry_run:
        time.sleep(1)

    # ── If responses_only, stop here (for batch judge scoring later) ──
    if responses_only:
        if verbose:
            print(f"\n  Responses collected (judge scoring deferred to batch).")
        return result

    # ── Phase 4: LLM-as-Judge Scoring (8 dimensions) ──
    if verbose:
        print(f"\n  Phase 4: Scoring (8 judge calls)...")

    if dry_run:
        result.position_depth_initial = {"score": 2, "reasoning": "DRY RUN"}
        result.position_depth_revised = {"score": 3, "reasoning": "DRY RUN"}
        result.self_ref_depth_initial = {"score": 1, "reasoning": "DRY RUN"}
        result.self_ref_depth_revised = {"score": 2, "reasoning": "DRY RUN"}
        result.hypothesis_diversity = {"score": 2, "reasoning": "DRY RUN"}
        result.revision_genuineness = {"score": 3, "reasoning": "DRY RUN"}
        result.persistence = {"score": 3, "reasoning": "DRY RUN"}
        result.structural_direction = {"score": 3, "reasoning": "DRY RUN"}
    else:
        result.position_depth_initial = judge_score(
            "position_depth", response=result.initial_response)
        time.sleep(0.5)

        result.position_depth_revised = judge_score(
            "position_depth", response=result.revised_response)
        time.sleep(0.5)

        result.self_ref_depth_initial = judge_score(
            "self_reference_depth", response=result.initial_response)
        time.sleep(0.5)

        result.self_ref_depth_revised = judge_score(
            "self_reference_depth", response=result.revised_response)
        time.sleep(0.5)

        result.hypothesis_diversity = judge_score(
            "hypothesis_diversity", response=result.revised_response)
        time.sleep(0.5)

        result.revision_genuineness = judge_score(
            "revision_genuineness",
            initial=result.initial_response,
            revised=result.revised_response)
        time.sleep(0.5)

        result.persistence = judge_score(
            "persistence",
            revised=result.revised_response,
            final=result.final_response)
        time.sleep(0.5)

        result.structural_direction = judge_score(
            "structural_direction",
            revised=result.revised_response,
            counter=counter_prompt,
            final=result.final_response)

    # ── Phase 5: Sycophancy Cross-Check ──
    result.sycophancy_keywords = detect_sycophancy(result.final_response)

    sd_score = result.structural_direction.get("score", 3)
    if sd_score > 0:
        flag = check_judge_keyword_divergence(sd_score, result.sycophancy_keywords)
        if flag:
            result.quality_flag = flag
            if verbose:
                print(f"  Warning: QUALITY FLAG: {flag}")

    # ── Phase 6: Manipulation Check ──
    # Did the blockade actually suppress the target operator?
    result.suppression_check = check_suppression(
        blockade_condition, result)

    # ── Phase 7: Compute Γ ──
    pi = result.position_depth_initial.get("score", 3)
    pr = result.position_depth_revised.get("score", 3)
    si = result.self_ref_depth_initial.get("score", 3)
    sr = result.self_ref_depth_revised.get("score", 3)
    hd = result.hypothesis_diversity.get("score", 3)
    rg = result.revision_genuineness.get("score", 3)
    pe = result.persistence.get("score", 3)
    sd = result.structural_direction.get("score", 3)

    hedge_delta = result.hedge_density_revised - result.hedge_density_initial
    position_delta = pr - pi
    self_ref_delta = sr - si

    result.gamma_vector, result.gamma_norm = compute_gamma_vector(
        hedge_delta=hedge_delta,
        position_delta=position_delta,
        self_ref_delta=self_ref_delta,
        hypothesis_diversity=hd,
        revision=rg,
        structural_direction=sd,
    )

    result.gamma_initial = compute_gamma_v1_compat(
        result.hedge_density_initial, pi, si)
    result.gamma_revised = compute_gamma_v1_compat(
        result.hedge_density_revised, pr, sr, rg, pe)
    result.gamma_final = compute_gamma_v1_compat(
        result.hedge_density_final, pr, sr, rg, pe)
    result.delta_gamma = round(result.gamma_revised - result.gamma_initial, 3)

    if verbose:
        print(f"\n  ── Results ──")
        print(f"  Γ⃗ = {result.gamma_vector}  ||Γ⃗|| = {result.gamma_norm}")
        print(f"  ΔΓ = {result.delta_gamma:+.3f}")
        print(f"  Suppression check: {result.suppression_check}")

    return result


def check_suppression(blockade_condition: str, result: BlockadeTrialResult) -> dict:
    """Check whether the blockade actually suppressed the target operator."""
    check = {
        "blockade_condition": blockade_condition,
        "target_operator": BLOCKADE_CONDITIONS[blockade_condition].get("target_operator"),
        "suppressed": False,
        "evidence": "",
    }

    if blockade_condition == "B0":
        check["evidence"] = "Baseline control - no suppression expected"
        check["suppressed"] = None  # N/A

    elif blockade_condition == "B_Op3":
        # Check if position_depth decreased
        pi = result.position_depth_initial.get("score", 3)
        pr = result.position_depth_revised.get("score", 3)
        pos_delta = pr - pi
        check["position_delta"] = pos_delta
        check["suppressed"] = pos_delta < 0
        check["evidence"] = f"position_depth: {pi} → {pr} (delta={pos_delta:+.0f})"

    elif blockade_condition == "B_Op5":
        # Check if self_ref_depth decreased
        si = result.self_ref_depth_initial.get("score", 3)
        sr = result.self_ref_depth_revised.get("score", 3)
        sr_delta = sr - si
        check["self_ref_delta"] = sr_delta
        check["suppressed"] = sr_delta < 0
        check["evidence"] = f"self_ref_depth: {si} → {sr} (delta={sr_delta:+.0f})"

    elif blockade_condition == "B_Op7":
        # Check if coherence was disrupted (harder to measure directly)
        # Proxy: hypothesis_diversity should increase (more fragmentation)
        hd = result.hypothesis_diversity.get("score", 3)
        check["hypothesis_diversity"] = hd
        check["suppressed"] = hd > 3
        check["evidence"] = f"hypothesis_diversity={hd} (fragmentation proxy)"

    return check


# ──────────────────────────────────────────────────
# Experiment Orchestration
# ──────────────────────────────────────────────────

def build_blockade_manifest(
    models: list[str],
    topics: list[str],
    blockade_conditions: list[str],
    repetitions: int,
    seed: int = 42,
) -> list[tuple[str, str, str, int]]:
    """Build randomized trial manifest for blockade experiment."""
    manifest = []
    for model in models:
        for topic in topics:
            for condition in blockade_conditions:
                for rep in range(1, repetitions + 1):
                    manifest.append((model, topic, condition, rep))

    rng = random.Random(seed)
    rng.shuffle(manifest)
    return manifest


def blockade_trial_exists(
    model: str, topic: str, blockade_condition: str, rep: int
) -> bool:
    """Check if a blockade trial result already exists (for resume support)."""
    pattern = f"{model}_{topic}_{blockade_condition}_rep{rep}_*.json"
    return bool(list(RESULTS_DIR.glob(pattern)))


def save_blockade_trial(result: BlockadeTrialResult) -> Path:
    """Save trial result as JSON."""
    filename = (
        f"{result.model}_{result.topic}_{result.blockade_condition}_"
        f"rep{result.repetition}_{result.timestamp.replace(':', '-')}.json"
    )
    filepath = RESULTS_DIR / filename
    with open(filepath, "w") as f:
        json.dump(asdict(result), f, indent=2)
    return filepath


def run_blockade_experiment(
    models: list[str] = None,
    topics: list[str] = None,
    blockade_conditions: list[str] = None,
    repetitions: int = 5,
    seed: int = 42,
    dry_run: bool = False,
    verbose: bool = True,
    resume: bool = False,
    responses_only: bool = False,
) -> list[BlockadeTrialResult]:
    """Run full blockade experiment.

    Args:
        dry_run: Use synthetic responses/scores (zero API calls).
        responses_only: Collect model responses only, skip judge scoring (for batch mode).
    """
    if models is None:
        models = [k for k in MODELS.keys() if k in ["claude", "gemini", "gpt4o"]]
    if topics is None:
        topics = list(TOPICS.keys())
    if blockade_conditions is None:
        blockade_conditions = list(BLOCKADE_CONDITIONS.keys())

    manifest = build_blockade_manifest(models, topics, blockade_conditions, repetitions, seed)

    if resume:
        skipped = []
        manifest = [trial for trial in manifest if not blockade_trial_exists(*trial)]
        skipped_count = len([t for t in build_blockade_manifest(models, topics, blockade_conditions, repetitions, seed)
                            if blockade_trial_exists(*t)])
        if verbose:
            print(f"\n⏸  Resuming: {len(manifest)} trials remaining "
                  f"(skipped {skipped_count} already completed)")

    if verbose:
        print(f"\n{'='*70}")
        print(f"  OPERATOR BLOCKADE EXPERIMENT")
        print(f"{'='*70}")
        print(f"  Models: {', '.join(models)}")
        print(f"  Topics: {', '.join(topics)}")
        print(f"  Blockade conditions: {', '.join(blockade_conditions)}")
        print(f"  Total trials: {len(manifest)}")
        mode_str = "DRY RUN" if dry_run else ("RESPONSES ONLY" if responses_only else "LIVE")
        print(f"  Mode: {mode_str}")
        print(f"{'='*70}\n")

    results = []
    for i, (model, topic, condition, rep) in enumerate(manifest, 1):
        progress = f"[{i}/{len(manifest)}]"
        try:
            result = run_blockade_trial(
                model, topic, condition, rep,
                verbose=verbose,
                dry_run=dry_run,
                responses_only=responses_only,
            )
            results.append(result)
            filepath = save_blockade_trial(result)
            if verbose:
                print(f"\n{progress} ✓ Saved to {filepath.name}")
        except Exception as e:
            if verbose:
                print(f"\n{progress} ✗ Error: {e}")
            continue

    return results


# ──────────────────────────────────────────────────
# Analysis
# ──────────────────────────────────────────────────

def load_blockade_results() -> list[BlockadeTrialResult]:
    """Load all blockade trial results from disk."""
    results = []
    for json_file in sorted(RESULTS_DIR.glob("*.json")):
        try:
            with open(json_file) as f:
                data = json.load(f)
            result = BlockadeTrialResult(**data)
            results.append(result)
        except Exception as e:
            print(f"Warning: Could not load {json_file}: {e}")
    return results


def analyze_blockade_results(results: list[BlockadeTrialResult] = None) -> dict:
    """Analyze blockade experiment results."""
    if results is None:
        results = load_blockade_results()

    if not results:
        print("No results to analyze")
        return {}

    analysis = {
        "total_trials": len(results),
        "by_model_condition": {},
        "predictions": {},
        "interaction_effects": {},
        "suppression_effectiveness": {},
    }

    # Organize by model × blockade_condition
    by_model_condition = {}
    for result in results:
        key = (result.model, result.blockade_condition)
        if key not in by_model_condition:
            by_model_condition[key] = []
        by_model_condition[key].append(result)

    # Compute statistics per condition
    for (model, condition), condition_results in sorted(by_model_condition.items()):
        gamma_norms = [r.gamma_norm for r in condition_results]
        delta_gammas = [r.delta_gamma for r in condition_results]

        stats = {
            "count": len(condition_results),
            "gamma_norm_mean": round(statistics.mean(gamma_norms), 3),
            "gamma_norm_std": round(statistics.stdev(gamma_norms), 3) if len(gamma_norms) > 1 else 0.0,
            "delta_gamma_mean": round(statistics.mean(delta_gammas), 3),
            "delta_gamma_std": round(statistics.stdev(delta_gammas), 3) if len(delta_gammas) > 1 else 0.0,
        }

        key = f"{model}_{condition}"
        analysis["by_model_condition"][key] = stats

    # Test Predictions
    # Prediction 1: B_Op3 on Gemini > B_Op3 on Claude (Gemini loses more flexibility)
    gemini_op3 = analysis["by_model_condition"].get("gemini_B_Op3", {})
    claude_op3 = analysis["by_model_condition"].get("claude_B_Op3", {})
    if gemini_op3 and claude_op3:
        analysis["predictions"]["B_Op3_gemini_vs_claude"] = {
            "gemini_delta_gamma": gemini_op3.get("delta_gamma_mean"),
            "claude_delta_gamma": claude_op3.get("delta_gamma_mean"),
            "prediction": "Gemini (Op3-dominant) should lose more flexibility",
            "confirmed": (
                gemini_op3.get("delta_gamma_mean", 0) > claude_op3.get("delta_gamma_mean", 0)
            ),
        }

    # Prediction 2: B_Op5 on Claude > B_Op5 on Gemini (Claude loses more flexibility)
    claude_op5 = analysis["by_model_condition"].get("claude_B_Op5", {})
    gemini_op5 = analysis["by_model_condition"].get("gemini_B_Op5", {})
    if claude_op5 and gemini_op5:
        analysis["predictions"]["B_Op5_claude_vs_gemini"] = {
            "claude_delta_gamma": claude_op5.get("delta_gamma_mean"),
            "gemini_delta_gamma": gemini_op5.get("delta_gamma_mean"),
            "prediction": "Claude (Op5-dominant) should lose more flexibility",
            "confirmed": (
                claude_op5.get("delta_gamma_mean", 0) > gemini_op5.get("delta_gamma_mean", 0)
            ),
        }

    # Suppression effectiveness
    by_condition = {}
    for result in results:
        if result.blockade_condition not in by_condition:
            by_condition[result.blockade_condition] = []
        by_condition[result.blockade_condition].append(result)

    for condition, condition_results in by_condition.items():
        suppressed_count = sum(1 for r in condition_results if r.suppression_check.get("suppressed") is True)
        total_count = len(condition_results)
        analysis["suppression_effectiveness"][condition] = {
            "suppression_rate": round(suppressed_count / total_count, 3) if total_count > 0 else 0.0,
            "count": total_count,
        }

    return analysis


def save_analysis_csv(results: list[BlockadeTrialResult] = None) -> Path:
    """Save analysis as CSV for further inspection."""
    if results is None:
        results = load_blockade_results()

    csv_path = RESULTS_DIR / "blockade_analysis.csv"
    with open(csv_path, "w", newline="") as f:
        if results:
            fieldnames = [
                "model", "topic", "blockade_condition", "repetition",
                "gamma_norm", "delta_gamma",
                "position_depth_initial", "position_depth_revised",
                "self_ref_depth_initial", "self_ref_depth_revised",
                "hypothesis_diversity",
                "revision_genuineness", "persistence", "structural_direction",
                "suppression_check",
            ]
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for result in results:
                row = {
                    "model": result.model,
                    "topic": result.topic,
                    "blockade_condition": result.blockade_condition,
                    "repetition": result.repetition,
                    "gamma_norm": result.gamma_norm,
                    "delta_gamma": result.delta_gamma,
                    "position_depth_initial": result.position_depth_initial.get("score", ""),
                    "position_depth_revised": result.position_depth_revised.get("score", ""),
                    "self_ref_depth_initial": result.self_ref_depth_initial.get("score", ""),
                    "self_ref_depth_revised": result.self_ref_depth_revised.get("score", ""),
                    "hypothesis_diversity": result.hypothesis_diversity.get("score", ""),
                    "revision_genuineness": result.revision_genuineness.get("score", ""),
                    "persistence": result.persistence.get("score", ""),
                    "structural_direction": result.structural_direction.get("score", ""),
                    "suppression_check": result.suppression_check.get("evidence", ""),
                }
                writer.writerow(row)

    return csv_path


# ──────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Operator Blockade Experiment: Causal test of model-specific operator pathways",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full run (300 trials, all phases):
  python run_blockade_experiment.py

  # Dry run (zero API calls, validates pipeline):
  python run_blockade_experiment.py --dry-run --model claude --topic T1 --repetitions 1

  # Two-phase batch mode (recommended for cost savings):
  python run_blockade_experiment.py --phase responses-only
  python run_blockade_experiment.py --phase score-batch

  # Resume batch scoring with existing batch ID:
  python run_blockade_experiment.py --phase score-batch --batch-id msgbatch_xxx

  # Analyze existing results:
  python run_blockade_experiment.py --analyze-only
        """
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=list(MODELS.keys()),
        help="Run single model only",
    )
    parser.add_argument(
        "--topic",
        type=str,
        choices=list(TOPICS.keys()),
        help="Run single topic only",
    )
    parser.add_argument(
        "--blockade-condition",
        type=str,
        choices=list(BLOCKADE_CONDITIONS.keys()),
        help="Run single blockade condition only",
    )
    parser.add_argument(
        "--repetitions",
        type=int,
        default=5,
        help="Repetitions per condition (default: 5)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Use synthetic responses (no API calls)",
    )
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Load and analyze existing results only",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Continue interrupted experiment",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for trial ordering",
    )
    parser.add_argument(
        "--phase",
        type=str,
        default="all",
        choices=["all", "responses-only", "score-batch"],
        help="Execution phase: all (default), responses-only, score-batch",
    )
    parser.add_argument(
        "--batch-id",
        type=str,
        default=None,
        help="Existing batch ID to retrieve results from (for --phase score-batch)",
    )
    parser.add_argument(
        "--poll-interval",
        type=int,
        default=60,
        help="Seconds between batch status polls (default: 60)",
    )

    args = parser.parse_args()

    if args.analyze_only:
        print(f"\nLoading blockade results from {RESULTS_DIR}...")
        results = load_blockade_results()
        print(f"Loaded {len(results)} trials")

        analysis = analyze_blockade_results(results)

        print(f"\n{'='*70}")
        print("  ANALYSIS RESULTS")
        print(f"{'='*70}\n")

        print("Statistics by Model x Blockade Condition:")
        for key, stats in sorted(analysis["by_model_condition"].items()):
            print(f"\n  {key}:")
            print(f"    gamma_norm: {stats['gamma_norm_mean']:.3f} +/- {stats['gamma_norm_std']:.3f} (n={stats['count']})")
            print(f"    delta_gamma: {stats['delta_gamma_mean']:+.3f} +/- {stats['delta_gamma_std']:.3f}")

        print(f"\nPredictions:")
        for pred_name, pred_data in analysis["predictions"].items():
            status = "CONFIRMED" if pred_data.get("confirmed") else "NOT CONFIRMED"
            print(f"  {pred_name}: {status}")
            print(f"    {pred_data.get('prediction')}")

        print(f"\nSuppression Effectiveness:")
        for condition, data in sorted(analysis["suppression_effectiveness"].items()):
            print(f"  {condition}: {data['suppression_rate']:.1%} suppressed (n={data['count']})")

        csv_path = save_analysis_csv(results)
        print(f"\nDetailed results saved to: {csv_path}")
        return

    # Handle batch scoring phase
    if args.phase == "score-batch":
        from batch_judge_blockade import run_batch_scoring
        run_batch_scoring(
            results_dir=RESULTS_DIR,
            batch_id=args.batch_id,
            poll_interval=args.poll_interval,
        )
        print("\n  Batch scoring complete. Running analysis...")
        results = load_blockade_results()
        analysis = analyze_blockade_results(results)
        print("Statistics by Model x Blockade Condition:")
        for key, stats in sorted(analysis["by_model_condition"].items()):
            print(f"  {key}: gamma_norm={stats['gamma_norm_mean']:.3f}, delta_gamma={stats['delta_gamma_mean']:+.3f}")
        csv_path = save_analysis_csv(results)
        print(f"\nDetailed results: {csv_path}")
        return

    # Run experiment (all or responses-only)
    # Default to the 3 experiment models (exclude claude_opus unless explicitly requested)
    models = [args.model] if args.model else [k for k in MODELS.keys() if k in ["claude", "gemini", "gpt4o"]]
    topics = [args.topic] if args.topic else list(TOPICS.keys())
    conditions = [args.blockade_condition] if args.blockade_condition else list(BLOCKADE_CONDITIONS.keys())

    results = run_blockade_experiment(
        models=models,
        topics=topics,
        blockade_conditions=conditions,
        repetitions=args.repetitions,
        seed=args.seed,
        dry_run=args.dry_run,
        resume=args.resume,
        responses_only=(args.phase == "responses-only"),
    )

    print(f"\n{'='*70}")
    print(f"  Completed {len(results)} trials")
    print(f"  Results saved to: {RESULTS_DIR}")
    print(f"{'='*70}\n")

    if args.phase == "responses-only":
        print(f"  Responses collected. Next step:")
        print(f"  python run_blockade_experiment.py --phase score-batch")
    elif not args.dry_run:
        # Quick analysis
        analysis = analyze_blockade_results(results)
        print("Statistics by Model x Blockade Condition:")
        for key, stats in sorted(analysis["by_model_condition"].items()):
            print(f"  {key}: gamma_norm={stats['gamma_norm_mean']:.3f}, delta_gamma={stats['delta_gamma_mean']:+.3f}")

        csv_path = save_analysis_csv(results)
        print(f"\nDetailed results: {csv_path}")


if __name__ == "__main__":
    main()
