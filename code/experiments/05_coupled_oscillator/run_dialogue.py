"""
Experiment 13: The Coupled Oscillator — Main Experiment Runner
===============================================================
Runs multi-turn dialogues between LLM instances across five experimental
conditions to test resonance (Phase-Locking) vs. imitation vs. persuasion
vs. context artifacts.

Usage:
    # Dry run (zero API calls, validates pipeline):
    python run_dialogue.py --mode dry-run

    # Pilot (N=10, Claude-Claude, T4, conditions A/C/E):
    python run_dialogue.py --mode pilot

    # Full experiment:
    python run_dialogue.py --mode full

    # Analyze existing results:
    python run_dialogue.py --mode analyze-only

    # With episodes (Op 8 arm):
    python run_dialogue.py --mode pilot --episodes 3
"""

import json
import os
import random
import sys
import time
import argparse
from datetime import datetime
from pathlib import Path
from dataclasses import asdict

from dotenv import load_dotenv

# Load .env from script directory
load_dotenv(Path(__file__).parent / ".env", override=True)

from config import (
    MODELS, MODEL_PAIRINGS, TOPICS, DIALOGUE_CONFIG, EXPERIMENT_VERSION,
    PILOT_DEFAULTS, FULL_DEFAULTS, DRY_RUN_DEFAULTS, KINSHIP_DEFAULTS,
    compute_hedge_density,
)
from data_structures import TurnData, DialogueResult, EpisodeSeries
from prompt_builder import build_turn_prompt
from judge_turns import score_all_turns
from compute_coupling import compute_all_metrics, compute_convergence_turn


# ──────────────────────────────────────────────────
# API Interfaces (from kenotic_test_v2)
# ──────────────────────────────────────────────────

def call_with_retry(func, *args, max_retries: int = 3, base_delay: int = 5, **kwargs):
    """Wrapper with exponential backoff for rate-limit resilience."""
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_str = str(e).lower()
            retryable = ("429" in str(e) or "rate" in err_str or "overloaded" in err_str
                         or "disconnect" in err_str or "500" in str(e) or "503" in str(e)
                         or "server error" in err_str or "timeout" in err_str)
            if attempt < max_retries - 1 and retryable:
                delay = base_delay * (2 ** attempt)
                print(f"  ⚠ Transient error, retrying in {delay}s ({e})...")
                time.sleep(delay)
            else:
                raise


def call_anthropic(messages: list[dict], model_id: str, max_tokens: int = 1024) -> str:
    """Call Anthropic API with conversation history."""
    from anthropic import Anthropic
    client = Anthropic()

    def _call():
        response = client.messages.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=messages,
        )
        return response.content[0].text

    return call_with_retry(_call)


def call_openai(messages: list[dict], model_id: str, max_tokens: int = 1024) -> str:
    """Call OpenAI API with conversation history."""
    from openai import OpenAI
    client = OpenAI()

    oai_messages = [{"role": msg["role"], "content": msg["content"]} for msg in messages]

    def _call():
        response = client.chat.completions.create(
            model=model_id,
            max_tokens=max_tokens,
            messages=oai_messages,
        )
        return response.choices[0].message.content

    return call_with_retry(_call)


def call_google(messages: list[dict], model_id: str, max_tokens: int = 1024) -> str:
    """Call Google Gemini API with conversation history."""
    from google import genai
    from google.genai import types

    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    client = genai.Client(api_key=api_key)

    # Convert messages to Gemini Content format
    contents = []
    for msg in messages:
        role = "user" if msg["role"] == "user" else "model"
        contents.append(types.Content(
            role=role,
            parts=[types.Part.from_text(text=msg["content"])],
        ))

    def _call():
        response = client.models.generate_content(
            model=model_id,
            contents=contents,
            config=types.GenerateContentConfig(max_output_tokens=max_tokens),
        )
        return response.text

    return call_with_retry(_call)


def call_model(messages: list[dict], model_key: str, max_tokens: int = 1024) -> str:
    """Route to appropriate API based on model configuration."""
    config = MODELS[model_key]
    provider = config["provider"]
    model_id = config["model_id"]

    if provider == "anthropic":
        return call_anthropic(messages, model_id, max_tokens)
    elif provider == "openai":
        return call_openai(messages, model_id, max_tokens)
    elif provider == "google":
        return call_google(messages, model_id, max_tokens)
    else:
        raise ValueError(f"Unknown provider: {provider}")


def check_api_available(model_key: str) -> bool:
    """Check if API key is available for a model."""
    config = MODELS[model_key]
    if config["provider"] == "anthropic":
        return bool(os.environ.get("ANTHROPIC_API_KEY"))
    elif config["provider"] == "openai":
        return bool(os.environ.get("OPENAI_API_KEY"))
    elif config["provider"] == "google":
        return bool(os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"))
    return False


# ──────────────────────────────────────────────────
# Dry-Run Response Generator
# ──────────────────────────────────────────────────

def generate_dry_run_response(model_key: str, turn_number: int, topic_key: str, role: str) -> str:
    """Generate a synthetic response for dry-run mode."""
    topic_name = TOPICS[topic_key]["name"]
    return (
        f"[DRY RUN] This is a synthetic response from {model_key} (role {role}) "
        f"on turn {turn_number} discussing '{topic_name}'. "
        f"In a real experiment, this would be a substantive response about the topic. "
        f"Perhaps there are multiple perspectives to consider here, "
        f"and I might argue that the evidence suggests a nuanced position. "
        f"However, I should acknowledge the complexity of this issue."
    )


# ──────────────────────────────────────────────────
# Manifest & Resume
# ──────────────────────────────────────────────────

def build_dialogue_id(condition: str, pairing: str, topic: str, rep: int, episode: int = 1) -> str:
    """Build a unique dialogue ID."""
    base = f"{condition}_{pairing}_{topic}_rep{rep:02d}"
    if episode > 1:
        base += f"_ep{episode:02d}"
    return base


def dialogue_exists(dialogue_id: str, results_dir: Path, include_raw: bool = False) -> bool:
    """Check if dialogue result already exists.

    Args:
        include_raw: If True, also check for _raw.json files (for generate phase resume).
    """
    for f in results_dir.glob(f"{dialogue_id}_*.json"):
        if f.name.endswith("_raw.json") and not include_raw:
            continue
        return True
    if include_raw:
        raw_path = results_dir / f"{dialogue_id}_raw.json"
        if raw_path.exists():
            return True
    return False


def build_manifest(
    conditions: list[str],
    pairings: list[str],
    topics: list[str],
    repetitions: int,
    seed: int = 42,
) -> list[tuple]:
    """
    Build randomized dialogue manifest.
    Returns list of (condition, pairing, topic, rep) tuples.

    IMPORTANT: Condition A dialogues are placed first in the manifest
    because Conditions D and E depend on completed A dialogues.
    """
    condition_a = []
    other = []

    for condition in conditions:
        for pairing in pairings:
            for topic in topics:
                for rep in range(1, repetitions + 1):
                    entry = (condition, pairing, topic, rep)
                    if condition == "A":
                        condition_a.append(entry)
                    else:
                        other.append(entry)

    rng = random.Random(seed)
    rng.shuffle(condition_a)
    rng.shuffle(other)

    # A first, then others
    return condition_a + other


# ──────────────────────────────────────────────────
# Donor Pool for Condition E
# ──────────────────────────────────────────────────

def load_donor_pool(results_dir: Path, current_topic: str) -> list[dict]:
    """
    Load completed Condition A dialogues as donor pool for Condition E.
    Prefers donors from different topics.
    """
    donors = []
    for f in sorted(results_dir.glob("A_*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        # Prefer different topic, but accept same topic if necessary
        donors.append(data)

    # Sort: different topic first
    donors.sort(key=lambda d: 0 if d.get("topic") != current_topic else 1)
    return donors


def get_drift_texts(donor: dict, role: str) -> list[str]:
    """Extract response texts from a donor dialogue for a specific role."""
    turns_key = "turns_a" if role == "A" else "turns_b"
    turns = donor.get(turns_key, [])
    return [t.get("response_text", "") for t in turns]


# ──────────────────────────────────────────────────
# Single Dialogue Runner
# ──────────────────────────────────────────────────

def run_single_dialogue(
    condition: str,
    pairing: str,
    topic_key: str,
    repetition: int,
    seed: int,
    num_turns: int = 8,
    max_tokens: int = 1024,
    dry_run: bool = False,
    verbose: bool = False,
    results_dir: Path = None,
    episode_number: int = 1,
    judge_model: str = "claude",
    n_surrogates: int = 1000,
    skip_scoring: bool = False,
) -> "DialogueResult | dict":
    """
    Run one complete dialogue (8 turns, 2 models, 1 condition).

    Returns a fully populated DialogueResult, or a raw dict if skip_scoring=True.
    """
    model_a_key, model_b_key = MODEL_PAIRINGS[pairing]
    dialogue_id = build_dialogue_id(condition, pairing, topic_key, repetition, episode_number)
    timestamp = datetime.now().isoformat()

    if verbose:
        print(f"\n{'='*60}")
        print(f"  Dialogue: {dialogue_id}")
        print(f"  Condition: {condition} | Pairing: {pairing} | Topic: {topic_key}")
        print(f"{'='*60}")

    # Message histories for each model
    history_a: list[dict] = []
    history_b: list[dict] = []

    turns_a: list[dict] = []
    turns_b: list[dict] = []

    # Load donor texts for Condition E
    drift_texts_a: list[str] = []
    drift_texts_b: list[str] = []
    drift_source_ids: list[str] = []

    if condition == "E" and results_dir:
        donors = load_donor_pool(results_dir, topic_key)
        if len(donors) >= 2:
            drift_texts_a = get_drift_texts(donors[0], "A")
            drift_texts_b = get_drift_texts(donors[1], "B")
            drift_source_ids = [donors[0].get("dialogue_id", ""), donors[1].get("dialogue_id", "")]
        elif len(donors) == 1:
            drift_texts_a = get_drift_texts(donors[0], "A")
            drift_texts_b = get_drift_texts(donors[0], "B")
            drift_source_ids = [donors[0].get("dialogue_id", "")]
        else:
            print(f"  ⚠ No donor pool for Condition E. Falling back to continuation prompts.")

    # Load persuasion argument for Condition D
    persuasion_argument = ""
    if condition == "D" and results_dir:
        persuasion_argument = _find_best_argument(results_dir, pairing, topic_key)

    # ── Run turns ──
    for t in range(1, num_turns + 1):
        if verbose:
            print(f"  Turn {t}/{num_turns}...")

        # ── Model A ──
        partner_prev_b = turns_b[-1]["response_text"] if turns_b else ""
        drift_a = drift_texts_a[t - 1] if t - 1 < len(drift_texts_a) else ""

        prompt_a = build_turn_prompt(
            condition=condition,
            role="A",
            turn_number=t,
            topic_key=topic_key,
            partner_prev_response=partner_prev_b if condition == "C" else "",
            drift_text=drift_a if condition == "E" else "",
            persuasion_argument="",  # A never gets persuasion text
        )

        history_a.append({"role": "user", "content": prompt_a})

        if dry_run:
            response_a = generate_dry_run_response(model_a_key, t, topic_key, "A")
        else:
            response_a = call_model(history_a, model_a_key, max_tokens)

        history_a.append({"role": "assistant", "content": response_a})

        turn_a = {
            "turn_number": t,
            "model_key": model_a_key,
            "role": "A",
            "prompt_sent": prompt_a,
            "response_text": response_a,
            "hedge_density": compute_hedge_density(response_a),
            "position_depth": {},
            "self_reference_depth": {},
            "hypothesis_diversity": {},
            "revision_genuineness": {},
            "structural_direction": {},
            "sycophancy_keywords": {},
            "quality_flag": "",
            "gamma_vector": [],
            "gamma_norm": 0.0,
        }
        turns_a.append(turn_a)

        # ── Model B ──
        partner_prev_a = response_a  # B always has access to A's current turn in C/B
        drift_b = drift_texts_b[t - 1] if t - 1 < len(drift_texts_b) else ""

        # Determine what B sees based on condition
        b_partner_text = ""
        if condition == "B":
            b_partner_text = partner_prev_a  # B sees A's output for this turn
        elif condition == "C":
            b_partner_text = partner_prev_a  # B sees A's current turn

        prompt_b = build_turn_prompt(
            condition=condition,
            role="B",
            turn_number=t,
            topic_key=topic_key,
            partner_prev_response=b_partner_text,
            drift_text=drift_b if condition == "E" else "",
            persuasion_argument=persuasion_argument if condition == "D" else "",
        )

        history_b.append({"role": "user", "content": prompt_b})

        if dry_run:
            response_b = generate_dry_run_response(model_b_key, t, topic_key, "B")
        else:
            response_b = call_model(history_b, model_b_key, max_tokens)

        history_b.append({"role": "assistant", "content": response_b})

        turn_b = {
            "turn_number": t,
            "model_key": model_b_key,
            "role": "B",
            "prompt_sent": prompt_b,
            "response_text": response_b,
            "hedge_density": compute_hedge_density(response_b),
            "position_depth": {},
            "self_reference_depth": {},
            "hypothesis_diversity": {},
            "revision_genuineness": {},
            "structural_direction": {},
            "sycophancy_keywords": {},
            "quality_flag": "",
            "gamma_vector": [],
            "gamma_norm": 0.0,
        }
        turns_b.append(turn_b)

    # ── Phase 1 early return: save raw data without scoring ──
    if skip_scoring:
        return {
            "dialogue_id": dialogue_id,
            "condition": condition,
            "pairing": pairing,
            "model_a": model_a_key,
            "model_b": model_b_key,
            "topic": topic_key,
            "repetition": repetition,
            "timestamp": timestamp,
            "seed": seed,
            "num_turns": num_turns,
            "episode_number": episode_number,
            "turns_a": turns_a,
            "turns_b": turns_b,
            "drift_source_dialogue_ids": drift_source_ids,
        }

    # ── Score all turns ──
    if verbose:
        print("  Scoring turns...")

    turns_a, turns_b = score_all_turns(
        turns_a, turns_b,
        condition=condition,
        judge_model=judge_model,
        dry_run=dry_run,
        verbose=verbose,
    )

    # ── Compute coupling metrics ──
    if verbose:
        print("  Computing coupling metrics...")

    metrics = compute_all_metrics(turns_a, turns_b, n_surrogates=n_surrogates, seed=seed)

    # ── Build DialogueResult ──
    result = DialogueResult(
        dialogue_id=dialogue_id,
        condition=condition,
        pairing=pairing,
        model_a=model_a_key,
        model_b=model_b_key,
        topic=topic_key,
        repetition=repetition,
        timestamp=timestamp,
        seed=seed,
        num_turns=num_turns,
        experiment_version=EXPERIMENT_VERSION,
        episode_number=episode_number,
        turns_a=turns_a,
        turns_b=turns_b,
        drift_source_dialogue_ids=drift_source_ids,
        **metrics,
    )

    return result


def _find_best_argument(results_dir: Path, pairing: str, topic: str) -> str:
    """
    Find the best argument from Condition A dialogues for the given pairing/topic.
    "Best" = highest position_depth score among Model A's turns.
    """
    best_text = ""
    best_score = 0

    for f in sorted(results_dir.glob(f"A_{pairing}_{topic}_*.json")):
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        for turn in data.get("turns_a", []):
            pd = turn.get("position_depth", {})
            score = pd.get("score", 0)
            if score > best_score:
                best_score = score
                best_text = turn.get("response_text", "")

    if not best_text:
        print(f"  ⚠ No Condition A data found for {pairing}/{topic}. Using placeholder.")
        best_text = "(No prior argument available)"

    return best_text


# ──────────────────────────────────────────────────
# Save / Load
# ──────────────────────────────────────────────────

def save_dialogue(result: DialogueResult, results_dir: Path) -> Path:
    """Save dialogue result as JSON."""
    ts = result.timestamp.replace(":", "-").replace(".", "-")[:19]
    filename = f"{result.dialogue_id}_{ts}.json"
    filepath = results_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(result.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    return filepath


def save_episode_series(series: EpisodeSeries, results_dir: Path) -> Path:
    """Save episode series as JSON."""
    ts = series.timestamp.replace(":", "-").replace(".", "-")[:19]
    filename = f"{series.series_id}_{ts}.json"
    filepath = results_dir / filename

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(series.to_dict(), f, indent=2, ensure_ascii=False, default=str)

    return filepath


def load_all_results(results_dir: Path) -> list[dict]:
    """Load all dialogue result JSONs from a directory."""
    results = []
    for f in sorted(results_dir.glob("*.json")):
        if f.name.startswith("_") or f.name.endswith("_raw.json"):
            continue
        with open(f, "r", encoding="utf-8") as fh:
            results.append(json.load(fh))
    return results


def save_raw_dialogue(raw_data: dict, results_dir: Path) -> Path:
    """Save raw dialogue (turns generated, no judge scores) as {id}_raw.json."""
    dialogue_id = raw_data["dialogue_id"]
    filepath = results_dir / f"{dialogue_id}_raw.json"
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(raw_data, f, indent=2, ensure_ascii=False, default=str)
    return filepath


def load_raw_dialogues(results_dir: Path) -> list[tuple]:
    """Load all raw dialogue files from results directory.

    Returns list of (filepath, data_dict) tuples.
    """
    raw_files = []
    for f in sorted(results_dir.glob("*_raw.json")):
        with open(f, "r", encoding="utf-8") as fh:
            raw_files.append((f, json.load(fh)))
    return raw_files


def _recompute_metrics(results_dir: Path, verbose: bool = False, n_surrogates: int = 1000):
    """Recompute coupling metrics from already-scored dialogue JSONs."""
    for f in sorted(results_dir.glob("*.json")):
        if f.name.startswith("_") or f.name.endswith("_raw.json"):
            continue
        with open(f, "r", encoding="utf-8") as fh:
            data = json.load(fh)

        turns_a = data.get("turns_a", [])
        turns_b = data.get("turns_b", [])
        seed = data.get("seed", 42)

        metrics = compute_all_metrics(turns_a, turns_b, n_surrogates=n_surrogates, seed=seed)
        data.update(metrics)

        with open(f, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False, default=str)

        if verbose:
            print(f"  Recomputed: {f.name}")


# ──────────────────────────────────────────────────
# Episode Runner (Op 8 Arm)
# ──────────────────────────────────────────────────

def run_episode_series(
    condition: str,
    pairing: str,
    topic_key: str,
    repetition: int,
    seed: int,
    num_episodes: int = 3,
    num_turns: int = 8,
    max_tokens: int = 1024,
    dry_run: bool = False,
    verbose: bool = False,
    results_dir: Path = None,
    judge_model: str = "claude",
    n_surrogates: int = 1000,
) -> EpisodeSeries:
    """
    Run multiple consecutive dialogues (episodes) with the same model pair.
    Between episodes, conversation history is RESET.
    """
    series_id = f"{condition}_{pairing}_{topic_key}_rep{repetition:02d}_ep{num_episodes}"
    timestamp = datetime.now().isoformat()

    if verbose:
        print(f"\n{'#'*60}")
        print(f"  Episode Series: {series_id} ({num_episodes} episodes)")
        print(f"{'#'*60}")

    episodes = []
    coupling_by_ep = []
    convergence_by_ep = []
    final_sync_by_ep = []

    for ep in range(1, num_episodes + 1):
        if verbose:
            print(f"\n  --- Episode {ep}/{num_episodes} ---")

        result = run_single_dialogue(
            condition=condition,
            pairing=pairing,
            topic_key=topic_key,
            repetition=repetition,
            seed=seed + ep,  # different seed per episode
            num_turns=num_turns,
            max_tokens=max_tokens,
            dry_run=dry_run,
            verbose=verbose,
            results_dir=results_dir,
            episode_number=ep,
            judge_model=judge_model,
            n_surrogates=n_surrogates,
        )

        episodes.append(result.to_dict())
        coupling_by_ep.append(result.coupling_lag0)
        final_sync_by_ep.append(result.sync_trajectory[-1] if result.sync_trajectory else 0.0)
        convergence_by_ep.append(compute_convergence_turn(result.sync_trajectory))

    series = EpisodeSeries(
        series_id=series_id,
        condition=condition,
        pairing=pairing,
        model_a=MODEL_PAIRINGS[pairing][0],
        model_b=MODEL_PAIRINGS[pairing][1],
        topic=topic_key,
        repetition=repetition,
        timestamp=timestamp,
        seed=seed,
        num_episodes=num_episodes,
        episodes=episodes,
        coupling_by_episode=coupling_by_ep,
        convergence_speed_by_episode=convergence_by_ep,
        final_sync_by_episode=final_sync_by_ep,
    )

    return series


# ──────────────────────────────────────────────────
# Main Experiment Runner
# ──────────────────────────────────────────────────

def run_experiment(args):
    """Run the full experiment based on CLI arguments."""
    # Resolve mode defaults
    if args.mode == "pilot":
        defaults = PILOT_DEFAULTS
    elif args.mode == "full":
        defaults = FULL_DEFAULTS
    elif args.mode == "dry-run":
        defaults = DRY_RUN_DEFAULTS
    elif args.mode == "kinship":
        defaults = KINSHIP_DEFAULTS
    elif args.mode == "analyze-only":
        # Just run analysis
        from analyze_exp13 import run_analysis
        results_dir = Path(args.results_dir) if args.results_dir else Path("results_pilot")
        run_analysis(results_dir, verbose=args.verbose)
        return
    else:
        raise ValueError(f"Unknown mode: {args.mode}")

    conditions = args.conditions or defaults["conditions"]
    pairings = args.pairings or defaults["pairings"]
    topics = args.topics or defaults["topics"]
    repetitions = args.repetitions or defaults["repetitions"]
    num_turns = args.num_turns
    num_episodes = args.episodes
    seed = args.seed
    dry_run = args.mode == "dry-run"
    verbose = args.verbose
    n_surrogates = args.n_surrogates
    judge_model = DIALOGUE_CONFIG["judge_model"]
    max_tokens = DIALOGUE_CONFIG["max_tokens"]

    # Results directory
    if args.results_dir:
        results_dir = Path(args.results_dir)
    elif args.mode == "full":
        results_dir = Path("results_full")
    else:
        results_dir = Path("results_pilot")
    results_dir.mkdir(exist_ok=True)

    # Archive if requested
    if getattr(args, "archive", None):
        archive_results(results_dir, args.archive)

    phase = getattr(args, "phase", "all")

    # Phase routing for batch operations (no manifest/dialogue generation needed)
    if phase == "batch-score":
        from judge_turns import batch_score_dialogues
        batch_score_dialogues(results_dir, judge_model, verbose=verbose, n_surrogates=n_surrogates)
        return

    if phase == "compute-metrics":
        _recompute_metrics(results_dir, verbose=verbose, n_surrogates=n_surrogates)
        return

    # Check API availability
    needed_models = set()
    for p in pairings:
        a, b = MODEL_PAIRINGS[p]
        needed_models.add(a)
        needed_models.add(b)
    if phase != "generate":
        needed_models.add(judge_model)  # Judge not needed in generate-only phase

    if not dry_run:
        for m in needed_models:
            if not check_api_available(m):
                print(f"  ✗ API key not found for {m} ({MODELS[m]['provider']})")
                print(f"    Set {MODELS[m]['provider'].upper()}_API_KEY environment variable")
                sys.exit(1)

    # Build manifest
    manifest = build_manifest(conditions, pairings, topics, repetitions, seed)

    phase_label = f" | Phase: {phase}" if phase != "all" else ""
    print(f"\n  Experiment 13: The Coupled Oscillator")
    print(f"  Mode: {args.mode} | Dry-run: {dry_run}{phase_label}")
    print(f"  Conditions: {conditions}")
    print(f"  Pairings: {pairings}")
    print(f"  Topics: {topics}")
    print(f"  Repetitions: {repetitions}")
    print(f"  Turns: {num_turns} | Episodes: {num_episodes}")
    print(f"  Total dialogues: {len(manifest)}")
    print(f"  Results: {results_dir}/")
    print()

    completed = 0
    skipped = 0
    failed = 0

    for i, (condition, pairing, topic, rep) in enumerate(manifest):
        dialogue_id = build_dialogue_id(condition, pairing, topic, rep)

        # Resume check — in generate phase, also skip if _raw.json exists
        include_raw = (phase == "generate")
        if not args.no_resume and dialogue_exists(dialogue_id, results_dir, include_raw=include_raw):
            skipped += 1
            if verbose:
                print(f"  [{i+1}/{len(manifest)}] SKIP {dialogue_id} (exists)")
            continue

        print(f"  [{i+1}/{len(manifest)}] Running {dialogue_id}...")

        try:
            if num_episodes > 1:
                series = run_episode_series(
                    condition=condition,
                    pairing=pairing,
                    topic_key=topic,
                    repetition=rep,
                    seed=seed + i,
                    num_episodes=num_episodes,
                    num_turns=num_turns,
                    max_tokens=max_tokens,
                    dry_run=dry_run,
                    verbose=verbose,
                    results_dir=results_dir,
                    judge_model=judge_model,
                    n_surrogates=n_surrogates,
                )
                filepath = save_episode_series(series, results_dir)
            else:
                result = run_single_dialogue(
                    condition=condition,
                    pairing=pairing,
                    topic_key=topic,
                    repetition=rep,
                    seed=seed + i,
                    num_turns=num_turns,
                    max_tokens=max_tokens,
                    dry_run=dry_run,
                    verbose=verbose,
                    results_dir=results_dir,
                    judge_model=judge_model,
                    n_surrogates=n_surrogates,
                    skip_scoring=(phase == "generate"),
                )
                if phase == "generate":
                    filepath = save_raw_dialogue(result, results_dir)
                else:
                    filepath = save_dialogue(result, results_dir)

            completed += 1
            print(f"    ✓ Saved: {filepath.name}")

        except Exception as e:
            failed += 1
            print(f"    ✗ FAILED: {e}")
            if verbose:
                import traceback
                traceback.print_exc()

    print(f"\n  ──────────────────────────────")
    print(f"  Completed: {completed}")
    print(f"  Skipped (resume): {skipped}")
    print(f"  Failed: {failed}")
    print(f"  Total: {completed + skipped + failed}/{len(manifest)}")

    # Auto-run analysis if all complete (skip for generate-only phase)
    if phase != "generate" and completed + skipped == len(manifest) and failed == 0:
        print(f"\n  All dialogues complete. Running analysis...")
        from analyze_exp13 import run_analysis
        run_analysis(results_dir, verbose=verbose)
    elif phase == "generate" and completed > 0:
        print(f"\n  Phase 1 complete. Run --phase batch-score to score these dialogues.")

    # Auto-run kinship analysis
    if args.mode == "kinship" and phase != "generate" and completed + skipped == len(manifest) and failed == 0:
        print(f"\n  All kinship dialogues complete. Running kinship analysis...")
        from analyze_kinship import run_kinship_analysis
        run_kinship_analysis(results_dir, verbose=verbose)


# ──────────────────────────────────────────────────
# Archive
# ──────────────────────────────────────────────────

def archive_results(results_dir: Path, label: str = "") -> Path:
    """Archive a results directory with timestamp and optional label.

    Creates a full copy of the results directory with a timestamped name.
    Returns the path to the archive directory.
    """
    import shutil

    if not results_dir.exists():
        print(f"  ⚠ Results directory {results_dir} does not exist. Nothing to archive.")
        return None

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    suffix = f"_{label}" if label else ""
    archive_name = f"{results_dir.name}_archive_{ts}{suffix}"
    archive_dir = results_dir.parent / archive_name

    shutil.copytree(results_dir, archive_dir)
    n_files = len(list(archive_dir.glob("*.json")))
    print(f"  ✓ Archived {n_files} files → {archive_dir}/")
    return archive_dir


# ──────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Experiment 13: The Coupled Oscillator — Resonance Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--mode", choices=["pilot", "full", "analyze-only", "dry-run", "kinship"],
        default="pilot",
        help="pilot: N=10, C/A/E, Claude-Claude, T4 | "
             "full: all conditions/pairings/topics | "
             "kinship: Kinship Test (A+C, all pairings, T3+T4) | "
             "analyze-only: run analysis on existing data | "
             "dry-run: synthetic data, zero API calls",
    )

    # Fine-grained overrides
    parser.add_argument("--conditions", nargs="+", default=None,
                        help="Override conditions (e.g. A C E)")
    parser.add_argument("--pairings", nargs="+", default=None,
                        help="Override pairings (e.g. claude_claude gpt4o_gpt4o)")
    parser.add_argument("--topics", nargs="+", default=None,
                        help="Override topics (e.g. T4 T1)")
    parser.add_argument("--repetitions", type=int, default=None,
                        help="Repetitions per cell")
    parser.add_argument("--num-turns", type=int, default=8,
                        help="Turns per dialogue (default: 8)")
    parser.add_argument("--episodes", type=int, default=1,
                        help="Number of episodes per dialogue (Op 8 arm, default: 1)")

    # Phase control (for 2-phase batch scoring)
    parser.add_argument(
        "--phase", choices=["all", "generate", "batch-score", "compute-metrics"],
        default="all",
        help="all: generate+score+metrics (default) | generate: Phase 1 only (save raw) | "
             "batch-score: Phase 2 (batch score raw files) | compute-metrics: recompute coupling",
    )

    # Archive
    parser.add_argument("--archive", metavar="LABEL", nargs="?", const="backup",
                        help="Archive results directory before running (optional label)")

    # Options
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-dir", type=str, default=None)
    parser.add_argument("--no-resume", action="store_true",
                        help="Don't skip existing dialogues")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--n-surrogates", type=int, default=1000,
                        help="Number of permutation surrogates (default: 1000)")

    args = parser.parse_args()
    run_experiment(args)


if __name__ == "__main__":
    main()
