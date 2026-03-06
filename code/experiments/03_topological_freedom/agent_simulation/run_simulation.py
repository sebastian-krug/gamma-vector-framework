#!/usr/bin/env python3
"""
Topological Freedom Test v2.0 — Unified Runner

Fuehrt Experimente aus der Registry aus und speichert Ergebnisse.
Siehe EXPERIMENT_PROTOCOL.md fuer Details.

Beispiele:
  python run_simulation.py --list
  python run_simulation.py --experiment exp10b_movement_harsh
  python run_simulation.py --experiment sens_k1_20 --runs 30
  python run_simulation.py --experiment exp10b_movement_harsh --no-progress --log-level DEBUG
"""

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Sicherstellen dass src und experiments importierbar sind
sys.path.insert(0, str(Path(__file__).parent))

from src.simulation import Simulation
from src.metrics import MetricsCollector
from src.config import GRID_SIZE, AGENTS_PER_TYPE, TYPE_A, TYPE_B, TYPE_C
from experiments.experiment_config import EXPERIMENTS, list_experiments, get_experiment

logger = logging.getLogger('topological_freedom')


def setup_logging(level_str):
    """Logging konfigurieren."""
    level = getattr(logging, level_str.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format='%(asctime)s [%(levelname)s] %(message)s',
        datefmt='%H:%M:%S',
    )


def run_single(seed, params, ticks, progress=True):
    """
    Einen einzelnen Simulationslauf durchfuehren.
    Returns: (survival_rates, cluster_metrics, metrics_collector)
    """
    sim = Simulation(seed=seed, params=params)
    metrics = sim.run(ticks=ticks, progress=progress)

    survival = sim.get_survival_rates()

    # Cluster-Metriken am Ende der Simulation messen
    cluster_metrics = metrics.measure_clusters(sim.agents, grid_size=GRID_SIZE)

    return survival, cluster_metrics, metrics


def run_experiment(exp_name, exp_config, output_dir, runs=None, seed_start=1,
                   ticks=2000, progress=True, save_timeseries=True):
    """
    Ein vollstaendiges Experiment durchfuehren.
    Returns: dict mit aggregierten Ergebnissen
    """
    params = exp_config['params']
    default_runs = exp_config.get('runs', 20)
    num_runs = runs if runs is not None else default_runs
    group = exp_config.get('group', '?')
    description = exp_config.get('description', '')

    # Experiment-Verzeichnis
    exp_dir = output_dir / exp_name
    exp_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'=' * 70}")
    print(f"  EXPERIMENT: {exp_name}  (Gruppe {group})")
    print(f"  {description}")
    print(f"  Runs: {num_runs}, Ticks: {ticks}, Seed-Start: {seed_start}")
    print(f"  Key params: K1={params.get('K1', 10)}, "
          f"Movement={'ON' if params.get('MOVEMENT_ENABLED') else 'OFF'}, "
          f"Policy={'V2' if params.get('POLICY_V2') else 'V1'}")
    print(f"{'=' * 70}")

    results = []
    t_start = time.time()

    for i in range(num_runs):
        seed = seed_start + i
        run_num = i + 1

        t_run_start = time.time()

        # Tick-Level tqdm nur bei einzelnem Run oder mit Progress
        show_tqdm = progress and num_runs <= 3
        survival, cluster, metrics = run_single(
            seed=seed,
            params=params,
            ticks=ticks,
            progress=show_tqdm,
        )

        t_run = time.time() - t_run_start

        # Zeitreihe speichern
        if save_timeseries:
            ts_path = exp_dir / f"run_{run_num:03d}.csv"
            metrics.save(ts_path)

        # Summary-Zeile
        row = {
            'seed': seed,
            'survival_A': survival[TYPE_A],
            'survival_B': survival[TYPE_B],
            'survival_C': survival[TYPE_C],
        }
        row.update(cluster)
        results.append(row)

        # Kompakte einzeilige Fortschrittsanzeige
        h1 = survival[TYPE_C] > survival[TYPE_A] > survival[TYPE_B]
        c_gt_b = survival[TYPE_C] > survival[TYPE_B]
        elapsed_so_far = time.time() - t_start
        eta = (elapsed_so_far / run_num) * (num_runs - run_num)
        flags = f"{'H1 ' if h1 else ''}{'C>B' if c_gt_b else ''}"

        print(
            f"  [{run_num:>{len(str(num_runs))}}/{num_runs}] "
            f"seed={seed:<4} "
            f"A={survival[TYPE_A]:.0%} B={survival[TYPE_B]:.0%} C={survival[TYPE_C]:.0%}  "
            f"Cl_C={cluster.get('avg_cluster_C', 0):.1f}  "
            f"{flags:<6} "
            f"({t_run:.1f}s, ETA {eta:.0f}s)",
            flush=True,
        )

    elapsed = time.time() - t_start

    # Summary speichern
    summary_df = pd.DataFrame(results)
    summary_df.to_csv(exp_dir / "summary.csv", index=False)

    # Aggregierte Ergebnisse berechnen
    agg = _compute_aggregates(summary_df, num_runs)

    # experiment_meta.json schreiben
    meta = {
        'experiment_name': exp_name,
        'group': group,
        'description': description,
        'params': _serialize_params(params),
        'num_runs': num_runs,
        'ticks': ticks,
        'seed_start': seed_start,
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': round(elapsed, 1),
        'results_summary': agg,
    }
    with open(exp_dir / "experiment_meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    # Ergebnis-Zusammenfassung ausgeben
    _print_summary(exp_name, agg, elapsed)

    return agg


def _compute_aggregates(df, num_runs):
    """Aggregierte Statistiken aus summary DataFrame."""
    agg = {}

    for typ in ['A', 'B', 'C']:
        col = f'survival_{typ}'
        agg[f'survival_{typ}_mean'] = round(df[col].mean(), 4)
        agg[f'survival_{typ}_std'] = round(df[col].std(), 4)
        agg[f'survival_{typ}_min'] = round(df[col].min(), 4)
        agg[f'survival_{typ}_max'] = round(df[col].max(), 4)

    # Cluster-Metriken (falls vorhanden)
    for typ in ['A', 'B', 'C']:
        avg_col = f'avg_cluster_{typ}'
        max_col = f'max_cluster_{typ}'
        if avg_col in df.columns:
            agg[f'avg_cluster_{typ}_mean'] = round(df[avg_col].mean(), 2)
            agg[f'max_cluster_{typ}_mean'] = round(df[max_col].mean(), 2)

    # H1 und Kipp-Punkt
    h1_passes = int(sum(
        (df['survival_C'] > df['survival_A']) & (df['survival_A'] > df['survival_B'])
    ))
    c_beats_b = int(sum(df['survival_C'] > df['survival_B']))

    agg['num_runs'] = num_runs
    agg['h1_passes'] = h1_passes
    agg['h1_rate'] = round(h1_passes / num_runs, 3)
    agg['c_beats_b'] = c_beats_b
    agg['c_beats_b_rate'] = round(c_beats_b / num_runs, 3)
    agg['kipp_punkt'] = c_beats_b >= 0.7 * num_runs

    return agg


def _serialize_params(params):
    """Macht params JSON-serialisierbar (dicts mit int-keys etc.)."""
    result = {}
    for k, v in params.items():
        if isinstance(v, dict):
            result[k] = {str(k2): v2 for k2, v2 in v.items()}
        elif isinstance(v, (np.integer,)):
            result[k] = int(v)
        elif isinstance(v, (np.floating,)):
            result[k] = float(v)
        else:
            result[k] = v
    return result


def _print_summary(exp_name, agg, elapsed):
    """Formatierte Zusammenfassung eines Experiments."""
    print(f"\n  {'─' * 60}")
    print(f"  ERGEBNIS: {exp_name}")
    print(f"  {'─' * 60}")
    print(f"  Survival A: {agg['survival_A_mean']:.1%} +/- {agg['survival_A_std']:.1%}")
    print(f"  Survival B: {agg['survival_B_mean']:.1%} +/- {agg['survival_B_std']:.1%}")
    print(f"  Survival C: {agg['survival_C_mean']:.1%} +/- {agg['survival_C_std']:.1%}")

    if 'avg_cluster_C_mean' in agg:
        print(f"  Cluster C (avg): {agg['avg_cluster_C_mean']:.1f}")
        print(f"  Cluster C (max): {agg['max_cluster_C_mean']:.1f}")

    total = agg['num_runs']
    print(f"  H1 (C>A>B): {agg['h1_passes']}/{total}")
    print(f"  C>B: {agg['c_beats_b']}/{total}")
    kipp = agg['kipp_punkt']
    print(f"  Kipp-Punkt: {'JA' if kipp else 'NEIN'}")
    print(f"  Dauer: {elapsed:.1f}s")
    print(f"  {'─' * 60}")


def main():
    parser = argparse.ArgumentParser(
        description='Topological Freedom Test v2.0 — Unified Runner',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Beispiele:
  python run_simulation.py --list
  python run_simulation.py --experiment exp10b_movement_harsh
  python run_simulation.py --experiment sens_k1_20 --runs 30
  python run_simulation.py --experiment exp10b_movement_harsh --no-progress
        """,
    )

    parser.add_argument('--experiment', type=str,
                        help='Experiment aus Registry (PFLICHT, ausser --list)')
    parser.add_argument('--runs', type=int, default=None,
                        help='Anzahl Runs (Default: aus Registry)')
    parser.add_argument('--seed-start', type=int, default=1,
                        help='Start-Seed (Default: 1)')
    parser.add_argument('--output-dir', type=str, default='data',
                        help='Ausgabeverzeichnis (Default: data/)')
    parser.add_argument('--ticks', type=int, default=2000,
                        help='Simulationsticks pro Run (Default: 2000)')
    parser.add_argument('--list', action='store_true',
                        help='Alle Experimente auflisten und beenden')
    parser.add_argument('--no-progress', action='store_true',
                        help='Progress-Bars deaktivieren')
    parser.add_argument('--log-level', type=str, default='INFO',
                        choices=['DEBUG', 'INFO', 'WARNING'],
                        help='Log-Level (Default: INFO)')
    parser.add_argument('--no-timeseries', action='store_true',
                        help='Keine Zeitreihen-CSVs speichern')

    args = parser.parse_args()

    setup_logging(args.log_level)

    # --list: Alle Experimente anzeigen
    if args.list:
        print("\n  Topological Freedom Test v2.0 — Experiment-Registry")
        print(f"  {'=' * 60}")
        list_experiments()
        return

    # --experiment: Pflicht (wenn nicht --list)
    if not args.experiment:
        parser.error("--experiment ist erforderlich (oder --list zum Auflisten)")

    # Experiment laden
    try:
        exp_config = get_experiment(args.experiment)
    except ValueError as e:
        parser.error(str(e))
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    progress = not args.no_progress
    save_ts = not args.no_timeseries

    print(f"\n  Topological Freedom Test v2.0")
    print(f"  Output: {output_dir.resolve()}")

    run_experiment(
        exp_name=args.experiment,
        exp_config=exp_config,
        output_dir=output_dir,
        runs=args.runs,
        seed_start=args.seed_start,
        ticks=args.ticks,
        progress=progress,
        save_timeseries=save_ts,
    )

    print("\nFertig.")


if __name__ == '__main__':
    main()
