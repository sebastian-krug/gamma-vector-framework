"""
Experiment-Registry fuer Topological Freedom Test v2.0
17 Experimente in 4 Gruppen (A-D), 340 Runs total.

Jedes Experiment erbt von BASELINE und ueberschreibt spezifische Parameter.
"""

# ============================================================
# Baseline: Gemeinsame Defaults fuer alle Experimente
# ============================================================
BASELINE = {
    # Environment
    'LAMBDA': 0.5,
    'DELTA': 0.05,
    'MU': 0.1,
    # Stealing
    'K1': 10,
    'C_STEAL': 1,
    'STEAL_RATE': 5,
    # Features
    'SELECTIVE_SHARE': False,
    'POLICY_V2': False,
    'RGS_ENABLED': False,
    'DISSONANCE_PENALTY': False,
    'DYNAMIC_SYNERGY': False,
    # Movement
    'MOVEMENT_ENABLED': False,
    'MOVE_COST': 2,
    'MOVE_COOLDOWN': 3,
    'MOVE_ENERGY_THRESHOLD': 10,
    'MOVE_SCORES': {
        'A': {'A': -2, 'B': +5, 'C': -1},
        'B': {'A': -3, 'B': +1, 'C': +1},
        'C': {'A': -10, 'B': +2, 'C': +5},
    },
}


def _merge(overrides):
    """Merged overrides in eine Kopie der Baseline."""
    params = dict(BASELINE)
    params.update(overrides)
    return params


# ============================================================
# Gruppe A: Core Exp10 (Reproduktion + Kontrolle) — 4 x 20 = 80 Runs
# ============================================================

EXPERIMENTS = {}

# --- exp10a: Movement + moderate Kosten ---
EXPERIMENTS['exp10a_movement_only'] = {
    'params': _merge({
        'MOVEMENT_ENABLED': True,
        'K1': 15,
        'C_STEAL': 2,
        'STEAL_RATE': 4,
        'POLICY_V2': True,
    }),
    'runs': 20,
    'group': 'A',
    'description': 'Movement + moderate Steal-Kosten (K1=15). Erwartet: C > B in 70%+ Runs.',
}

# --- exp10b: Movement + harte Kosten (KIPP-PUNKT) ---
EXPERIMENTS['exp10b_movement_harsh'] = {
    'params': _merge({
        'MOVEMENT_ENABLED': True,
        'K1': 30,
        'C_STEAL': 5,
        'STEAL_RATE': 3,
        'POLICY_V2': True,
    }),
    'runs': 20,
    'group': 'A',
    'description': 'Movement + harte Steal-Kosten (K1=30). KIPP-PUNKT. Erwartet: C >> B.',
}

# --- exp10c: Movement + Full Features (inkl. RGS) ---
EXPERIMENTS['exp10c_movement_full'] = {
    'params': _merge({
        'MOVEMENT_ENABLED': True,
        'K1': 15,
        'C_STEAL': 2,
        'STEAL_RATE': 4,
        'POLICY_V2': True,
        'RGS_ENABLED': True,
    }),
    'runs': 20,
    'group': 'A',
    'description': 'Movement + Policy V2 + RGS. Erwartet: RGS verstaerkt C-Vorteil.',
}

# --- exp10d: Statische Kontrolle (kein Movement) ---
EXPERIMENTS['exp10d_static_control'] = {
    'params': _merge({
        'MOVEMENT_ENABLED': False,
        'K1': 15,
        'C_STEAL': 2,
        'STEAL_RATE': 4,
        'POLICY_V2': True,
    }),
    'runs': 20,
    'group': 'A',
    'description': 'Kein Movement, sonst wie exp10a. Erwartet: A dominiert.',
}


# ============================================================
# Gruppe B: K1 Sensitivity Sweep — 5 x 20 = 100 Runs
# ============================================================

for k1_val, c_steal, steal_rate in [
    (10, 1, 5),
    (15, 2, 4),
    (20, 3, 4),
    (25, 4, 3),
    (30, 5, 3),
]:
    name = f'sens_k1_{k1_val}'
    EXPERIMENTS[name] = {
        'params': _merge({
            'MOVEMENT_ENABLED': True,
            'K1': k1_val,
            'C_STEAL': c_steal,
            'STEAL_RATE': steal_rate,
            'POLICY_V2': True,
        }),
        'runs': 10,
        'group': 'B',
        'description': f'K1-Sensitivity: K1={k1_val}, C_STEAL={c_steal}, STEAL_RATE={steal_rate}.',
    }


# ============================================================
# Gruppe C: Movement Score Sensitivity (C-A Repulsion) — 4 x 20 = 80 Runs
# ============================================================

for ca_score in [-3, -5, -7, -10]:
    name = f'sens_move_ca_{abs(ca_score)}'
    custom_scores = {
        'A': {'A': -2, 'B': +5, 'C': -1},
        'B': {'A': -3, 'B': +1, 'C': +1},
        'C': {'A': ca_score, 'B': +2, 'C': +5},
    }
    EXPERIMENTS[name] = {
        'params': _merge({
            'MOVEMENT_ENABLED': True,
            'K1': 30,
            'C_STEAL': 5,
            'STEAL_RATE': 3,
            'POLICY_V2': True,
            'MOVE_SCORES': custom_scores,
        }),
        'runs': 10,
        'group': 'C',
        'description': f'C-A Repulsion = {ca_score}. Exp10b-Regime (K1=30).',
    }


# ============================================================
# Gruppe D: Ablation Studies — 4 x 20 = 80 Runs
# ============================================================

# ablation_movement_only: Movement OHNE Policy V2
EXPERIMENTS['ablation_movement_only'] = {
    'params': _merge({
        'MOVEMENT_ENABLED': True,
        'K1': 30,
        'C_STEAL': 5,
        'STEAL_RATE': 3,
        'POLICY_V2': False,  # Kein Trust
    }),
    'runs': 20,
    'group': 'D',
    'description': 'Movement OHNE Policy V2. Erwartet: Movement allein reicht nicht.',
}

# ablation_movement_trust: Movement MIT Policy V2
EXPERIMENTS['ablation_movement_trust'] = {
    'params': _merge({
        'MOVEMENT_ENABLED': True,
        'K1': 30,
        'C_STEAL': 5,
        'STEAL_RATE': 3,
        'POLICY_V2': True,  # Mit Trust
    }),
    'runs': 20,
    'group': 'D',
    'description': 'Movement + Policy V2. Erwartet: Kipp-Punkt erreicht.',
}

# ablation_movement_rgs: Movement + Policy V2 + RGS
EXPERIMENTS['ablation_movement_rgs'] = {
    'params': _merge({
        'MOVEMENT_ENABLED': True,
        'K1': 30,
        'C_STEAL': 5,
        'STEAL_RATE': 3,
        'POLICY_V2': True,
        'RGS_ENABLED': True,
        'DISSONANCE_PENALTY': True,
    }),
    'runs': 20,
    'group': 'D',
    'description': 'Movement + Trust + RGS + Dissonance. Erwartet: Verstaerkt, nicht notwendig.',
}

# ablation_static_trust: Kein Movement, MIT Policy V2
EXPERIMENTS['ablation_static_trust'] = {
    'params': _merge({
        'MOVEMENT_ENABLED': False,
        'K1': 30,
        'C_STEAL': 5,
        'STEAL_RATE': 3,
        'POLICY_V2': True,
    }),
    'runs': 20,
    'group': 'D',
    'description': 'Kein Movement, aber Policy V2. Erwartet: Trust ohne Topologie wirkungslos.',
}


# ============================================================
# Hilfsfunktionen
# ============================================================

def list_experiments():
    """Gibt eine formatierte Uebersicht aller Experimente aus."""
    groups = {}
    for name, cfg in EXPERIMENTS.items():
        g = cfg['group']
        if g not in groups:
            groups[g] = []
        groups[g].append((name, cfg))

    group_labels = {
        'A': 'Core Exp10 (Reproduktion)',
        'B': 'K1 Sensitivity Sweep',
        'C': 'Movement Score Sensitivity',
        'D': 'Ablation Studies',
    }

    total_runs = 0
    for g in sorted(groups.keys()):
        label = group_labels.get(g, g)
        print(f"\n  Gruppe {g}: {label}")
        print(f"  {'─' * 60}")
        for name, cfg in groups[g]:
            runs = cfg['runs']
            total_runs += runs
            p = cfg['params']
            move = 'Move' if p.get('MOVEMENT_ENABLED') else 'Static'
            pv2 = 'V2' if p.get('POLICY_V2') else 'V1'
            k1 = p.get('K1', 10)
            print(f"    {name:<30} {runs:>3} runs  K1={k1:<3} {move:<7} {pv2}")

    print(f"\n  Total: {len(EXPERIMENTS)} Experimente, {total_runs} Runs")


def get_experiment(name):
    """Gibt die Konfiguration eines Experiments zurueck."""
    if name not in EXPERIMENTS:
        available = ', '.join(sorted(EXPERIMENTS.keys()))
        raise ValueError(f"Unbekanntes Experiment: '{name}'. Verfuegbar: {available}")
    return EXPERIMENTS[name]
