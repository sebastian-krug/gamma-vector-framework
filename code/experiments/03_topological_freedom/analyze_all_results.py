#!/usr/bin/env python3
"""
Topological Freedom Test v2 — Complete Analysis of all 17 Experiments (340 Runs)
Generates publication-quality visualizations and statistical comparisons.
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from scipy.stats import mannwhitneyu

# ── Setup ──────────────────────────────────────────────────────────────
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.titleweight': 'bold',
    'figure.facecolor': 'white',
})

DATA_DIR = Path("agent_simulation/data")
OUT_DIR = Path("analysis_output")
OUT_DIR.mkdir(exist_ok=True)

# Color palette - consistent across all charts
COL_A = '#e74c3c'   # Tyrant - red
COL_B = '#3498db'   # Martyr - blue
COL_C = '#2ecc71'   # Unity  - green
COL_KIPP = '#f39c12' # Kipp-Punkt marker

# ── Load all experiments ───────────────────────────────────────────────
def load_experiment(name):
    meta_path = DATA_DIR / name / "experiment_meta.json"
    summary_path = DATA_DIR / name / "summary.csv"
    with open(meta_path) as f:
        meta = json.load(f)
    df = pd.read_csv(summary_path)
    return meta, df

experiments = {}
for d in sorted(DATA_DIR.iterdir()):
    if d.is_dir() and (d / "experiment_meta.json").exists():
        meta, df = load_experiment(d.name)
        experiments[d.name] = {'meta': meta, 'df': df}

print(f"Loaded {len(experiments)} experiments")
for name, data in experiments.items():
    m = data['meta']['results_summary']
    kipp = "✓ KIPP" if m['kipp_punkt'] else "  —"
    print(f"  {name:<30} A={m['survival_A_mean']:.1%} B={m['survival_B_mean']:.1%} "
          f"C={m['survival_C_mean']:.1%}  C>B={m['c_beats_b']}/{m['num_runs']}  {kipp}")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 1: Group A — Core Exp10 Overview (4 panels)
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 4, figsize=(16, 5), sharey=True)
fig.suptitle('Gruppe A: Core Exp10 — Survival nach Bedingung', fontsize=15, fontweight='bold', y=1.02)

group_a = ['exp10a_movement_only', 'exp10b_movement_harsh', 'exp10c_movement_full', 'exp10d_static_control']
labels_a = ['10a\nMove+Mod', '10b\nMove+Harsh', '10c\nMove+Full', '10d\nStatic']

for i, (name, label) in enumerate(zip(group_a, labels_a)):
    ax = axes[i]
    m = experiments[name]['meta']['results_summary']

    vals = [m['survival_A_mean'], m['survival_B_mean'], m['survival_C_mean']]
    stds = [m['survival_A_std'], m['survival_B_std'], m['survival_C_std']]
    colors = [COL_A, COL_B, COL_C]
    types = ['A\nTyrant', 'B\nMartyr', 'C\nUnity']

    bars = ax.bar(types, vals, yerr=stds, color=colors, capsize=4, alpha=0.85, edgecolor='white', linewidth=0.5)

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{val:.0%}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    kipp = m['kipp_punkt']
    ax.set_title(label, fontsize=11)
    if kipp:
        ax.text(0.95, 0.95, 'KIPP ✓', transform=ax.transAxes, ha='right', va='top',
                fontsize=9, color=COL_KIPP, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#fef9e7', edgecolor=COL_KIPP, alpha=0.9))

    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

axes[0].set_ylabel('Survival Rate', fontsize=12)
plt.tight_layout()
plt.savefig(OUT_DIR / 'fig1_group_a_core.png', dpi=150, bbox_inches='tight')
plt.close()
print("\n✓ Figure 1 saved: Group A Core Exp10")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 2: Group B — K1 Sensitivity Sweep
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(10, 6))

k1_values = [10, 15, 20, 25, 30]
surv_a, surv_b, surv_c = [], [], []
std_a, std_b, std_c = [], [], []

for k1 in k1_values:
    m = experiments[f'sens_k1_{k1}']['meta']['results_summary']
    surv_a.append(m['survival_A_mean']); std_a.append(m['survival_A_std'])
    surv_b.append(m['survival_B_mean']); std_b.append(m['survival_B_std'])
    surv_c.append(m['survival_C_mean']); std_c.append(m['survival_C_std'])

ax.errorbar(k1_values, surv_a, yerr=std_a, marker='s', capsize=5, color=COL_A,
            linewidth=2, markersize=8, label='A (Tyrant)')
ax.errorbar(k1_values, surv_b, yerr=std_b, marker='^', capsize=5, color=COL_B,
            linewidth=2, markersize=8, label='B (Martyr)')
ax.errorbar(k1_values, surv_c, yerr=std_c, marker='o', capsize=5, color=COL_C,
            linewidth=2, markersize=8, label='C (Unity)')

# Mark the crossover zone
ax.axvspan(17, 23, alpha=0.1, color=COL_KIPP, label='Kritischer Bereich')
ax.axhline(y=0.5, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)

ax.set_title('K1 Sensitivity: Steal-Kosten bestimmen das Ranking', fontsize=14, fontweight='bold')
ax.set_xlabel('K1 (Steal-Entropie-Kosten)', fontsize=12)
ax.set_ylabel('Survival Rate', fontsize=12)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_xticks(k1_values)
ax.set_ylim(0, 1.0)
ax.legend(loc='center left', frameon=True, framealpha=0.9)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Annotate crossover
ax.annotate('A↔B Crossover\n(K1 ≈ 20)', xy=(20, 0.56), xytext=(22, 0.85),
            fontsize=9, ha='center', color='#7f8c8d',
            arrowprops=dict(arrowstyle='->', color='#7f8c8d', lw=1.2))

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig2_k1_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 2 saved: K1 Sensitivity")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 3: Group C — Movement Score (C-A Repulsion) Sensitivity
# ═══════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5))
fig.suptitle('Gruppe C: C-A Repulsion — Stärke der Fluchtreaktion', fontsize=14, fontweight='bold', y=1.02)

ca_values = [3, 5, 7, 10]
surv_c_ca, std_c_ca, cluster_c_ca, cb_rate = [], [], [], []

for ca in ca_values:
    m = experiments[f'sens_move_ca_{ca}']['meta']['results_summary']
    surv_c_ca.append(m['survival_C_mean'])
    std_c_ca.append(m['survival_C_std'])
    cluster_c_ca.append(m.get('avg_cluster_C_mean', 0))
    cb_rate.append(m['c_beats_b_rate'])

# Left panel: Survival C vs Repulsion
ax1.errorbar(ca_values, surv_c_ca, yerr=std_c_ca, marker='o', capsize=5, color=COL_C,
             linewidth=2.5, markersize=10, label='C Survival')
ax1.fill_between(ca_values, [s-e for s,e in zip(surv_c_ca, std_c_ca)],
                 [s+e for s,e in zip(surv_c_ca, std_c_ca)], alpha=0.15, color=COL_C)
ax1.set_xlabel('|C→A Repulsion Score|', fontsize=12)
ax1.set_ylabel('Type-C Survival Rate', fontsize=12)
ax1.set_title('C-Survival steigt mit Repulsion', fontsize=12)
ax1.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax1.set_xticks(ca_values)
ax1.set_ylim(0.55, 0.85)
ax1.spines['top'].set_visible(False)
ax1.spines['right'].set_visible(False)

# Annotate monotonic trend
for i, (ca, sv) in enumerate(zip(ca_values, surv_c_ca)):
    ax1.annotate(f'{sv:.0%}', (ca, sv), textcoords="offset points",
                 xytext=(0, 12), ha='center', fontsize=9, fontweight='bold', color=COL_C)

# Right panel: Cluster Size vs Repulsion
ax2.bar(ca_values, cluster_c_ca, color=COL_C, alpha=0.7, width=1.5, edgecolor='white')
ax2.set_xlabel('|C→A Repulsion Score|', fontsize=12)
ax2.set_ylabel('Avg Cluster Size (C)', fontsize=12)
ax2.set_title('Cluster-Bildung nimmt zu', fontsize=12)
ax2.set_xticks(ca_values)
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

for i, (ca, cl) in enumerate(zip(ca_values, cluster_c_ca)):
    ax2.text(ca, cl + 0.08, f'{cl:.1f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig3_movement_sensitivity.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 3 saved: Movement Score Sensitivity")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 4: Group D — Ablation Studies (the key chart!)
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(11, 6))

ablation_names = ['ablation_movement_only', 'ablation_movement_trust',
                  'ablation_movement_rgs', 'ablation_static_trust']
ablation_labels = ['Movement\nonly', 'Movement\n+ Trust', 'Movement\n+ Trust + RGS', 'Static\n+ Trust']

x = np.arange(len(ablation_names))
width = 0.22

for i, (name, label) in enumerate(zip(ablation_names, ablation_labels)):
    m = experiments[name]['meta']['results_summary']
    ax.bar(x[i] - width, m['survival_A_mean'], width, color=COL_A, alpha=0.85, edgecolor='white')
    ax.bar(x[i], m['survival_B_mean'], width, color=COL_B, alpha=0.85, edgecolor='white')
    ax.bar(x[i] + width, m['survival_C_mean'], width, color=COL_C, alpha=0.85, edgecolor='white')

    # Values on top
    ax.text(x[i] - width, m['survival_A_mean'] + 0.015, f"{m['survival_A_mean']:.0%}",
            ha='center', fontsize=8, color=COL_A, fontweight='bold')
    ax.text(x[i], m['survival_B_mean'] + 0.015, f"{m['survival_B_mean']:.0%}",
            ha='center', fontsize=8, color=COL_B, fontweight='bold')
    ax.text(x[i] + width, m['survival_C_mean'] + 0.015, f"{m['survival_C_mean']:.0%}",
            ha='center', fontsize=8, color=COL_C, fontweight='bold')

    # Kipp marker
    if m['kipp_punkt']:
        ax.text(x[i], 0.97, 'KIPP ✓', ha='center', fontsize=8, color=COL_KIPP, fontweight='bold')
    else:
        ax.text(x[i], 0.97, '— kein Kipp', ha='center', fontsize=8, color='#95a5a6')

# Legend entries
ax.bar([], [], color=COL_A, alpha=0.85, label='A (Tyrant)')
ax.bar([], [], color=COL_B, alpha=0.85, label='B (Martyr)')
ax.bar([], [], color=COL_C, alpha=0.85, label='C (Unity)')

ax.set_title('Ablation: Movement + Trust sind beide notwendig', fontsize=14, fontweight='bold')
ax.set_ylabel('Survival Rate', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(ablation_labels, fontsize=10)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_ylim(0, 1.08)
ax.legend(loc='upper left', frameon=True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Draw separating line before static_trust
ax.axvline(x=2.5, color='#bdc3c7', linestyle='--', linewidth=1, alpha=0.7)
ax.text(2.7, 0.02, '← Movement ON | Movement OFF →', fontsize=8, color='#7f8c8d', style='italic')

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig4_ablation.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 4 saved: Ablation Studies")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 5: Grand Summary — 2×2 Interaction Matrix
# ═══════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 2, figsize=(11, 9))
fig.suptitle('Interaktionsstruktur: Movement × Steal-Kosten', fontsize=15, fontweight='bold', y=1.01)

# Define the 4 conditions
conditions = [
    ('exp10d_static_control', 'Static + Moderate (K1=15)', axes[0, 0]),
    ('ablation_static_trust', 'Static + Harsh (K1=30)', axes[0, 1]),
    ('exp10a_movement_only', 'Movement + Moderate (K1=15)', axes[1, 0]),
    ('exp10b_movement_harsh', 'Movement + Harsh (K1=30)', axes[1, 1]),
]

for name, title, ax in conditions:
    m = experiments[name]['meta']['results_summary']
    vals = [m['survival_A_mean'], m['survival_B_mean'], m['survival_C_mean']]
    stds = [m['survival_A_std'], m['survival_B_std'], m['survival_C_std']]
    types = ['A', 'B', 'C']
    colors = [COL_A, COL_B, COL_C]

    bars = ax.bar(types, vals, yerr=stds, color=colors, capsize=4, alpha=0.85, edgecolor='white')

    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.025,
                f'{val:.0%}', ha='center', fontsize=11, fontweight='bold')

    # Build ranking string
    ranking_pairs = sorted(zip(['A', 'B', 'C'], vals), key=lambda x: -x[1])
    ranking_str = ' > '.join([f'{p[0]}' for p in ranking_pairs])

    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Ranking annotation
    kipp = m['kipp_punkt']
    box_color = '#e8f8f5' if kipp else '#fef9e7'
    edge_color = COL_C if kipp else '#95a5a6'
    ax.text(0.5, 0.92, ranking_str, transform=ax.transAxes, ha='center',
            fontsize=12, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor=box_color, edgecolor=edge_color, alpha=0.9))

# Row/column labels
axes[0, 0].set_ylabel('Static\n\nSurvival Rate', fontsize=11)
axes[1, 0].set_ylabel('Movement\n\nSurvival Rate', fontsize=11)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig5_interaction_matrix.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 5 saved: Interaction Matrix")


# ═══════════════════════════════════════════════════════════════════════
# FIGURE 6: Cluster Analysis across all Movement experiments
# ═══════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(12, 5.5))

movement_exps = ['exp10a_movement_only', 'exp10b_movement_harsh', 'exp10c_movement_full',
                 'exp10d_static_control', 'ablation_static_trust']
cluster_labels = ['10a\nMove+Mod', '10b\nMove+Harsh', '10c\nMove+Full',
                  '10d\nStatic Mod', 'Static\nHarsh']

x = np.arange(len(movement_exps))
width = 0.22

for i, (name, label) in enumerate(zip(movement_exps, cluster_labels)):
    m = experiments[name]['meta']['results_summary']
    ca = m.get('avg_cluster_A_mean', 0)
    cb = m.get('avg_cluster_B_mean', 0)
    cc = m.get('avg_cluster_C_mean', 0)

    ax.bar(x[i] - width, ca, width, color=COL_A, alpha=0.85, edgecolor='white')
    ax.bar(x[i], cb, width, color=COL_B, alpha=0.85, edgecolor='white')
    ax.bar(x[i] + width, cc, width, color=COL_C, alpha=0.85, edgecolor='white')

    ax.text(x[i] + width, cc + 0.1, f'{cc:.1f}', ha='center', fontsize=9, fontweight='bold', color=COL_C)

ax.bar([], [], color=COL_A, alpha=0.85, label='A (Tyrant)')
ax.bar([], [], color=COL_B, alpha=0.85, label='B (Martyr)')
ax.bar([], [], color=COL_C, alpha=0.85, label='C (Unity)')

ax.axvline(x=2.5, color='#bdc3c7', linestyle='--', linewidth=1, alpha=0.7)
ax.text(2.7, 0.2, '← Movement | Static →', fontsize=8, color='#7f8c8d', style='italic')

ax.set_title('Cluster-Bildung: Nur C bildet signifikante Cluster (nur bei Movement)', fontsize=13, fontweight='bold')
ax.set_ylabel('Avg Cluster Size', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(cluster_labels)
ax.legend(loc='upper right', frameon=True)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

plt.tight_layout()
plt.savefig(OUT_DIR / 'fig6_cluster_analysis.png', dpi=150, bbox_inches='tight')
plt.close()
print("✓ Figure 6 saved: Cluster Analysis")


# ═══════════════════════════════════════════════════════════════════════
# STATISTICAL TESTS
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  STATISTISCHE TESTS")
print("=" * 70)

# Test 1: exp10b vs exp10d (Movement effect)
df_b = experiments['exp10b_movement_harsh']['df']
df_d = experiments['exp10d_static_control']['df']

u, p = mannwhitneyu(df_b['survival_C'], df_d['survival_C'], alternative='greater')
print(f"\n  [1] Movement-Effekt (10b vs 10d, C-Survival):")
print(f"      10b: {df_b['survival_C'].mean():.1%} ± {df_b['survival_C'].std():.1%}")
print(f"      10d: {df_d['survival_C'].mean():.1%} ± {df_d['survival_C'].std():.1%}")
print(f"      U={u:.0f}, p={p:.6f} {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'}")

# Test 2: ablation_movement_trust vs ablation_static_trust
df_mt = experiments['ablation_movement_trust']['df']
df_st = experiments['ablation_static_trust']['df']

u, p = mannwhitneyu(df_mt['survival_C'], df_st['survival_C'], alternative='greater')
print(f"\n  [2] Movement + Trust vs Static + Trust (C-Survival):")
print(f"      Move+Trust: {df_mt['survival_C'].mean():.1%} ± {df_mt['survival_C'].std():.1%}")
print(f"      Static+Trust: {df_st['survival_C'].mean():.1%} ± {df_st['survival_C'].std():.1%}")
print(f"      U={u:.0f}, p={p:.6f} {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'}")

# Test 3: ablation_movement_only vs ablation_movement_trust (Trust effect)
df_mo = experiments['ablation_movement_only']['df']

u, p = mannwhitneyu(df_mt['survival_C'], df_mo['survival_C'], alternative='greater')
print(f"\n  [3] Trust-Effekt bei Movement (move+trust vs move only, C-Survival):")
print(f"      Move+Trust: {df_mt['survival_C'].mean():.1%} ± {df_mt['survival_C'].std():.1%}")
print(f"      Move only:  {df_mo['survival_C'].mean():.1%} ± {df_mo['survival_C'].std():.1%}")
print(f"      U={u:.0f}, p={p:.6f} {'***' if p<0.001 else '**' if p<0.01 else '*' if p<0.05 else 'n.s.'}")

# Test 4: K1 monotonicity
print(f"\n  [4] K1-Sensitivitaet (C-Survival):")
for k1 in [10, 15, 20, 25, 30]:
    m = experiments[f'sens_k1_{k1}']['meta']['results_summary']
    print(f"      K1={k1}: C={m['survival_C_mean']:.1%}, A={m['survival_A_mean']:.1%}, "
          f"B={m['survival_B_mean']:.1%}  H1={m['h1_passes']}/{m['num_runs']}")

# Test 5: sens_k1_30 vs exp10b consistency check
df_k30 = experiments['sens_k1_30']['df']
u, p = mannwhitneyu(df_b['survival_C'], df_k30['survival_C'])
print(f"\n  [5] Konsistenzcheck: exp10b vs sens_k1_30 (identische Params):")
print(f"      exp10b:    C={df_b['survival_C'].mean():.1%}")
print(f"      sens_k1_30: C={df_k30['survival_C'].mean():.1%}")
print(f"      U={u:.0f}, p={p:.4f} (erwartet: nicht signifikant)")


# ═══════════════════════════════════════════════════════════════════════
# PLAUSIBILITAETS-MATRIX (Protocol §7.3)
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  PLAUSIBILITAETS-MATRIX (Protokoll §7.3)")
print("=" * 70)

checks = [
    ("exp10b vs exp10d: C-Surv exp10b >> exp10d",
     experiments['exp10b_movement_harsh']['meta']['results_summary']['survival_C_mean'] >
     experiments['exp10d_static_control']['meta']['results_summary']['survival_C_mean'] + 0.10),

    ("sens_k1_30 vs sens_k1_10: C-Surv k30 > k10",
     experiments['sens_k1_30']['meta']['results_summary']['survival_C_mean'] >
     experiments['sens_k1_10']['meta']['results_summary']['survival_C_mean']),

    ("sens_move_ca_10 vs ca_3: C-Cluster ca10 > ca3",
     experiments['sens_move_ca_10']['meta']['results_summary']['avg_cluster_C_mean'] >
     experiments['sens_move_ca_3']['meta']['results_summary']['avg_cluster_C_mean']),

    ("ablation_movement_trust vs static_trust: C-Surv move >> static",
     experiments['ablation_movement_trust']['meta']['results_summary']['survival_C_mean'] >
     experiments['ablation_static_trust']['meta']['results_summary']['survival_C_mean'] + 0.10),

    ("ablation_movement_only vs movement_trust: Trust hilft bei Movement",
     experiments['ablation_movement_trust']['meta']['results_summary']['survival_C_mean'] >=
     experiments['ablation_movement_only']['meta']['results_summary']['survival_C_mean']),

    ("exp10d kein Kipp-Punkt",
     not experiments['exp10d_static_control']['meta']['results_summary']['kipp_punkt']),

    ("ablation_static_trust kein Kipp-Punkt",
     not experiments['ablation_static_trust']['meta']['results_summary']['kipp_punkt']),
]

all_pass = True
for desc, result in checks:
    status = "✓ PASS" if result else "✗ FAIL"
    if not result:
        all_pass = False
    print(f"  [{status}] {desc}")

print(f"\n  Gesamtergebnis: {'ALLE CHECKS BESTANDEN' if all_pass else 'EINIGE CHECKS FEHLGESCHLAGEN'}")


# ═══════════════════════════════════════════════════════════════════════
# GRAND SUMMARY TABLE
# ═══════════════════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("  GESAMTÜBERSICHT: 17 Experimente, 340 Runs")
print("=" * 70)
print(f"  {'Experiment':<30} {'Grp':>3} {'Runs':>4} {'Surv A':>8} {'Surv B':>8} {'Surv C':>8} "
      f"{'Cl_C':>5} {'C>B':>5} {'Kipp':>5}")
print("  " + "─" * 90)

total_runs = 0
for name in sorted(experiments.keys()):
    m = experiments[name]['meta']['results_summary']
    g = experiments[name]['meta']['group']
    n = m['num_runs']
    total_runs += n
    kipp = "JA" if m['kipp_punkt'] else "—"
    print(f"  {name:<30} {g:>3} {n:>4} "
          f"{m['survival_A_mean']:>7.1%} {m['survival_B_mean']:>7.1%} {m['survival_C_mean']:>7.1%} "
          f"{m.get('avg_cluster_C_mean', 0):>5.1f} "
          f"{m['c_beats_b']}/{n:>2} {kipp:>5}")

print("  " + "─" * 90)
print(f"  Total: {len(experiments)} Experimente, {total_runs} Runs")
print(f"  Kipp-Punkt bestätigt: {sum(1 for e in experiments.values() if e['meta']['results_summary']['kipp_punkt'])}/17 Experimente")
print(f"  Kein Kipp-Punkt:      {sum(1 for e in experiments.values() if not e['meta']['results_summary']['kipp_punkt'])}/17 Experimente")

print("\nAnalyse abgeschlossen. Charts gespeichert in:", OUT_DIR.resolve())
