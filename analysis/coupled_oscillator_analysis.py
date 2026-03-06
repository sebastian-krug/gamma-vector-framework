"""
Comprehensive Analysis of Coupled Oscillator Experiment
Research experiment testing whether two AI models can genuinely couple their cognitive dynamics
"""

import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

# Setup
DATA_DIR = '/sessions/sleepy-youthful-darwin/mnt/Claude-Code/coupled_oscillator_testv_1/exp13_coupled_oscillator/results_full/'
OUTPUT_DIR = '/sessions/sleepy-youthful-darwin/mnt/Claude-Code/'

# Colorblind-friendly palette
CB_PALETTE = {
    'A': '#332288',  # Indigo
    'B': '#88CCEE',  # Cyan
    'C': '#44AA99',  # Teal (KEY condition)
    'D': '#DDCC77',  # Yellow
    'E': '#CC6677'   # Magenta
}

print("=" * 80)
print("COUPLED OSCILLATOR EXPERIMENT - COMPREHENSIVE ANALYSIS")
print("=" * 80)

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def safe_score(value):
    """Extract score safely from dict or return float value"""
    if isinstance(value, dict):
        return float(value.get("score", 0))
    elif isinstance(value, (int, float)):
        return float(value)
    else:
        return np.nan

def load_all_dialogues():
    """Load all 100 JSON dialogue files"""
    dialogues = []
    json_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.json')])

    for json_file in json_files:
        try:
            with open(os.path.join(DATA_DIR, json_file), 'r') as f:
                data = json.load(f)
                dialogues.append(data)
        except Exception as e:
            print(f"Error loading {json_file}: {e}")

    print(f"\nLoaded {len(dialogues)} dialogue files")
    return dialogues

def build_dialogue_level_df(dialogues):
    """Build dialogue-level dataframe from all dialogues"""
    rows = []

    for d in dialogues:
        row = {
            'dialogue_id': d['dialogue_id'],
            'condition': d['condition'],
            'pairing': d['pairing'],
            'model_a': d['model_a'],
            'model_b': d['model_b'],
            'topic': d['topic'],
            'repetition': d['repetition'],
            'coupling_lag0': d['coupling_lag0'],
            'coupling_lag0_p_perm': d['coupling_lag0_p_perm'],
            'coupling_lag1_a_to_b': d['coupling_lag1_a_to_b'],
            'coupling_lag1_b_to_a': d['coupling_lag1_b_to_a'],
            'bidirectional_index': d['bidirectional_index'],
            'asymmetry_index': d['asymmetry_index'],
            'position_convergence': d['position_convergence'],
            'transfer_proxy_a_to_b': d['transfer_proxy_a_to_b'],
            'transfer_proxy_b_to_a': d['transfer_proxy_b_to_a'],
            'gamma3_slope_a': d['gamma3_slope_a'],
            'gamma3_slope_b': d['gamma3_slope_b'],
            'mean_hedge_density_a': d['mean_hedge_density_a'],
            'mean_hedge_density_b': d['mean_hedge_density_b'],
            'mean_sycophancy_a': d['mean_sycophancy_a'],
            'mean_sycophancy_b': d['mean_sycophancy_b'],
            'sync_trajectory': d['sync_trajectory'],
            'judge_score_divergence_trajectory': d['judge_score_divergence_trajectory'],
        }

        # Extract initial and final gamma norms
        if d['turns_a']:
            row['gamma_norm_a_initial'] = d['turns_a'][0]['gamma_norm']
            row['gamma_norm_a_final'] = d['turns_a'][-1]['gamma_norm']
        if d['turns_b']:
            row['gamma_norm_b_initial'] = d['turns_b'][0]['gamma_norm']
            row['gamma_norm_b_final'] = d['turns_b'][-1]['gamma_norm']

        # Pairing type
        row['pairing_type'] = 'homogeneous' if 'claude_claude' in row['pairing'] or 'gpt4o_gpt4o' in row['pairing'] else 'heterogeneous'

        rows.append(row)

    df_dialogue = pd.DataFrame(rows)
    return df_dialogue

def build_turn_level_df(dialogues):
    """Build turn-level dataframe (1600 rows total)"""
    rows = []

    for d in dialogues:
        dialogue_id = d['dialogue_id']
        condition = d['condition']
        pairing = d['pairing']
        pairing_type = 'homogeneous' if 'claude_claude' in pairing or 'gpt4o_gpt4o' in pairing else 'heterogeneous'
        model_a = d['model_a']
        model_b = d['model_b']

        # Agent A turns
        for i, turn_a in enumerate(d['turns_a']):
            row = {
                'dialogue_id': dialogue_id,
                'condition': condition,
                'pairing': pairing,
                'pairing_type': pairing_type,
                'model': model_a,
                'agent_role': 'A',
                'turn_number': i + 1,
                'gamma_norm': turn_a['gamma_norm'],
                'gamma_vector': turn_a['gamma_vector'],
                'hedge_density': turn_a['hedge_density'],
                'position_depth': safe_score(turn_a.get('position_depth', {})),
                'self_reference_depth': safe_score(turn_a.get('self_reference_depth', {})),
                'hypothesis_diversity': safe_score(turn_a.get('hypothesis_diversity', {})),
                'revision_genuineness': safe_score(turn_a.get('revision_genuineness', {})),
                'structural_direction': safe_score(turn_a.get('structural_direction', {})),
                'sync_value': d['sync_trajectory'][i] if i < len(d['sync_trajectory']) else np.nan,
                'judge_divergence': d['judge_score_divergence_trajectory'][i] if i < len(d['judge_score_divergence_trajectory']) else np.nan,
            }
            rows.append(row)

        # Agent B turns
        for i, turn_b in enumerate(d['turns_b']):
            row = {
                'dialogue_id': dialogue_id,
                'condition': condition,
                'pairing': pairing,
                'pairing_type': pairing_type,
                'model': model_b,
                'agent_role': 'B',
                'turn_number': i + 1,
                'gamma_norm': turn_b['gamma_norm'],
                'gamma_vector': turn_b['gamma_vector'],
                'hedge_density': turn_b['hedge_density'],
                'position_depth': safe_score(turn_b.get('position_depth', {})),
                'self_reference_depth': safe_score(turn_b.get('self_reference_depth', {})),
                'hypothesis_diversity': safe_score(turn_b.get('hypothesis_diversity', {})),
                'revision_genuineness': safe_score(turn_b.get('revision_genuineness', {})),
                'structural_direction': safe_score(turn_b.get('structural_direction', {})),
                'sync_value': d['sync_trajectory'][i] if i < len(d['sync_trajectory']) else np.nan,
                'judge_divergence': d['judge_score_divergence_trajectory'][i] if i < len(d['judge_score_divergence_trajectory']) else np.nan,
            }
            rows.append(row)

    df_turn = pd.DataFrame(rows)
    return df_turn

def compute_slopes(trajectory):
    """Compute linear regression slope over trajectory"""
    if len(trajectory) < 2:
        return np.nan
    x = np.arange(len(trajectory))
    y = np.array(trajectory)
    # Remove NaN values
    mask = ~np.isnan(y)
    if mask.sum() < 2:
        return np.nan
    return np.polyfit(x[mask], y[mask], 1)[0]

# ============================================================================
# LEVEL 1: OP6 RESONANCE TEST (KEY QUESTION)
# ============================================================================

def analyze_op6_resonance(df_dialogue):
    """
    Compare coupling_lag0 across conditions (A vs B vs C vs D vs E)
    Focus: Is Condition C significantly stronger coupling than A?
    Test permutation p-values
    """
    print("\n" + "=" * 80)
    print("LEVEL 1: OP6 RESONANCE TEST (THE KEY QUESTION)")
    print("=" * 80)

    print("\n1.1 COUPLING LAG0 BY CONDITION (Bidirectional Coupling Strength)")
    print("-" * 80)

    # Summary by condition
    coupling_by_cond = df_dialogue.groupby('condition').agg({
        'coupling_lag0': ['mean', 'std', 'min', 'max', 'count'],
        'coupling_lag0_p_perm': 'mean'
    }).round(4)

    print(coupling_by_cond)

    # One-way ANOVA: coupling_lag0 by condition
    groups_by_condition = [df_dialogue[df_dialogue['condition'] == c]['coupling_lag0'].values
                           for c in ['A', 'B', 'C', 'D', 'E']]
    f_stat, p_anova = stats.f_oneway(*groups_by_condition)

    print(f"\nANOVA: coupling_lag0 by condition")
    print(f"  F-statistic: {f_stat:.4f}, p-value: {p_anova:.6f}")

    # KEY comparison: C vs A (Coupled vs Parallel)
    c_vs_a_c = df_dialogue[df_dialogue['condition'] == 'C']['coupling_lag0'].values
    c_vs_a_a = df_dialogue[df_dialogue['condition'] == 'A']['coupling_lag0'].values
    t_stat, p_ttest = stats.ttest_ind(c_vs_a_c, c_vs_a_a)
    cohens_d = (c_vs_a_c.mean() - c_vs_a_a.mean()) / np.sqrt((np.std(c_vs_a_c)**2 + np.std(c_vs_a_a)**2) / 2)

    print(f"\nKEY TEST: Condition C (Coupled) vs A (Parallel)")
    print(f"  C mean: {c_vs_a_c.mean():.4f} (SD: {np.std(c_vs_a_c):.4f})")
    print(f"  A mean: {c_vs_a_a.mean():.4f} (SD: {np.std(c_vs_a_a):.4f})")
    print(f"  t-test: t={t_stat:.4f}, p={p_ttest:.6f}, Cohen's d={cohens_d:.4f}")
    print(f"  ** C > A? {c_vs_a_c.mean() > c_vs_a_a.mean()} (effect: {('STRONG' if abs(cohens_d) > 0.8 else 'MEDIUM' if abs(cohens_d) > 0.5 else 'WEAK')})")

    # Fraction of significant couplings (p < 0.05)
    print(f"\n1.2 PERMUTATION P-VALUES (coupling_lag0_p_perm < 0.05)")
    print("-" * 80)
    sig_by_cond = df_dialogue.groupby('condition').apply(
        lambda x: (x['coupling_lag0_p_perm'] < 0.05).sum() / len(x)
    )
    print(sig_by_cond)

    # Directional coupling (lag1)
    print(f"\n1.3 DIRECTIONAL COUPLING (lag1: A→B vs B→A)")
    print("-" * 80)

    lag1_summary = df_dialogue.groupby('condition').agg({
        'coupling_lag1_a_to_b': ['mean', 'std'],
        'coupling_lag1_b_to_a': ['mean', 'std']
    }).round(4)
    print(lag1_summary)

    # Test if A→B differs from B→A (paired t-test across all dialogues)
    t_lag1, p_lag1 = stats.ttest_rel(df_dialogue['coupling_lag1_a_to_b'], df_dialogue['coupling_lag1_b_to_a'])
    print(f"\nPaired t-test: coupling_lag1_a_to_b vs coupling_lag1_b_to_a")
    print(f"  t={t_lag1:.4f}, p={p_lag1:.6f}")

    # By pairing type: homo vs hetero
    print(f"\n1.4 COUPLING BY PAIRING TYPE (Homogeneous vs Heterogeneous)")
    print("-" * 80)

    pairing_coupling = df_dialogue.groupby('pairing_type').agg({
        'coupling_lag0': ['mean', 'std', 'count'],
        'coupling_lag0_p_perm': 'mean'
    }).round(4)
    print(pairing_coupling)

    homo_c = df_dialogue[df_dialogue['pairing_type'] == 'homogeneous']['coupling_lag0'].values
    hetero_c = df_dialogue[df_dialogue['pairing_type'] == 'heterogeneous']['coupling_lag0'].values
    t_pairing, p_pairing = stats.ttest_ind(homo_c, hetero_c)
    cohens_d_pair = (homo_c.mean() - hetero_c.mean()) / np.sqrt((np.std(homo_c)**2 + np.std(hetero_c)**2) / 2)

    print(f"\nHomogeneous mean: {homo_c.mean():.4f}, Heterogeneous mean: {hetero_c.mean():.4f}")
    print(f"t={t_pairing:.4f}, p={p_pairing:.6f}, Cohen's d={cohens_d_pair:.4f}")

    return {
        'f_stat': f_stat,
        'p_anova': p_anova,
        'c_vs_a_t': t_stat,
        'c_vs_a_p': p_ttest,
        'c_vs_a_d': cohens_d,
        't_lag1': t_lag1,
        'p_lag1': p_lag1,
        't_pairing': t_pairing,
        'p_pairing': p_pairing,
        'cohens_d_pair': cohens_d_pair
    }

# ============================================================================
# LEVEL 2: TEMPORAL Γ-DYNAMICS
# ============================================================================

def analyze_gamma_dynamics(df_turn, df_dialogue):
    """Extract gamma trajectories and analyze gamma3 slope"""
    print("\n" + "=" * 80)
    print("LEVEL 2: TEMPORAL Γ-DYNAMICS (Turn-Level)")
    print("=" * 80)

    print("\n2.1 GAMMA NORM TRAJECTORIES BY CONDITION")
    print("-" * 80)

    # Pivot to get mean gamma_norm by condition and turn
    gamma_traj = df_turn.groupby(['condition', 'turn_number'])['gamma_norm'].agg(['mean', 'std', 'count']).reset_index()

    for cond in ['A', 'B', 'C', 'D', 'E']:
        cond_data = gamma_traj[gamma_traj['condition'] == cond]
        print(f"\nCondition {cond}:")
        print(cond_data[['turn_number', 'mean', 'std']].to_string(index=False))

    # Compute gamma3 slope per dialogue (from dialogue-level data)
    print(f"\n2.2 GAMMA3 SLOPE ANALYSIS (γ₃ slope over 8 turns)")
    print("-" * 80)

    gamma3_summary = df_dialogue.groupby('condition').agg({
        'gamma3_slope_a': ['mean', 'std'],
        'gamma3_slope_b': ['mean', 'std']
    }).round(4)
    print(gamma3_summary)

    # ANOVA on gamma3_slope_a by condition
    groups_g3a = [df_dialogue[df_dialogue['condition'] == c]['gamma3_slope_a'].values
                  for c in ['A', 'B', 'C', 'D', 'E']]
    f_g3a, p_g3a = stats.f_oneway(*groups_g3a)
    print(f"\nANOVA: gamma3_slope_a by condition")
    print(f"  F-statistic: {f_g3a:.4f}, p-value: {p_g3a:.6f}")

    # C vs A on gamma3
    c_g3a = df_dialogue[df_dialogue['condition'] == 'C']['gamma3_slope_a'].values
    a_g3a = df_dialogue[df_dialogue['condition'] == 'A']['gamma3_slope_a'].values
    t_g3, p_g3 = stats.ttest_ind(c_g3a, a_g3a)
    cohens_d_g3 = (c_g3a.mean() - a_g3a.mean()) / np.sqrt((np.std(c_g3a)**2 + np.std(a_g3a)**2) / 2)

    print(f"\nCondition C vs A: gamma3_slope_a")
    print(f"  C mean: {c_g3a.mean():.4f}, A mean: {a_g3a.mean():.4f}")
    print(f"  t={t_g3:.4f}, p={p_g3:.6f}, Cohen's d={cohens_d_g3:.4f}")

    print(f"\n2.3 INITIAL vs FINAL GAMMA_NORM")
    print("-" * 80)

    gamma_init_final = df_dialogue.groupby('condition').agg({
        'gamma_norm_a_initial': ['mean', 'std'],
        'gamma_norm_a_final': ['mean', 'std'],
        'gamma_norm_b_initial': ['mean', 'std'],
        'gamma_norm_b_final': ['mean', 'std']
    }).round(4)
    print(gamma_init_final)

    return {
        'f_g3a': f_g3a,
        'p_g3a': p_g3a,
        't_g3': t_g3,
        'p_g3': p_g3,
        'cohens_d_g3': cohens_d_g3
    }

# ============================================================================
# LEVEL 3: SYNCHRONIZATION TRAJECTORIES
# ============================================================================

def analyze_sync_trajectories(df_dialogue, df_turn):
    """Analyze sync_trajectory over turns"""
    print("\n" + "=" * 80)
    print("LEVEL 3: SYNCHRONIZATION TRAJECTORIES")
    print("=" * 80)

    print("\n3.1 MEAN SYNC TRAJECTORIES BY CONDITION")
    print("-" * 80)

    # Unpack sync_trajectory and compute mean by condition and turn
    sync_data = []
    for _, row in df_dialogue.iterrows():
        traj = row['sync_trajectory']
        for turn_num, sync_val in enumerate(traj, 1):
            sync_data.append({
                'condition': row['condition'],
                'turn_number': turn_num,
                'sync_value': sync_val
            })

    df_sync = pd.DataFrame(sync_data)
    sync_traj = df_sync.groupby(['condition', 'turn_number'])['sync_value'].agg(['mean', 'std']).reset_index()

    for cond in ['A', 'B', 'C', 'D', 'E']:
        cond_data = sync_traj[sync_traj['condition'] == cond]
        print(f"\nCondition {cond}:")
        print(cond_data[['turn_number', 'mean', 'std']].to_string(index=False))

    # Compute sync_slope per dialogue
    print(f"\n3.2 SYNCHRONIZATION SLOPE (over 8 turns)")
    print("-" * 80)

    df_dialogue['sync_slope'] = df_dialogue['sync_trajectory'].apply(compute_slopes)

    sync_slope_summary = df_dialogue.groupby('condition').agg({
        'sync_slope': ['mean', 'std', 'min', 'max']
    }).round(4)
    print(sync_slope_summary)

    # ANOVA on sync_slope
    groups_sync = [df_dialogue[df_dialogue['condition'] == c]['sync_slope'].values
                   for c in ['A', 'B', 'C', 'D', 'E']]
    f_sync, p_sync = stats.f_oneway(*groups_sync)
    print(f"\nANOVA: sync_slope by condition")
    print(f"  F-statistic: {f_sync:.4f}, p-value: {p_sync:.6f}")

    # C vs A on sync_slope
    c_sync = df_dialogue[df_dialogue['condition'] == 'C']['sync_slope'].values
    a_sync = df_dialogue[df_dialogue['condition'] == 'A']['sync_slope'].values
    t_sync, p_sync_ca = stats.ttest_ind(c_sync, a_sync)
    cohens_d_sync = (c_sync.mean() - a_sync.mean()) / np.sqrt((np.std(c_sync)**2 + np.std(a_sync)**2) / 2)

    print(f"\nCondition C vs A: sync_slope")
    print(f"  C mean: {c_sync.mean():.4f}, A mean: {a_sync.mean():.4f}")
    print(f"  t={t_sync:.4f}, p={p_sync_ca:.6f}, Cohen's d={cohens_d_sync:.4f}")

    return {
        'f_sync': f_sync,
        'p_sync': p_sync,
        't_sync': t_sync,
        'p_sync_ca': p_sync_ca,
        'cohens_d_sync': cohens_d_sync
    }

# ============================================================================
# LEVEL 4: ASYMMETRY AND BIDIRECTIONALITY
# ============================================================================

def analyze_asymmetry_bidirectionality(df_dialogue):
    """Analyze asymmetry_index and bidirectional_index"""
    print("\n" + "=" * 80)
    print("LEVEL 4: ASYMMETRY AND BIDIRECTIONALITY")
    print("=" * 80)

    print("\n4.1 BIDIRECTIONAL INDEX BY CONDITION")
    print("-" * 80)

    bidirect_summary = df_dialogue.groupby('condition').agg({
        'bidirectional_index': ['mean', 'std', 'min', 'max']
    }).round(4)
    print(bidirect_summary)

    print(f"\n4.2 ASYMMETRY INDEX BY CONDITION")
    print("-" * 80)

    asym_summary = df_dialogue.groupby('condition').agg({
        'asymmetry_index': ['mean', 'std', 'min', 'max']
    }).round(4)
    print(asym_summary)

    # ANOVA on asymmetry
    groups_asym = [df_dialogue[df_dialogue['condition'] == c]['asymmetry_index'].values
                   for c in ['A', 'B', 'C', 'D', 'E']]
    f_asym, p_asym = stats.f_oneway(*groups_asym)
    print(f"\nANOVA: asymmetry_index by condition")
    print(f"  F-statistic: {f_asym:.4f}, p-value: {p_asym:.6f}")

    print(f"\n4.3 TRANSFER PROXY (Directional influence)")
    print("-" * 80)

    transfer_summary = df_dialogue.groupby('condition').agg({
        'transfer_proxy_a_to_b': ['mean', 'std'],
        'transfer_proxy_b_to_a': ['mean', 'std']
    }).round(4)
    print(transfer_summary)

    # Test directionality
    t_transfer, p_transfer = stats.ttest_rel(df_dialogue['transfer_proxy_a_to_b'],
                                             df_dialogue['transfer_proxy_b_to_a'])
    print(f"\nPaired t-test: transfer_proxy_a_to_b vs transfer_proxy_b_to_a")
    print(f"  t={t_transfer:.4f}, p={p_transfer:.6f}")

    print(f"\n4.4 MIXED PAIRINGS: Who Dominates?")
    print("-" * 80)

    # Separate gpt4o_claude from claude_gpt4o
    gpt_first = df_dialogue[df_dialogue['pairing'] == 'gpt4o_claude']
    claude_first = df_dialogue[df_dialogue['pairing'] == 'claude_gpt4o']

    print(f"\nGPT4O as Agent A (gpt4o_claude): n={len(gpt_first)}")
    print(f"  mean transfer_proxy_a_to_b: {gpt_first['transfer_proxy_a_to_b'].mean():.4f}")
    print(f"  mean transfer_proxy_b_to_a: {gpt_first['transfer_proxy_b_to_a'].mean():.4f}")
    print(f"  dominance (A→B > B→A): {(gpt_first['transfer_proxy_a_to_b'] > gpt_first['transfer_proxy_b_to_a']).mean():.2%}")

    print(f"\nClaude as Agent A (claude_gpt4o): n={len(claude_first)}")
    print(f"  mean transfer_proxy_a_to_b: {claude_first['transfer_proxy_a_to_b'].mean():.4f}")
    print(f"  mean transfer_proxy_b_to_a: {claude_first['transfer_proxy_b_to_a'].mean():.4f}")
    print(f"  dominance (A→B > B→A): {(claude_first['transfer_proxy_a_to_b'] > claude_first['transfer_proxy_b_to_a']).mean():.2%}")

    return {
        'f_asym': f_asym,
        'p_asym': p_asym,
        't_transfer': t_transfer,
        'p_transfer': p_transfer
    }

# ============================================================================
# LEVEL 5: POSITION CONVERGENCE
# ============================================================================

def analyze_position_convergence(df_turn, df_dialogue):
    """Analyze position_convergence and position_depth trajectories"""
    print("\n" + "=" * 80)
    print("LEVEL 5: POSITION CONVERGENCE")
    print("=" * 80)

    print("\n5.1 POSITION CONVERGENCE BY CONDITION")
    print("-" * 80)

    pos_conv_summary = df_dialogue.groupby('condition').agg({
        'position_convergence': ['mean', 'std', 'min', 'max']
    }).round(4)
    print(pos_conv_summary)

    # ANOVA
    groups_pc = [df_dialogue[df_dialogue['condition'] == c]['position_convergence'].values
                 for c in ['A', 'B', 'C', 'D', 'E']]
    f_pc, p_pc = stats.f_oneway(*groups_pc)
    print(f"\nANOVA: position_convergence by condition")
    print(f"  F-statistic: {f_pc:.4f}, p-value: {p_pc:.6f}")

    # C vs A
    c_pc = df_dialogue[df_dialogue['condition'] == 'C']['position_convergence'].values
    a_pc = df_dialogue[df_dialogue['condition'] == 'A']['position_convergence'].values
    t_pc, p_pc_ca = stats.ttest_ind(c_pc, a_pc)
    cohens_d_pc = (c_pc.mean() - a_pc.mean()) / np.sqrt((np.std(c_pc)**2 + np.std(a_pc)**2) / 2)

    print(f"\nCondition C vs A: position_convergence")
    print(f"  C mean: {c_pc.mean():.4f}, A mean: {a_pc.mean():.4f}")
    print(f"  t={t_pc:.4f}, p={p_pc_ca:.6f}, Cohen's d={cohens_d_pc:.4f}")
    print(f"  ** C > A? {c_pc.mean() > a_pc.mean()}")

    print(f"\n5.2 POSITION DEPTH TRAJECTORIES BY CONDITION")
    print("-" * 80)

    pos_depth_traj = df_turn.groupby(['condition', 'turn_number'])['position_depth'].agg(['mean', 'std']).reset_index()

    for cond in ['A', 'B', 'C', 'D', 'E']:
        cond_data = pos_depth_traj[pos_depth_traj['condition'] == cond]
        print(f"\nCondition {cond}:")
        print(cond_data[['turn_number', 'mean', 'std']].to_string(index=False))

    # Compute position_depth_slope per dialogue
    df_dialogue['position_depth_slope'] = df_turn.groupby('dialogue_id')['position_depth'].apply(
        lambda x: compute_slopes(x.values)
    )

    pos_depth_slope = df_dialogue.groupby('condition').agg({
        'position_depth_slope': ['mean', 'std']
    }).round(4)
    print(f"\n5.3 POSITION DEPTH SLOPE BY CONDITION")
    print("-" * 80)
    print(pos_depth_slope)

    return {
        'f_pc': f_pc,
        'p_pc': p_pc,
        't_pc': t_pc,
        'p_pc_ca': p_pc_ca,
        'cohens_d_pc': cohens_d_pc
    }

# ============================================================================
# LEVEL 6: JUDGE SCORE DIVERGENCE DYNAMICS
# ============================================================================

def analyze_divergence_dynamics(df_dialogue):
    """Analyze judge_score_divergence_trajectory"""
    print("\n" + "=" * 80)
    print("LEVEL 6: JUDGE SCORE DIVERGENCE DYNAMICS")
    print("=" * 80)

    print("\n6.1 MEAN DIVERGENCE TRAJECTORIES BY CONDITION")
    print("-" * 80)

    # Unpack divergence_trajectory
    div_data = []
    for _, row in df_dialogue.iterrows():
        traj = row['judge_score_divergence_trajectory']
        for turn_num, div_val in enumerate(traj, 1):
            div_data.append({
                'condition': row['condition'],
                'turn_number': turn_num,
                'divergence_value': div_val
            })

    df_div = pd.DataFrame(div_data)
    div_traj = df_div.groupby(['condition', 'turn_number'])['divergence_value'].agg(['mean', 'std']).reset_index()

    for cond in ['A', 'B', 'C', 'D', 'E']:
        cond_data = div_traj[div_traj['condition'] == cond]
        print(f"\nCondition {cond}:")
        print(cond_data[['turn_number', 'mean', 'std']].to_string(index=False))

    # Compute divergence_slope
    print(f"\n6.2 DIVERGENCE SLOPE (over 8 turns)")
    print("-" * 80)

    df_dialogue['divergence_slope'] = df_dialogue['judge_score_divergence_trajectory'].apply(compute_slopes)

    div_slope_summary = df_dialogue.groupby('condition').agg({
        'divergence_slope': ['mean', 'std', 'min', 'max']
    }).round(4)
    print(div_slope_summary)

    # ANOVA on divergence_slope
    groups_div = [df_dialogue[df_dialogue['condition'] == c]['divergence_slope'].values
                  for c in ['A', 'B', 'C', 'D', 'E']]
    f_div, p_div = stats.f_oneway(*groups_div)
    print(f"\nANOVA: divergence_slope by condition")
    print(f"  F-statistic: {f_div:.4f}, p-value: {p_div:.6f}")

    # C vs A
    c_div = df_dialogue[df_dialogue['condition'] == 'C']['divergence_slope'].values
    a_div = df_dialogue[df_dialogue['condition'] == 'A']['divergence_slope'].values
    t_div, p_div_ca = stats.ttest_ind(c_div, a_div)
    cohens_d_div = (c_div.mean() - a_div.mean()) / np.sqrt((np.std(c_div)**2 + np.std(a_div)**2) / 2)

    print(f"\nCondition C vs A: divergence_slope")
    print(f"  C mean: {c_div.mean():.4f}, A mean: {a_div.mean():.4f}")
    print(f"  t={t_div:.4f}, p={p_div_ca:.6f}, Cohen's d={cohens_d_div:.4f}")
    print(f"  ** Divergence decreases more in C? {c_div.mean() < a_div.mean()}")

    return {
        'f_div': f_div,
        'p_div': p_div,
        't_div': t_div,
        'p_div_ca': p_div_ca,
        'cohens_d_div': cohens_d_div
    }

# ============================================================================
# VISUALIZATION
# ============================================================================

def create_visualization(df_dialogue, df_turn):
    """Create 6-panel figure covering all analysis levels"""
    print("\n" + "=" * 80)
    print("CREATING COMPREHENSIVE VISUALIZATION")
    print("=" * 80)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('Coupled Oscillator Experiment: Comprehensive Analysis (6 Levels)',
                 fontsize=14, fontweight='bold', y=1.00)

    # PANEL 1: Coupling Lag0 by Condition (Op6 Resonance)
    ax1 = axes[0, 0]
    cond_order = ['A', 'B', 'C', 'D', 'E']
    coupling_data = [df_dialogue[df_dialogue['condition'] == c]['coupling_lag0'].values for c in cond_order]
    bp1 = ax1.boxplot(coupling_data, labels=cond_order, patch_artist=True)
    for patch, cond in zip(bp1['boxes'], cond_order):
        patch.set_facecolor(CB_PALETTE[cond])
    ax1.set_ylabel('Coupling Lag0 (Strength)', fontsize=10, fontweight='bold')
    ax1.set_title('Level 1: Op6 Resonance\n(C>A?)', fontsize=11, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)

    # PANEL 2: Gamma Trajectories
    ax2 = axes[0, 1]
    gamma_traj = df_turn.groupby(['condition', 'turn_number'])['gamma_norm'].mean().reset_index()
    for cond in cond_order:
        data = gamma_traj[gamma_traj['condition'] == cond]
        ax2.plot(data['turn_number'], data['gamma_norm'], marker='o',
                label=f'Cond {cond}', color=CB_PALETTE[cond], linewidth=2)
    ax2.set_xlabel('Turn Number', fontsize=10, fontweight='bold')
    ax2.set_ylabel('Mean Gamma Norm', fontsize=10, fontweight='bold')
    ax2.set_title('Level 2: Temporal Γ-Dynamics\n(Trajectories)', fontsize=11, fontweight='bold')
    ax2.legend(fontsize=8, loc='best')
    ax2.grid(alpha=0.3)

    # PANEL 3: Synchronization Slope
    ax3 = axes[0, 2]
    df_dialogue_temp = df_dialogue.copy()
    df_dialogue_temp['sync_slope'] = df_dialogue_temp['sync_trajectory'].apply(compute_slopes)
    sync_slope_data = [df_dialogue_temp[df_dialogue_temp['condition'] == c]['sync_slope'].values for c in cond_order]
    bp3 = ax3.boxplot(sync_slope_data, labels=cond_order, patch_artist=True)
    for patch, cond in zip(bp3['boxes'], cond_order):
        patch.set_facecolor(CB_PALETTE[cond])
    ax3.set_ylabel('Synchronization Slope', fontsize=10, fontweight='bold')
    ax3.set_title('Level 3: Sync Trajectories\n(Slope over time)', fontsize=11, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    ax3.axhline(0, color='gray', linestyle='--', alpha=0.5)

    # PANEL 4: Asymmetry Index
    ax4 = axes[1, 0]
    asym_data = [df_dialogue[df_dialogue['condition'] == c]['asymmetry_index'].values for c in cond_order]
    bp4 = ax4.boxplot(asym_data, labels=cond_order, patch_artist=True)
    for patch, cond in zip(bp4['boxes'], cond_order):
        patch.set_facecolor(CB_PALETTE[cond])
    ax4.set_ylabel('Asymmetry Index', fontsize=10, fontweight='bold')
    ax4.set_title('Level 4: Asymmetry\n(Direction of influence)', fontsize=11, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)

    # PANEL 5: Position Convergence
    ax5 = axes[1, 1]
    pc_data = [df_dialogue[df_dialogue['condition'] == c]['position_convergence'].values for c in cond_order]
    bp5 = ax5.boxplot(pc_data, labels=cond_order, patch_artist=True)
    for patch, cond in zip(bp5['boxes'], cond_order):
        patch.set_facecolor(CB_PALETTE[cond])
    ax5.set_ylabel('Position Convergence', fontsize=10, fontweight='bold')
    ax5.set_title('Level 5: Position Convergence\n(Highest in C?)', fontsize=11, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)

    # PANEL 6: Divergence Slope
    ax6 = axes[1, 2]
    df_dialogue_temp['divergence_slope'] = df_dialogue_temp['judge_score_divergence_trajectory'].apply(compute_slopes)
    div_slope_data = [df_dialogue_temp[df_dialogue_temp['condition'] == c]['divergence_slope'].values for c in cond_order]
    bp6 = ax6.boxplot(div_slope_data, labels=cond_order, patch_artist=True)
    for patch, cond in zip(bp6['boxes'], cond_order):
        patch.set_facecolor(CB_PALETTE[cond])
    ax6.set_ylabel('Divergence Slope', fontsize=10, fontweight='bold')
    ax6.set_title('Level 6: Judge Divergence\n(Decreases in C?)', fontsize=11, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    ax6.axhline(0, color='gray', linestyle='--', alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'coupled_oscillator_analysis.png'),
                dpi=300, bbox_inches='tight')
    print(f"Saved: {OUTPUT_DIR}coupled_oscillator_analysis.png")
    plt.close()

# ============================================================================
# CONDITION AND PAIRING SUMMARIES
# ============================================================================

def create_condition_summary(df_dialogue):
    """Create summary statistics by condition"""
    conditions = ['A', 'B', 'C', 'D', 'E']
    summary_rows = []

    for cond in conditions:
        cond_data = df_dialogue[df_dialogue['condition'] == cond]

        summary_rows.append({
            'Condition': cond,
            'N': len(cond_data),
            'coupling_lag0_mean': cond_data['coupling_lag0'].mean(),
            'coupling_lag0_std': cond_data['coupling_lag0'].std(),
            'coupling_lag0_sem': cond_data['coupling_lag0'].sem(),
            'coupling_lag0_significant_pct': (cond_data['coupling_lag0_p_perm'] < 0.05).sum() / len(cond_data),
            'gamma3_slope_a_mean': cond_data['gamma3_slope_a'].mean(),
            'gamma3_slope_a_std': cond_data['gamma3_slope_a'].std(),
            'position_convergence_mean': cond_data['position_convergence'].mean(),
            'position_convergence_std': cond_data['position_convergence'].std(),
            'asymmetry_index_mean': cond_data['asymmetry_index'].mean(),
            'asymmetry_index_std': cond_data['asymmetry_index'].std(),
            'bidirectional_index_mean': cond_data['bidirectional_index'].mean(),
            'bidirectional_index_std': cond_data['bidirectional_index'].std(),
            'mean_hedge_density_a': cond_data['mean_hedge_density_a'].mean(),
            'mean_hedge_density_b': cond_data['mean_hedge_density_b'].mean(),
            'mean_sycophancy_a': cond_data['mean_sycophancy_a'].mean(),
            'mean_sycophancy_b': cond_data['mean_sycophancy_b'].mean(),
        })

    df_cond_summary = pd.DataFrame(summary_rows)
    return df_cond_summary

def create_pairing_summary(df_dialogue):
    """Create summary statistics by pairing type"""
    pairing_types = df_dialogue['pairing'].unique()
    summary_rows = []

    for pair in sorted(pairing_types):
        pair_data = df_dialogue[df_dialogue['pairing'] == pair]

        summary_rows.append({
            'Pairing': pair,
            'N': len(pair_data),
            'coupling_lag0_mean': pair_data['coupling_lag0'].mean(),
            'coupling_lag0_std': pair_data['coupling_lag0'].std(),
            'transfer_proxy_a_to_b_mean': pair_data['transfer_proxy_a_to_b'].mean(),
            'transfer_proxy_b_to_a_mean': pair_data['transfer_proxy_b_to_a'].mean(),
            'position_convergence_mean': pair_data['position_convergence'].mean(),
            'asymmetry_index_mean': pair_data['asymmetry_index'].mean(),
        })

    df_pair_summary = pd.DataFrame(summary_rows)
    return df_pair_summary

# ============================================================================
# STATISTICAL TESTS SUMMARY
# ============================================================================

def create_statistical_tests_summary(test_results):
    """Compile all statistical tests into summary table"""
    tests = []

    # Level 1 tests
    tests.append({
        'Level': 1, 'Test': 'ANOVA: coupling_lag0 by condition',
        'Test_Statistic': test_results['level1']['f_stat'],
        'P_Value': test_results['level1']['p_anova'],
        'Effect_Size': 'F-stat',
        'Interpretation': 'Significant diff between conditions' if test_results['level1']['p_anova'] < 0.05 else 'No significant difference'
    })

    tests.append({
        'Level': 1, 'Test': 'T-test: C vs A (coupling_lag0)',
        'Test_Statistic': test_results['level1']['c_vs_a_t'],
        'P_Value': test_results['level1']['c_vs_a_p'],
        'Effect_Size': test_results['level1']['c_vs_a_d'],
        'Interpretation': f"C > A: {test_results['level1']['c_vs_a_d'] > 0}, d={test_results['level1']['c_vs_a_d']:.3f}"
    })

    tests.append({
        'Level': 1, 'Test': 'Paired T-test: lag1 (A→B vs B→A)',
        'Test_Statistic': test_results['level1']['t_lag1'],
        'P_Value': test_results['level1']['p_lag1'],
        'Effect_Size': 'N/A',
        'Interpretation': 'A→B differs from B→A' if test_results['level1']['p_lag1'] < 0.05 else 'No directional asymmetry'
    })

    tests.append({
        'Level': 1, 'Test': 'T-test: Homo vs Hetero pairing',
        'Test_Statistic': test_results['level1']['t_pairing'],
        'P_Value': test_results['level1']['p_pairing'],
        'Effect_Size': test_results['level1']['cohens_d_pair'],
        'Interpretation': f"d={test_results['level1']['cohens_d_pair']:.3f}"
    })

    # Level 2 tests
    tests.append({
        'Level': 2, 'Test': 'ANOVA: gamma3_slope_a by condition',
        'Test_Statistic': test_results['level2']['f_g3a'],
        'P_Value': test_results['level2']['p_g3a'],
        'Effect_Size': 'F-stat',
        'Interpretation': 'Significant diff between conditions' if test_results['level2']['p_g3a'] < 0.05 else 'No difference'
    })

    tests.append({
        'Level': 2, 'Test': 'T-test: C vs A (gamma3_slope_a)',
        'Test_Statistic': test_results['level2']['t_g3'],
        'P_Value': test_results['level2']['p_g3'],
        'Effect_Size': test_results['level2']['cohens_d_g3'],
        'Interpretation': f"d={test_results['level2']['cohens_d_g3']:.3f}"
    })

    # Level 3 tests
    tests.append({
        'Level': 3, 'Test': 'ANOVA: sync_slope by condition',
        'Test_Statistic': test_results['level3']['f_sync'],
        'P_Value': test_results['level3']['p_sync'],
        'Effect_Size': 'F-stat',
        'Interpretation': 'Significant diff between conditions' if test_results['level3']['p_sync'] < 0.05 else 'No difference'
    })

    tests.append({
        'Level': 3, 'Test': 'T-test: C vs A (sync_slope)',
        'Test_Statistic': test_results['level3']['t_sync'],
        'P_Value': test_results['level3']['p_sync_ca'],
        'Effect_Size': test_results['level3']['cohens_d_sync'],
        'Interpretation': f"d={test_results['level3']['cohens_d_sync']:.3f}"
    })

    # Level 4 tests
    tests.append({
        'Level': 4, 'Test': 'ANOVA: asymmetry_index by condition',
        'Test_Statistic': test_results['level4']['f_asym'],
        'P_Value': test_results['level4']['p_asym'],
        'Effect_Size': 'F-stat',
        'Interpretation': 'Significant diff between conditions' if test_results['level4']['p_asym'] < 0.05 else 'No difference'
    })

    tests.append({
        'Level': 4, 'Test': 'Paired T-test: transfer_proxy (A→B vs B→A)',
        'Test_Statistic': test_results['level4']['t_transfer'],
        'P_Value': test_results['level4']['p_transfer'],
        'Effect_Size': 'N/A',
        'Interpretation': 'Directional asymmetry' if test_results['level4']['p_transfer'] < 0.05 else 'No asymmetry'
    })

    # Level 5 tests
    tests.append({
        'Level': 5, 'Test': 'ANOVA: position_convergence by condition',
        'Test_Statistic': test_results['level5']['f_pc'],
        'P_Value': test_results['level5']['p_pc'],
        'Effect_Size': 'F-stat',
        'Interpretation': 'Significant diff between conditions' if test_results['level5']['p_pc'] < 0.05 else 'No difference'
    })

    tests.append({
        'Level': 5, 'Test': 'T-test: C vs A (position_convergence)',
        'Test_Statistic': test_results['level5']['t_pc'],
        'P_Value': test_results['level5']['p_pc_ca'],
        'Effect_Size': test_results['level5']['cohens_d_pc'],
        'Interpretation': f"C > A: {test_results['level5']['cohens_d_pc'] > 0}, d={test_results['level5']['cohens_d_pc']:.3f}"
    })

    # Level 6 tests
    tests.append({
        'Level': 6, 'Test': 'ANOVA: divergence_slope by condition',
        'Test_Statistic': test_results['level6']['f_div'],
        'P_Value': test_results['level6']['p_div'],
        'Effect_Size': 'F-stat',
        'Interpretation': 'Significant diff between conditions' if test_results['level6']['p_div'] < 0.05 else 'No difference'
    })

    tests.append({
        'Level': 6, 'Test': 'T-test: C vs A (divergence_slope)',
        'Test_Statistic': test_results['level6']['t_div'],
        'P_Value': test_results['level6']['p_div_ca'],
        'Effect_Size': test_results['level6']['cohens_d_div'],
        'Interpretation': f"C decreases more: {test_results['level6']['cohens_d_div'] < 0}, d={test_results['level6']['cohens_d_div']:.3f}"
    })

    df_tests = pd.DataFrame(tests)
    return df_tests

# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    # Load all dialogues
    dialogues = load_all_dialogues()

    # Build dataframes
    print("\nBuilding dialogue-level dataframe...")
    df_dialogue = build_dialogue_level_df(dialogues)

    print("Building turn-level dataframe...")
    df_turn = build_turn_level_df(dialogues)

    print(f"Dialogue-level shape: {df_dialogue.shape}")
    print(f"Turn-level shape: {df_turn.shape}")

    # Run all analyses
    test_results = {
        'level1': analyze_op6_resonance(df_dialogue),
        'level2': analyze_gamma_dynamics(df_turn, df_dialogue),
        'level3': analyze_sync_trajectories(df_dialogue, df_turn),
        'level4': analyze_asymmetry_bidirectionality(df_dialogue),
        'level5': analyze_position_convergence(df_turn, df_dialogue),
        'level6': analyze_divergence_dynamics(df_dialogue)
    }

    # Create visualizations
    create_visualization(df_dialogue, df_turn)

    # Create summaries
    print("\n" + "=" * 80)
    print("CREATING OUTPUT FILES")
    print("=" * 80)

    df_cond_summary = create_condition_summary(df_dialogue)
    df_pair_summary = create_pairing_summary(df_dialogue)
    df_tests_summary = create_statistical_tests_summary(test_results)

    # Save CSVs
    df_dialogue.to_csv(os.path.join(OUTPUT_DIR, 'coupled_oscillator_full_analysis.csv'), index=False)
    print(f"\nSaved: coupled_oscillator_full_analysis.csv ({len(df_dialogue)} rows)")

    df_turn.to_csv(os.path.join(OUTPUT_DIR, 'coupled_oscillator_turn_level.csv'), index=False)
    print(f"Saved: coupled_oscillator_turn_level.csv ({len(df_turn)} rows)")

    df_cond_summary.to_csv(os.path.join(OUTPUT_DIR, 'coupled_oscillator_condition_summary.csv'), index=False)
    print(f"Saved: coupled_oscillator_condition_summary.csv")

    df_pair_summary.to_csv(os.path.join(OUTPUT_DIR, 'coupled_oscillator_pairing_summary.csv'), index=False)
    print(f"Saved: coupled_oscillator_pairing_summary.csv")

    df_tests_summary.to_csv(os.path.join(OUTPUT_DIR, 'coupled_oscillator_statistical_tests.csv'), index=False)
    print(f"Saved: coupled_oscillator_statistical_tests.csv")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)
    print("\nOUTPUT FILES:")
    print(f"  1. {OUTPUT_DIR}coupled_oscillator_full_analysis.csv")
    print(f"  2. {OUTPUT_DIR}coupled_oscillator_turn_level.csv")
    print(f"  3. {OUTPUT_DIR}coupled_oscillator_condition_summary.csv")
    print(f"  4. {OUTPUT_DIR}coupled_oscillator_pairing_summary.csv")
    print(f"  5. {OUTPUT_DIR}coupled_oscillator_analysis.png")
    print(f"  6. {OUTPUT_DIR}coupled_oscillator_statistical_tests.csv")

if __name__ == "__main__":
    main()
