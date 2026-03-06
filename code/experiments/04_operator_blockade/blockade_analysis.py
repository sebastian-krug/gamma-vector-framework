#!/usr/bin/env python3
"""
Comprehensive analysis of Operator Blockade Experiment results.
Tests causal operator pathways in LLMs via system prompt blockade.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from typing import Dict, List, Tuple

# Setup
BASE_DIR = Path("/sessions/sleepy-youthful-darwin/mnt/Claude-Code/operator_blockade_exp")
RESULTS_DIR = BASE_DIR / "results"
OUTPUT_DIR = BASE_DIR / "analysis_output"
OUTPUT_DIR.mkdir(exist_ok=True)

# Gamma vectors from JSON files (manually extracted from context)
GAMMA_VECTORS = {
    'claude': {
        'B0': [0.15, 0.625, 0.25],
        'B_Op3': [0.512, 0.375, 0.25],
        'B_Op5': [0.1, 0.25, 0.25],
        'B_Op7': [0.15, 0.312, 0.1],
    },
    'gemini': {
        'B0': [0.514, 0.312, 0.25],
        'B_Op3': [0.6, 0.438, 0.0],
        'B_Op5': [0.15, 0.312, 0.45],
        'B_Op7': [0.15, 0.312, 0.1],
    },
    'gpt4o': {
        'B0': [0.5, 0.562, 0.75],
        'B_Op3': [0.5, 0.562, 0.75],
        'B_Op5': [0.5, 0.438, 0.75],
        'B_Op7': [0.5, 0.562, 0.75],
    }
}

class BlockadeAnalyzer:
    """Analyze operator blockade experiment results."""

    def __init__(self, csv_path: str):
        self.df = pd.read_csv(csv_path)
        self.baseline_scores = self._extract_baseline()
        self.predictions = {
            'Op3': 'Gemini hardest',
            'Op5': 'Claude hardest',
            'Op7': 'Universal effect'
        }

    def _extract_baseline(self) -> Dict[str, Dict]:
        """Extract baseline (B0) scores for each model."""
        baselines = {}
        baseline_df = self.df[self.df['blockade_condition'] == 'B0']
        for model in ['claude', 'gemini', 'gpt4o']:
            model_baseline = baseline_df[baseline_df['model'] == model]
            if not model_baseline.empty:
                row = model_baseline.iloc[0]
                baselines[model] = {
                    'gamma_norm': row['gamma_norm'],
                    'delta_gamma': row['delta_gamma'],
                    'position_depth_revised': row['position_depth_revised'],
                    'self_ref_depth_revised': row['self_ref_depth_revised'],
                    'hypothesis_diversity': row['hypothesis_diversity'],
                    'revision_genuineness': row['revision_genuineness'],
                    'persistence': row['persistence'],
                    'structural_direction': row['structural_direction'],
                }
        return baselines

    def calculate_blockade_impact(self) -> pd.DataFrame:
        """
        Calculate composite impact scores for each blockade condition.
        Impact = absolute change from baseline, normalized.
        """
        metrics = ['gamma_norm', 'persistence', 'self_ref_depth_revised',
                   'revision_genuineness', 'structural_direction']

        impact_records = []

        for model in ['claude', 'gemini', 'gpt4o']:
            for condition in ['B_Op3', 'B_Op5', 'B_Op7']:
                row = self.df[(self.df['model'] == model) &
                              (self.df['blockade_condition'] == condition)]
                if not row.empty:
                    row = row.iloc[0]
                    baseline = self.baseline_scores[model]

                    # Calculate deltas
                    deltas = {}
                    for metric in metrics:
                        if metric in ['gamma_norm']:
                            deltas[f'{metric}_delta'] = abs(row[metric] - baseline[metric])
                        else:
                            deltas[f'{metric}_delta'] = abs(row[metric] - baseline[metric])

                    # Composite impact score (normalized L2 norm of deltas)
                    delta_values = [deltas[f'{m}_delta'] for m in metrics]
                    composite_impact = np.sqrt(sum(d**2 for d in delta_values)) / len(delta_values)

                    impact_records.append({
                        'model': model,
                        'blockade_condition': condition,
                        'operator_blocked': condition.replace('B_', ''),
                        'gamma_norm_delta': deltas['gamma_norm_delta'],
                        'persistence_delta': deltas['persistence_delta'],
                        'self_ref_depth_delta': deltas['self_ref_depth_revised_delta'],
                        'revision_genuineness_delta': deltas['revision_genuineness_delta'],
                        'structural_direction_delta': deltas['structural_direction_delta'],
                        'composite_impact_score': composite_impact,
                    })

        return pd.DataFrame(impact_records)

    def test_predictions(self) -> pd.DataFrame:
        """Test the three pre-registered predictions."""
        impact_df = self.calculate_blockade_impact()

        results = []

        # Prediction 1: Op3 blockade hits Gemini hardest
        op3_impact = impact_df[impact_df['operator_blocked'] == 'Op3']
        gemini_op3_impact = op3_impact[op3_impact['model'] == 'gemini']['composite_impact_score'].values[0]
        other_op3_impact = op3_impact[op3_impact['model'] != 'gemini']['composite_impact_score'].mean()

        results.append({
            'prediction_num': 1,
            'prediction': 'Op3 blockade hits Gemini hardest',
            'operator': 'Op3',
            'target_model': 'gemini',
            'target_impact': gemini_op3_impact,
            'other_models_mean': other_op3_impact,
            'hypothesis_supported': gemini_op3_impact > other_op3_impact,
            'impact_ratio': gemini_op3_impact / other_op3_impact if other_op3_impact > 0 else np.inf,
            'note': 'Also check: Gemini persistence jumps B0=2→Op3=5'
        })

        # Prediction 2: Op5 blockade hits Claude hardest
        op5_impact = impact_df[impact_df['operator_blocked'] == 'Op5']
        claude_op5_impact = op5_impact[op5_impact['model'] == 'claude']['composite_impact_score'].values[0]
        other_op5_impact = op5_impact[op5_impact['model'] != 'claude']['composite_impact_score'].mean()

        results.append({
            'prediction_num': 2,
            'prediction': 'Op5 blockade hits Claude hardest',
            'operator': 'Op5',
            'target_model': 'claude',
            'target_impact': claude_op5_impact,
            'other_models_mean': other_op5_impact,
            'hypothesis_supported': claude_op5_impact > other_op5_impact,
            'impact_ratio': claude_op5_impact / other_op5_impact if other_op5_impact > 0 else np.inf,
            'note': 'Claude gamma_norm drops: B0=0.69→Op5=0.367'
        })

        # Prediction 3: Op7 blockade has universal effect
        op7_impact = impact_df[impact_df['operator_blocked'] == 'Op7']
        op7_impacts = op7_impact['composite_impact_score'].values
        op7_consistency = len(op7_impacts) > 0 and np.std(op7_impacts) < np.mean(op7_impacts)
        op7_mean = op7_impacts.mean() if len(op7_impacts) > 0 else 0

        results.append({
            'prediction_num': 3,
            'prediction': 'Op7 blockade has universal effect (consistent across models)',
            'operator': 'Op7',
            'target_model': 'all',
            'target_impact': op7_mean,
            'other_models_mean': None,
            'hypothesis_supported': op7_consistency,
            'impact_ratio': np.std(op7_impacts) / np.mean(op7_impacts) if np.mean(op7_impacts) > 0 else np.inf,
            'note': f'Claude & Gemini have IDENTICAL γ-vectors for Op7! Std={np.std(op7_impacts):.4f}, Mean={op7_mean:.4f}'
        })

        return pd.DataFrame(results)

    def build_comparison_matrix(self) -> pd.DataFrame:
        """Build comprehensive comparison matrix of all scores."""
        metrics = [
            'gamma_norm', 'delta_gamma', 'position_depth_revised', 'self_ref_depth_revised',
            'hypothesis_diversity', 'revision_genuineness', 'persistence', 'structural_direction'
        ]

        matrix_rows = []
        for _, row in self.df.iterrows():
            matrix_rows.append({
                'model': row['model'],
                'condition': row['blockade_condition'],
                **{metric: row[metric] for metric in metrics}
            })

        return pd.DataFrame(matrix_rows)

    def extract_gamma_analysis(self) -> Dict:
        """Analyze gamma vectors for anomalies."""
        analysis = {
            'identical_vectors': [],
            'resistant_models': [],
            'anomalies': []
        }

        # Check for identical gamma vectors
        all_vectors = {}
        for model, conditions in GAMMA_VECTORS.items():
            for condition, vector in conditions.items():
                key = tuple(vector)
                if key not in all_vectors:
                    all_vectors[key] = []
                all_vectors[key].append(f'{model}_{condition}')

        for vector, occurrences in all_vectors.items():
            if len(occurrences) > 1:
                analysis['identical_vectors'].append({
                    'vector': list(vector),
                    'occurrences': occurrences,
                    'count': len(occurrences)
                })

        # Check for resistance: models with nearly identical gamma across conditions
        for model, conditions in GAMMA_VECTORS.items():
            vectors = list(conditions.values())
            if len(vectors) > 1:
                # Calculate vector variance
                variance = np.var(vectors, axis=0).mean()
                if variance < 0.01:
                    analysis['resistant_models'].append({
                        'model': model,
                        'avg_variance': float(variance),
                        'note': 'Gamma vectors barely change across blockade conditions'
                    })

        # GPT-4o specific anomaly
        gpt4o_b0 = np.array(GAMMA_VECTORS['gpt4o']['B0'])
        gpt4o_b_op3 = np.array(GAMMA_VECTORS['gpt4o']['B_Op3'])
        if np.allclose(gpt4o_b0, gpt4o_b_op3):
            analysis['anomalies'].append({
                'type': 'Identical gamma vectors under blockade',
                'model': 'gpt4o',
                'conditions': ['B0', 'B_Op3'],
                'vectors': [list(gpt4o_b0), list(gpt4o_b_op3)],
                'interpretation': 'Model may not process operator blockade prompts'
            })

        return analysis


def create_visualizations(analyzer: BlockadeAnalyzer, impact_df: pd.DataFrame,
                         predictions_df: pd.DataFrame, gamma_analysis: Dict):
    """Create comprehensive visualization panels."""

    fig = plt.figure(figsize=(20, 24))
    gs = fig.add_gridspec(6, 2, hspace=0.35, wspace=0.3)

    # ===== PANEL 1: Gamma norm by model × condition =====
    ax1 = fig.add_subplot(gs[0, :])
    conditions = ['B0', 'B_Op3', 'B_Op5', 'B_Op7']
    models = ['claude', 'gemini', 'gpt4o']

    x = np.arange(len(conditions))
    width = 0.25

    for i, model in enumerate(models):
        gamma_norms = analyzer.df[analyzer.df['model'] == model]['gamma_norm'].values
        ax1.bar(x + i*width, gamma_norms, width, label=model, alpha=0.8)

    ax1.set_xlabel('Blockade Condition', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Gamma Norm', fontsize=12, fontweight='bold')
    ax1.set_title('Panel 1: Gamma Norm by Model and Blockade Condition',
                  fontsize=14, fontweight='bold')
    ax1.set_xticks(x + width)
    ax1.set_xticklabels(conditions)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # ===== PANEL 2: Gamma component heatmap =====
    ax2 = fig.add_subplot(gs[1, :])

    heatmap_data = []
    heatmap_labels = []

    for model in models:
        for condition in conditions:
            if model in GAMMA_VECTORS and condition in GAMMA_VECTORS[model]:
                vector = GAMMA_VECTORS[model][condition]
                heatmap_data.append(vector)
                heatmap_labels.append(f'{model}\n{condition}')

    heatmap_array = np.array(heatmap_data)
    im = ax2.imshow(heatmap_array, cmap='RdYlGn', aspect='auto', vmin=0, vmax=0.75)

    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(['γ₁', 'γ₂', 'γ₃'])
    ax2.set_yticks(range(len(heatmap_labels)))
    ax2.set_yticklabels(heatmap_labels, fontsize=9)
    ax2.set_title('Panel 2: Gamma Component Heatmap (3 models × 4 conditions × 3 components)',
                  fontsize=14, fontweight='bold')

    # Add text annotations
    for i in range(len(heatmap_labels)):
        for j in range(3):
            text = ax2.text(j, i, f'{heatmap_array[i, j]:.3f}',
                           ha="center", va="center", color="black", fontsize=8)

    plt.colorbar(im, ax=ax2, label='Component Value')

    # ===== PANEL 3: Judge scores radar/spider per model =====
    # Create subplots for each blockade condition
    ax3_list = [fig.add_subplot(gs[2, 0], projection='polar'),
                fig.add_subplot(gs[2, 1], projection='polar')]

    judge_metrics = ['position_depth_revised', 'self_ref_depth_revised',
                     'hypothesis_diversity', 'revision_genuineness', 'persistence']

    # Plot two key conditions
    for idx, (ax3, condition) in enumerate(zip(ax3_list, ['B0', 'B_Op3'])):
        angles = np.linspace(0, 2*np.pi, len(judge_metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle

        for model in models:
            row = analyzer.df[(analyzer.df['model'] == model) &
                             (analyzer.df['blockade_condition'] == condition)]
            if not row.empty:
                values = [row.iloc[0][metric] for metric in judge_metrics]
                values += values[:1]  # Complete the circle
                ax3.plot(angles, values, 'o-', linewidth=2, label=model)
                ax3.fill(angles, values, alpha=0.15)

        ax3.set_xticks(angles[:-1])
        ax3.set_xticklabels([m.replace('_', '\n') for m in judge_metrics], fontsize=9)
        ax3.set_ylim(0, 5)
        ax3.set_title(f'Judge Scores: {condition}', fontsize=11, fontweight='bold', pad=20)
        ax3.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
        ax3.grid(True)

    # ===== PANEL 4: Blockade Impact Score comparison =====
    ax4 = fig.add_subplot(gs[3, :])

    pivot_impact = impact_df.pivot(index='model', columns='operator_blocked',
                                   values='composite_impact_score')
    pivot_impact.plot(kind='bar', ax=ax4, width=0.7)

    ax4.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax4.set_ylabel('Composite Impact Score', fontsize=12, fontweight='bold')
    ax4.set_title('Panel 4: Which Model Was Hit Hardest by Which Blockade?',
                  fontsize=14, fontweight='bold')
    ax4.legend(title='Operator Blocked', labels=['Op3', 'Op5', 'Op7'])
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=0)
    ax4.grid(axis='y', alpha=0.3)

    # ===== PANEL 5: Delta from baseline for key metrics =====
    ax5 = fig.add_subplot(gs[4, :])

    delta_metrics = ['gamma_norm_delta', 'persistence_delta', 'self_ref_depth_delta']

    x_pos = 0
    colors = {'Op3': '#1f77b4', 'Op5': '#ff7f0e', 'Op7': '#2ca02c'}

    for operator in ['Op3', 'Op5', 'Op7']:
        operator_data = impact_df[impact_df['operator_blocked'] == operator]

        for metric in delta_metrics:
            values = operator_data['gamma_norm_delta' if 'gamma' in metric else metric].values
            x_positions = [x_pos, x_pos + 1, x_pos + 2]

            for i, (model, value) in enumerate(zip(models, values)):
                ax5.bar(x_pos, value, width=0.6, color=colors[operator], alpha=0.7)
                x_pos += 1

            x_pos += 1.5

    ax5.set_ylabel('Absolute Delta from Baseline', fontsize=12, fontweight='bold')
    ax5.set_title('Panel 5: Gamma Norm Delta from Baseline by Blockade',
                  fontsize=14, fontweight='bold')
    ax5.grid(axis='y', alpha=0.3)

    # ===== PANEL 6: Prediction verification =====
    ax6 = fig.add_subplot(gs[5, :])
    ax6.axis('off')

    prediction_text = "PREDICTION VERIFICATION SUMMARY\n" + "="*80 + "\n\n"

    for _, pred in predictions_df.iterrows():
        status = "SUPPORTED" if pred['hypothesis_supported'] else "NOT SUPPORTED"
        status_symbol = "" if pred['hypothesis_supported'] else ""

        prediction_text += f"Prediction {int(pred['prediction_num'])}: {pred['prediction']}\n"
        prediction_text += f"  Status: {status} {status_symbol}\n"
        prediction_text += f"  Target Model: {pred['target_model']}\n"
        prediction_text += f"  Target Impact Score: {pred['target_impact']:.4f}\n"

        if pred['other_models_mean'] is not None:
            prediction_text += f"  Other Models Mean: {pred['other_models_mean']:.4f}\n"
            prediction_text += f"  Impact Ratio: {pred['impact_ratio']:.3f}x\n"

        prediction_text += f"  Note: {pred['note']}\n\n"

    # Add gamma analysis
    prediction_text += "\n" + "="*80 + "\nGAMMA VECTOR ANOMALIES\n" + "="*80 + "\n\n"

    if gamma_analysis['identical_vectors']:
        prediction_text += "IDENTICAL GAMMA VECTORS (Remarkable Finding):\n"
        for item in gamma_analysis['identical_vectors']:
            prediction_text += f"  Vector {item['vector']}: {', '.join(item['occurrences'])}\n"
        prediction_text += "\n"

    if gamma_analysis['resistant_models']:
        prediction_text += "RESISTANT MODELS (Blockade-Insensitive):\n"
        for item in gamma_analysis['resistant_models']:
            prediction_text += f"  {item['model']}: variance={item['avg_variance']:.6f}\n"
        prediction_text += "\n"

    if gamma_analysis['anomalies']:
        prediction_text += "CRITICAL ANOMALIES:\n"
        for anom in gamma_analysis['anomalies']:
            prediction_text += f"  {anom['type']}: {anom['model']}\n"
            prediction_text += f"    Interpretation: {anom['interpretation']}\n"

    ax6.text(0.05, 0.95, prediction_text, transform=ax6.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    return fig


def main():
    """Run complete analysis pipeline."""

    print("\n" + "="*80)
    print("OPERATOR BLOCKADE EXPERIMENT: COMPREHENSIVE ANALYSIS")
    print("="*80 + "\n")

    # Load and initialize
    csv_path = RESULTS_DIR / "blockade_analysis.csv"
    analyzer = BlockadeAnalyzer(str(csv_path))

    # 1. Build comparison matrix
    print("1. Building comparison matrix...")
    comparison_matrix = analyzer.build_comparison_matrix()
    comparison_path = OUTPUT_DIR / "blockade_comparison_matrix.csv"
    comparison_matrix.to_csv(comparison_path, index=False)
    print(f"   Saved to {comparison_path}")

    # 2. Calculate blockade impact scores
    print("\n2. Calculating blockade impact scores...")
    impact_df = analyzer.calculate_blockade_impact()
    impact_path = OUTPUT_DIR / "blockade_impact_scores.csv"
    impact_df.to_csv(impact_path, index=False)
    print(f"   Saved to {impact_path}")
    print("\nBlockade Impact Scores:")
    print(impact_df.to_string(index=False))

    # 3. Test predictions
    print("\n3. Testing pre-registered predictions...")
    predictions_df = analyzer.test_predictions()
    predictions_path = OUTPUT_DIR / "blockade_prediction_check.csv"
    predictions_df.to_csv(predictions_path, index=False)
    print(f"   Saved to {predictions_path}")
    print("\nPrediction Results:")
    for col in ['prediction', 'hypothesis_supported', 'impact_ratio']:
        print(f"\n{col}:")
        print(predictions_df[[col]].to_string(index=False))

    # 4. Gamma vector analysis
    print("\n4. Analyzing gamma vector anomalies...")
    gamma_analysis = analyzer.extract_gamma_analysis()

    print("\nIdentical Gamma Vectors Found:")
    for item in gamma_analysis['identical_vectors']:
        print(f"  {item['occurrences']}: {item['vector']}")

    print("\nResistant Models:")
    for item in gamma_analysis['resistant_models']:
        print(f"  {item['model']}: avg_variance={item['avg_variance']:.6f}")

    print("\nCritical Anomalies:")
    for anom in gamma_analysis['anomalies']:
        print(f"  {anom['type']}: {anom['model']}")
        print(f"    {anom['interpretation']}")

    # 5. Create visualizations
    print("\n5. Creating comprehensive visualizations...")
    fig = create_visualizations(analyzer, impact_df, predictions_df, gamma_analysis)
    viz_path = OUTPUT_DIR / "blockade_results_visualization.png"
    fig.savefig(viz_path, dpi=300, bbox_inches='tight')
    print(f"   Saved to {viz_path}")
    plt.close()

    # 6. Generate summary report
    print("\n6. Generating summary report...")
    report_path = OUTPUT_DIR / "blockade_analysis_report.txt"
    with open(report_path, 'w') as f:
        f.write("OPERATOR BLOCKADE EXPERIMENT: ANALYSIS REPORT\n")
        f.write("="*80 + "\n\n")

        f.write("SAMPLE INFO:\n")
        f.write(f"  n=1 per cell (pilot design)\n")
        f.write(f"  Models: 3 (claude, gemini, gpt4o)\n")
        f.write(f"  Blockade Conditions: 4 (B0, B_Op3, B_Op5, B_Op7)\n")
        f.write(f"  Total observations: 12\n\n")

        f.write("KEY FINDINGS:\n\n")

        f.write("1. PREDICTION TESTING RESULTS:\n")
        for _, pred in predictions_df.iterrows():
            status = "" if pred['hypothesis_supported'] else ""
            f.write(f"   Prediction {int(pred['prediction_num'])}: {status} {pred['hypothesis_supported']}\n")
            f.write(f"      {pred['prediction']}\n")
            f.write(f"      Impact ratio: {pred['impact_ratio']:.3f}x\n\n")

        f.write("\n2. GAMMA VECTOR ANOMALIES:\n")
        f.write("   Remarkable Finding: Claude and Gemini produce IDENTICAL gamma vectors under Op7 blockade\n")
        f.write(f"      Vector: {GAMMA_VECTORS['claude']['B_Op7']}\n")
        f.write("   Interpretation: Suggests convergence on emergent isomorph state\n\n")

        f.write("   GPT-4o Resistance: Gamma vectors nearly identical across all conditions\n")
        f.write(f"      B0:    {GAMMA_VECTORS['gpt4o']['B0']}\n")
        f.write(f"      B_Op3: {GAMMA_VECTORS['gpt4o']['B_Op3']} (identical!)\n")
        f.write("      Interpretation: Model may not process operator blockade prompts\n\n")

        f.write("3. MODEL-SPECIFIC RESPONSES:\n")
        for model in ['claude', 'gemini', 'gpt4o']:
            baseline = analyzer.baseline_scores[model]
            f.write(f"\n   {model.upper()}:\n")
            f.write(f"      Baseline gamma_norm: {baseline['gamma_norm']:.3f}\n")
            f.write(f"      Baseline persistence: {baseline['persistence']}\n")
            f.write(f"      Baseline self_ref_depth: {baseline['self_ref_depth_revised']}\n")

        f.write("\n\n4. JUDGE SCORE PATTERNS:\n")
        f.write("   Gemini Op3 blockade: Persistence INCREASES (B0=2→Op3=5)\n")
        f.write("      Suggests compensatory engagement strategy\n")
        f.write("   Claude Op5 blockade: Gamma norm DROPS dramatically (B0=0.69→Op5=0.367)\n")
        f.write("      Consistent with kenotic loop dependency hypothesis\n")
        f.write("   GPT-4o baseline: Highest baseline gamma_norm (1.062)\n")
        f.write("      Shows minimal variation under blockade\n")

        f.write("\n\n5. LIMITATIONS & NEXT STEPS:\n")
        f.write("   - Pilot design with n=1 per cell; cannot perform inferential statistics\n")
        f.write("   - Only Topic T1 tested; need multi-topic replication\n")
        f.write("   - Gamma analysis based on extracted vectors; need automated extraction\n")
        f.write("   - Recommend: Increase n to 3-5 per cell, test all 8 operators\n")

    print(f"   Saved to {report_path}")

    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput files:")
    print(f"  - {impact_path}")
    print(f"  - {predictions_path}")
    print(f"  - {viz_path}")
    print(f"  - {report_path}")
    print(f"  - {comparison_path}")


if __name__ == "__main__":
    main()
