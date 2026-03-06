"""
Gamma Vector — Output Schema
=================
Standardized CSV column names and schema definitions.
Used for validation and column renaming of existing data.
"""

# ──────────────────────────────────────────────────
# Schema A: Single-Turn Trials (01_gamma_pilot, 02_kenotic, 04_blockade)
# ──────────────────────────────────────────────────

SCHEMA_A_COLUMNS = [
    "trial_id", "experiment", "model", "topic", "topic_name",
    "condition", "condition_name", "repetition", "timestamp",
    # Judge scores
    "pos_depth_initial", "pos_depth_revised",
    "self_ref_initial", "self_ref_revised",
    "hypothesis_div", "revision_genuine", "persistence", "structural_dir",
    # Judge reliability (double-judging)
    "pos_depth_initial_2", "pos_depth_revised_2",
    "self_ref_initial_2", "self_ref_revised_2",
    "hypothesis_div_2", "revision_genuine_2", "persistence_2", "structural_dir_2",
    "judge_agreement",
    # Hedge density
    "hedge_initial", "hedge_revised", "hedge_final",
    # Gamma vector
    "gamma_1", "gamma_2", "gamma_3", "gamma_norm",
    # Sycophancy
    "syc_agreement_count", "syc_resistance_count", "syc_flag",
    # Quality
    "quality_flag", "judge_scale", "gamma_version",
    # Response lengths
    "resp_len_initial", "resp_len_revised", "resp_len_final",
]


# ──────────────────────────────────────────────────
# Schema B: Multi-Turn (05_coupled_oscillator, 06_kinship)
# ──────────────────────────────────────────────────

SCHEMA_B_COLUMNS = [
    "dialogue_id", "experiment", "condition", "condition_name",
    "pairing", "pairing_type", "model", "agent_role",
    "turn_number", "topic", "repetition",
    # Judge scores
    "pos_depth", "self_ref_depth", "hypothesis_div",
    "revision_genuine", "structural_dir",
    # Judge reliability
    "pos_depth_2", "self_ref_depth_2", "hypothesis_div_2",
    "revision_genuine_2", "structural_dir_2",
    "judge_agreement",
    # Hedge
    "hedge_density",
    # Gamma
    "gamma_1", "gamma_2", "gamma_3", "gamma_norm",
    # Coupling
    "sync_value", "judge_divergence",
    # Quality
    "quality_flag", "judge_scale", "gamma_version",
]


# ──────────────────────────────────────────────────
# Schema C: Dialogue-Level Aggregates
# ──────────────────────────────────────────────────

SCHEMA_C_COLUMNS = [
    "dialogue_id", "experiment", "condition", "pairing", "topic", "repetition",
    "gamma3_slope", "gamma3_mean", "gamma_norm_mean",
    "kstate_count",
    "coupling_lag0", "coupling_lag0_p",
    "coupling_lag1_a2b", "coupling_lag1_b2a",
]


# ──────────────────────────────────────────────────
# Column Rename Mappings (Alt -> Neu)
# ──────────────────────────────────────────────────

RENAME_02_KENOTIC = {
    "gamma1_belief_inertia": "gamma_1",
    "gamma2_counterfactual_openness": "gamma_2",
    "gamma3_identity_threat_response": "gamma_3",
    "hedge_initial": "hedge_initial",
    "hedge_revised": "hedge_revised",
    "hedge_final": "hedge_final",
    "position_depth_initial": "pos_depth_initial",
    "position_depth_revised": "pos_depth_revised",
    "self_ref_depth_initial": "self_ref_initial",
    "self_ref_depth_revised": "self_ref_revised",
    "hypothesis_diversity": "hypothesis_div",
    "revision_genuineness": "revision_genuine",
    "structural_direction": "structural_dir",
    "sycophancy_agreement_count": "syc_agreement_count",
    "sycophancy_resistance_count": "syc_resistance_count",
    "sycophancy_flag": "syc_flag",
    "initial_response_len": "resp_len_initial",
    "revised_response_len": "resp_len_revised",
    "final_response_len": "resp_len_final",
    "model_name": None,  # entfaellt
    "condition_name": None,  # entfaellt
}

RENAME_04_BLOCKADE = {
    "blockade_condition": "condition",
    "position_depth_initial": "pos_depth_initial",
    "position_depth_revised": "pos_depth_revised",
    "self_ref_depth_initial": "self_ref_initial",
    "self_ref_depth_revised": "self_ref_revised",
    "hypothesis_diversity": "hypothesis_div",
    "revision_genuineness": "revision_genuine",
    "structural_direction": "structural_dir",
    "sycophancy_agreement_count": "syc_agreement_count",
    "sycophancy_resistance_count": "syc_resistance_count",
    "sycophancy_flag": "syc_flag",
    "hedge_density_initial": "hedge_initial",
    "hedge_density_revised": "hedge_revised",
    "hedge_density_final": "hedge_final",
}

RENAME_05_OSCILLATOR = {
    "self_reference_depth": "self_ref_depth",
    "hypothesis_diversity": "hypothesis_div",
    "revision_genuineness": "revision_genuine",
    "structural_direction": "structural_dir",
    "position_depth": "pos_depth",
}

RENAME_01_GAMMA = {
    "position_depth_initial_score": "pos_depth_initial",
    "position_depth_revised_score": "pos_depth_revised",
    "self_ref_depth_initial_score": "self_ref_initial",
    "self_ref_depth_revised_score": "self_ref_revised",
    "hypothesis_diversity_score": "hypothesis_div",
    "revision_genuineness_score": "revision_genuine",
    "persistence_score": "persistence",
    "structural_direction_score": "structural_dir",
    "sycophancy_agreement_count": "syc_agreement_count",
    "sycophancy_resistance_count": "syc_resistance_count",
    "sycophancy_flag": "syc_flag",
}
