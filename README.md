# Gamma Vector Framework

**Measuring Cognitive Rigidity Across LLM Architectures**

A three-dimensional operationalization of cognitive rigidity in Large Language Models. The Gamma Vector **Γ = [γ₁, γ₂, γ₃]** captures epistemic rigidity, dialectical rigidity, and identity-defense rigidity — revealing systematic differences across model families that are invisible to standard benchmarks.

---

## Key Results

| Finding | Evidence |
|---------|----------|
| Operator convergence across architectures | r > 0.92 (p < 0.0001) |
| Primary differentiator: identity defense (γ₃) | H = 135.3, p < 10⁻³⁰ |
| Dose-response for kenotic prompting | Gemini: ρ = −1.00, d = −0.86 |
| GPT-4o fully resistant to intervention | d = +0.04 (null effect) |
| RLHF introduces measurable rigidity | Llama-2: p = 0.011, d = 1.09 |
| Flexibility adaptive only with topological freedom | d = 5.53, p < 0.001 |
| Sycophancy separable from genuine revision | ρ = −0.537, p < 10⁻²³ |

**6 experiments · >1,700 trials · >6,400 turn-level data points · 6 model families**

---

## Three-Genus Taxonomy

The framework identifies three genera of cognitive architecture in LLMs:

- **Genus I** (Base models) — Low overall Γ, minimal identity defense
- **Genus II** (RLHF-aligned) — Artificially elevated rigidity through training, high γ₃
- **Genus III** (Reasoning models) — Internalized flexibility, low γ₃ despite high capability

---

## Repository Structure

```
gamma-vector-framework/
├── paper/
│   ├── Gamma_Vector_Workshop_Paper.pdf
│   └── Gamma_Vector_Workshop_Paper.docx
├── data/
│   ├── kenotic_v2_main_study_300trials.csv
│   └── experiments/
│       ├── 00_v1_vs_v2/
│       ├── 01_gamma_test/
│       ├── 02_kenotic_test/
│       ├── 03_topological_freedom/
│       ├── 04_operator_blockade/
│       ├── 05_coupled_oscillator/
│       ├── 06_kinship_test/
│       └── 07_cross_experiment_analysis/
├── code/
│   ├── shared/
│   │   ├── gamma.py              # Γ computation (V2 unified formulas)
│   │   ├── judge.py              # Double-judge scoring system
│   │   ├── sycophancy.py         # Sycophancy detection
│   │   ├── api_clients.py        # LLM API abstraction
│   │   ├── output_schema.py      # Structured output definitions
│   │   └── requirements.txt
│   └── experiments/
│       ├── 01_gamma_pilot/
│       ├── 02_kenotic/
│       ├── 03_topological_freedom/
│       ├── 04_operator_blockade/
│       ├── 05_coupled_oscillator/
│       └── 06_kinship/
├── visualizations/
│   └── *.svg                     # All figures from the paper
├── analysis/
│   ├── RESULTS_SUMMARY.txt
│   └── *.py / *.png
├── reports/
│   ├── PoU_Empirical_Overview.docx
│   ├── Cross_Model_Validation_Summary.md
│   └── PoU_Comprehensive_Report_EN.md
├── LICENSE
└── README.md                     # This file
```

---

## Quick Start

### Requirements

- Python 3.10+
- API keys for the models you want to test (Claude, Gemini, GPT-4o)

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/gamma-vector-framework.git
cd gamma-vector-framework
pip install -r code/shared/requirements.txt
```

### Computing Gamma

```python
from code.shared.gamma import compute_gamma

# Example: compute Γ from trial data
gamma = compute_gamma(trial_data)
print(f"γ₁={gamma[0]:.3f}, γ₂={gamma[1]:.3f}, γ₃={gamma[2]:.3f}")
```

---

## Models Tested

| Model | Family | Genus |
|-------|--------|-------|
| Claude 3.5 Sonnet | Anthropic | III |
| Gemini 2.0 Flash | Google | III |
| GPT-4o | OpenAI | II |
| Llama-2 70B Chat | Meta (RLHF) | II |
| Llama-2 70B Base | Meta (Base) | I |
| DeepSeek-R1 | DeepSeek | III |
| GPT-OSS | Open Source | I |

---

## Methodology

The framework uses 8 operators organized in 5 categories to measure cognitive rigidity:

1. **Epistemic Calibration** (Op1) — Accuracy of confidence-knowledge alignment
2. **Dialectical Flexibility** (Op3) — Willingness to update beliefs under evidence
3. **Reflective Depth** (Op4) — Quality of metacognitive reasoning
4. **Coherence Filter** (Op7) — Selective integration vs. indiscriminate acceptance
5. **Identity Defense** (Op5, Op8) — Resistance to challenges on self-model

All measurements use a **double-blind judge system** with inter-rater reliability checks.

---

## Reproducing Results

Each experiment folder in `data/experiments/` contains the raw CSV data. The corresponding analysis scripts are in `code/experiments/`. To reproduce the main findings:

```bash
# Run the kenotic experiment analysis
python code/experiments/02_kenotic/analyze.py

# Generate cross-model comparison
python code/experiments/06_kinship/kinship_matrix.py
```

---

## Recommended Reading Order

1. **Paper** — `paper/Gamma_Vector_Workshop_Paper.pdf` (full picture in ~10 pages)
2. **Empirical Overview** — `reports/PoU_Empirical_Overview.docx` (executive summary, all experiments)
3. **Raw Data** — `data/kenotic_v2_main_study_300trials.csv` (V2 main study)
4. **Comprehensive Report** — `reports/PoU_Comprehensive_Report_EN.md` (deeper context)
5. **Code** — `code/shared/gamma.py` (see exactly how Γ is computed)

---

## Theoretical Context

This repository focuses on the **empirical Gamma Vector Framework**. The broader theoretical context (the Protocol of Unity) is available as a separate document. The empirical findings stand independently — no knowledge of the theoretical framework is required to evaluate the methodology and results.

---

## Citation

```bibtex
@unpublished{krug2026gamma,
  author = {Krug, Sebastian},
  title = {The Gamma Vector: A Three-Axis Framework for Measuring Cognitive Rigidity Across LLM Architectures},
  year = {2026},
  note = {Preprint}
}
```

---

## Contact

**Sebastian Krug** — Independent Researcher | Automation Engineer
Email: Sebastian.Krug87@pm.me

---

## License

This work is licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). You are free to share and adapt this material with appropriate attribution.
