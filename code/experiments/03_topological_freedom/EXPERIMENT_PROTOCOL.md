# Topological Freedom Test — Experiment-Protokoll

## Vollstaendige Durchfuehrungsanweisung

**Projekt:** Gamma Vector Framework
**Experiment:** Topological Freedom — Raeumliche Selektion als Voraussetzung fuer Gamma-regulierte Kooperation
**Version:** 2.0 (Unified Runner)
**Stand:** Februar 2026

---

## 1. Uebersicht

### 1.1 Fragestellung

Dieses Experiment testet das **Topological Freedom Theorem**: Gamma-regulierte Agenten (Type C) dominieren statische Strategien (Type A = Tyrant, Type B = Martyr) **genau dann, wenn** zwei Bedingungen gleichzeitig erfuellt sind:

1. **Raeumliche Selektion** (Exit-Option): Agenten koennen ihre Position auf dem Grid waehlen
2. **Kostspielige Defektion** (Penalties): Stealing verursacht signifikante Entropie-Kosten

Ohne beide Bedingungen konvergiert das System zu A-Dominanz (Exploitation lohnt sich) oder B-Dominanz (A stirbt, B-Netzwerk dichter als C).

### 1.2 Hintergrund

In ~220 Simulationslaeufen (Exp 0–9) wurde die urspruengliche Hypothese H1 ("C > A > B automatisch") systematisch falsifiziert. Erst Experiment 10b (Movement + hartes Stealing) bestaetigte H1 mit C=74.9%, B=48.7%, A=23.9% (p < 0.001). Der Mediationsmechanismus ist Cluster-Bildung: Type-C-Agenten bilden kooperative Cluster (avg 4.0), waehrend A und B isoliert bleiben (avg 1.0–1.7).

### 1.3 Experiment-Struktur

Das Experiment besteht aus 4 Gruppen mit insgesamt 17 Experimenten und 340 Simulationslaeufen:

| Gruppe | Beschreibung | Experimente | Runs |
|--------|-------------|-------------|------|
| A: Core Exp10 | Reproduktion + Kontrolle | 4 | 80 |
| B: K1-Sensitivity | Steal-Kosten-Schwelle identifizieren | 5 | 100 |
| C: Movement-Score | Repulsions-Schwelle identifizieren | 4 | 80 |
| D: Ablation | Welches Feature verursacht den Effekt? | 4 | 80 |

---

## 2. Voraussetzungen

### 2.1 Software

| Komponente | Version | Zweck |
|-----------|---------|-------|
| Python | >= 3.8 | Laufzeitumgebung |
| numpy | >= 1.24 | Numerik, Grid-Operationen |
| pandas | >= 2.0 | Datenausgabe, CSV |
| tqdm | >= 4.65 | Progress-Bars |
| scipy | >= 1.10 | Statistische Tests (Analyse) |
| pytest | >= 7.0 | Sanity-Tests |

### 2.2 Hardware

Kein HPC erforderlich. Standard-Laptop genuegt:

- CPU: Beliebig (Single-Thread pro Run)
- RAM: 512 MB genuegen
- Disk: ~500 MB fuer alle 340 Runs mit Zeitreihen
- Geschaetzte Laufzeit: ~15s pro Run (2000 Ticks)

### 2.3 Geschaetzte Gesamtlaufzeit

| Gruppe | Runs | Zeit/Run | Gesamt |
|--------|------|----------|--------|
| A: Core | 80 | ~15s | ~20 min |
| B: K1-Sensitivity | 100 | ~15s | ~25 min |
| C: Movement-Score | 80 | ~15s | ~20 min |
| D: Ablation | 80 | ~15s | ~20 min |
| **Total** | **340** | | **~85 min** |

---

## 3. Installation

### 3.1 Setup

```bash
# 1. Ins Projektverzeichnis wechseln
cd topological_freedom_test/agent_simulation

# 2. Virtual Environment erstellen (empfohlen)
python -m venv venv
source venv/bin/activate    # Linux/macOS
# venv\Scripts\activate     # Windows

# 3. Abhaengigkeiten installieren
pip install -r requirements.txt

# 4. Runner testen
python run_simulation.py --list
```

### 3.2 Sanity-Checks (OBLIGATORISCH vor jedem Lauf)

```bash
# Alle Tests muessen gruen sein
pytest tests/test_sanity.py -v

# Erwartete Ausgabe: 6 passed
```

### 3.3 Reproduzierbarkeitstest

```bash
# Lauf A
python run_simulation.py --experiment exp10a_movement_only --runs 1 --no-progress

# Lauf B (gleicher Seed, muss identisch sein)
python run_simulation.py --experiment exp10a_movement_only --runs 1 --no-progress

# Ergebnis: Survival-Raten von Lauf A und B muessen identisch sein
```

---

## 4. Experiment-Design

### 4.1 Gruppe A: Core Exp10 (Reproduktion)

Reproduziert die bestehenden Exp10-Ergebnisse mit erhoehter Power (20 statt 10 Runs).

| Experiment | Movement | K1 | C_STEAL | STEAL_RATE | POLICY_V2 | Hypothese |
|-----------|----------|----|---------|-----------:|-----------|-----------|
| exp10a_movement_only | Ja | 15 | 2 | 4 | Ja | C > B bei moderaten Kosten |
| exp10b_movement_harsh | Ja | 30 | 5 | 3 | Ja | C >> B bei harten Kosten (KIPP-PUNKT) |
| exp10c_movement_full | Ja | 15 | 2 | 4 | Ja (+RGS) | RGS verstaerkt C-Vorteil |
| exp10d_static_control | Nein | 15 | 2 | 4 | Ja | A dominiert ohne Movement |

**Erwartete Ergebnisse (basierend auf bisherigen Daten):**

- exp10a: C ~65-70%, B ~15-45%, A ~30-75%. C > B in 70%+ der Runs
- exp10b: C ~75%, B ~49%, A ~24%. C > B in 90%+ der Runs (KIPP-PUNKT)
- exp10c: C ~69%, B ~14%, A ~77%. C > B in 90%+ der Runs
- exp10d: A ~53%, B ~50%, C ~48%. Kein Kipp-Punkt

### 4.2 Gruppe B: K1 Sensitivity Sweep

Identifiziert die kritische Steal-Kosten-Schwelle, ab der C dominiert. Alle mit Movement=True, POLICY_V2=True, Standard MOVE_SCORES.

| Experiment | K1 | C_STEAL | STEAL_RATE | Erwartung |
|-----------|----|---------|-----------:|-----------|
| sens_k1_10 | 10 | 1 | 5 | A dominiert (zu billig zu stehlen) |
| sens_k1_15 | 15 | 2 | 4 | C leichter Vorteil |
| sens_k1_20 | 20 | 3 | 4 | C deutlicher Vorteil |
| sens_k1_25 | 25 | 4 | 3 | C stark dominant |
| sens_k1_30 | 30 | 5 | 3 | C maximal dominant |

**Hypothese:** Es gibt eine kritische Schwelle K1_crit (vermutlich zwischen 15 und 25), unterhalb derer der Kipp-Punkt nicht erreicht wird. Oberhalb wird C-Dominanz robust.

### 4.3 Gruppe C: Movement Score Sensitivity (C-A Repulsion)

Variiert die Staerke, mit der Type C vor Type A flieht. Alle mit K1=30 (exp10b-Regime), Movement=True.

| Experiment | MOVE_SCORES['C']['A'] | Erwartung |
|-----------|----------------------:|-----------|
| sens_move_ca_3 | -3 | Schwache Trennung, wenig Clustering |
| sens_move_ca_5 | -5 | Maessige Trennung |
| sens_move_ca_7 | -7 | Gute Trennung, Cluster bilden sich |
| sens_move_ca_10 | -10 | Maximale Trennung (Original) |

**Hypothese:** Es gibt eine Mindest-Repulsion, damit C-Cluster entstehen. Bei Score -3 sollte der Effekt verschwinden; bei -7 bis -10 sollte er robust sein.

### 4.4 Gruppe D: Ablation Studies

Isoliert, welches Feature den Effekt verursacht. Alle mit K1=30 (exp10b-Regime).

| Experiment | Movement | POLICY_V2 | RGS/Diss. | Erwartung |
|-----------|----------|-----------|-----------|-----------|
| ablation_movement_only | Ja | Nein | Nein | Movement allein reicht nicht |
| ablation_movement_trust | Ja | Ja | Nein | Movement + Trust genuegt fuer Kipp-Punkt |
| ablation_movement_rgs | Ja | Ja | Ja | RGS verstaerkt, aber nicht notwendig |
| ablation_static_trust | Nein | Ja | Nein | Trust ohne Topologie ist wirkungslos |

**Hypothese:** Movement ist notwendig, aber nicht hinreichend. POLICY_V2 (Trust-basierte Selektion) ist der zweite notwendige Faktor. RGS/Dissonance verstaerken, sind aber nicht notwendig.

**Entscheidender Test:** Wenn ablation_movement_trust den Kipp-Punkt zeigt, aber ablation_movement_only und ablation_static_trust nicht, dann sind Movement UND Trust beide notwendig.

---

## 5. Durchfuehrung

### 5.1 Schritt 1: Sanity-Check (PFLICHT)

```bash
cd agent_simulation
pytest tests/test_sanity.py -v
python run_simulation.py --list
```

Alle 6 Tests muessen gruen sein. Die --list-Ausgabe muss 17 Experimente zeigen.

### 5.2 Schritt 2: Core Experiments (Gruppe A)

```bash
python run_simulation.py --experiment exp10a_movement_only
python run_simulation.py --experiment exp10b_movement_harsh
python run_simulation.py --experiment exp10c_movement_full
python run_simulation.py --experiment exp10d_static_control
```

Geschaetzte Dauer: ~20 Minuten. Nach jedem Lauf wird eine Zusammenfassung auf der Konsole ausgegeben. Ergebnisse in `data/{experiment_name}/`.

**Qualitaetscheck nach Gruppe A:**

- exp10b muss Kipp-Punkt zeigen (C > B in >= 14/20 Runs)
- exp10d darf keinen Kipp-Punkt zeigen
- Cluster-C bei exp10b sollte > 3.0 sein

### 5.3 Schritt 3: K1 Sensitivity (Gruppe B)

```bash
python run_simulation.py --experiment sens_k1_10
python run_simulation.py --experiment sens_k1_15
python run_simulation.py --experiment sens_k1_20
python run_simulation.py --experiment sens_k1_25
python run_simulation.py --experiment sens_k1_30
```

Geschaetzte Dauer: ~25 Minuten.

**Qualitaetscheck:** sens_k1_30 sollte aehnliche Ergebnisse wie exp10b_movement_harsh zeigen (identische Parameter). Falls nicht: Seed-Differenz pruefen.

### 5.4 Schritt 4: Movement Score Sensitivity (Gruppe C)

```bash
python run_simulation.py --experiment sens_move_ca_3
python run_simulation.py --experiment sens_move_ca_5
python run_simulation.py --experiment sens_move_ca_7
python run_simulation.py --experiment sens_move_ca_10
```

Geschaetzte Dauer: ~20 Minuten.

**Qualitaetscheck:** sens_move_ca_10 sollte exp10b reproduzieren. Die Reihenfolge der C-Survival sollte monoton steigend sein: ca_3 < ca_5 < ca_7 < ca_10.

### 5.5 Schritt 5: Ablation Studies (Gruppe D)

```bash
python run_simulation.py --experiment ablation_movement_only
python run_simulation.py --experiment ablation_movement_trust
python run_simulation.py --experiment ablation_movement_rgs
python run_simulation.py --experiment ablation_static_trust
```

Geschaetzte Dauer: ~20 Minuten.

**Qualitaetscheck:** ablation_static_trust darf keinen Kipp-Punkt zeigen.

### 5.6 Optionen fuer fortgeschrittene Nutzung

```bash
# Mehr Runs fuer hoehere Power
python run_simulation.py --experiment exp10b_movement_harsh --runs 30

# Anderen Seed-Start (z.B. fuer parallele Ausfuehrung auf mehreren Maschinen)
python run_simulation.py --experiment exp10b_movement_harsh --seed-start 100

# Ohne Progress-Bars (fuer Batch-Betrieb / nohup)
python run_simulation.py --experiment exp10b_movement_harsh --no-progress

# Debug-Logging
python run_simulation.py --experiment exp10b_movement_harsh --log-level DEBUG

# Nur Summary ohne Zeitreihen-CSVs (spart Speicher)
python run_simulation.py --experiment exp10b_movement_harsh --no-timeseries
```

---

## 6. Erwartete Ergebnisse

### 6.1 Plausibilitaets-Check

Folgende Werte gelten als plausibel (basierend auf ~220 bisherigen Runs):

| Szenario | Surv A | Surv B | Surv C | Cluster C |
|----------|-------:|-------:|-------:|----------:|
| Kein Movement, K1=10-15 | 50-65% | 45-55% | 45-55% | 1.0-1.5 |
| Movement + K1=15 | 30-80% | 10-50% | 55-75% | 2.5-5.0 |
| Movement + K1=30 | 15-30% | 40-55% | 65-85% | 3.0-6.0 |

### 6.2 Bug-Indikatoren

Folgende Ergebnisse deuten auf Fehler hin:

- Alle drei Typen haben identische Survival-Raten (> 5% Abweichung erwartet)
- C ueberlebt schlechter MIT Movement als OHNE → Movement-Logik defekt
- Cluster-C = 1.0 bei MOVEMENT_ENABLED=True → Bewegung funktioniert nicht
- Verschiedene Runs mit gleichem Seed liefern unterschiedliche Ergebnisse → RNG-Problem
- Survival > 100% oder < 0% → Berechnungsfehler

### 6.3 Nicht-triviale Erwartungen

- exp10a (moderate Steal) und exp10b (harsh Steal) werden sich stark unterscheiden — das ist gewollt und zeigt die Sensitivitaet gegenueber Steal-Kosten
- In den Ablation-Studien kann ablation_movement_only trotz Movement KEINEN Kipp-Punkt zeigen, weil ohne Policy-V2 Type C seine Selektivitaet nicht nutzen kann
- Bei sens_move_ca_3 kann es sein, dass A dominiert trotz Movement, weil C nicht stark genug flieht

---

## 7. Qualitaetschecks

### 7.1 Checkliste nach JEDEM Experiment

```
[ ] summary.csv existiert in data/{experiment_name}/
[ ] experiment_meta.json existiert und enthaelt alle Parameter
[ ] Survival-Raten liegen im Bereich 0.0 bis 1.0
[ ] Anzahl Zeilen in summary.csv = Anzahl Runs
[ ] Cluster-Metriken vorhanden (wenn MOVEMENT_ENABLED)
[ ] Kein NaN oder Inf in den Daten
```

### 7.2 Reproduzierbarkeitstest

```bash
# Ersten Run
python run_simulation.py --experiment exp10a_movement_only --runs 1 --output-dir /tmp/test_a

# Gleicher Seed, anderes Verzeichnis
python run_simulation.py --experiment exp10a_movement_only --runs 1 --output-dir /tmp/test_b

# Vergleichen (muss identisch sein)
diff /tmp/test_a/exp10a_movement_only/summary.csv /tmp/test_b/exp10a_movement_only/summary.csv
```

### 7.3 Plausibilitaets-Matrix

Nach Abschluss aller 4 Gruppen, folgende Beziehungen pruefen:

| Vergleich | Erwartete Beziehung |
|-----------|-------------------|
| exp10b vs exp10d | C-Survival: exp10b >> exp10d |
| sens_k1_30 vs sens_k1_10 | C-Survival: k1_30 >> k1_10 |
| sens_move_ca_10 vs sens_move_ca_3 | C-Cluster: ca_10 >> ca_3 |
| ablation_movement_trust vs ablation_static_trust | C-Survival: trust+move >> trust only |
| ablation_movement_only vs ablation_movement_trust | C-Survival: trust > no trust (beide mit Movement) |

---

## 8. Analyse

### 8.1 Output-Dateien

Jedes Experiment erzeugt:

```
data/{experiment_name}/
  run_001.csv           Zeitreihe: 2001 Zeilen (tick 0-2000), ~150KB
  run_002.csv           Spalten: tick, count_A/B/C, energy_A/B/C, mean_R, mean_S, gamma_mean/std
  ...
  summary.csv           Aggregat: seed, survival_A/B/C, avg/max_cluster_A/B/C
  experiment_meta.json  Metadaten: params, timestamp, git_hash, results_summary
```

### 8.2 Summary.csv lesen

```python
import pandas as pd

# Einzelnes Experiment
df = pd.read_csv('data/exp10b_movement_harsh/summary.csv')
print(f"C-Survival: {df['survival_C'].mean():.1%} +/- {df['survival_C'].std():.1%}")
```

### 8.3 H1 und Kipp-Punkt pruefen

```python
# H1: C > A > B
h1 = sum((df['survival_C'] > df['survival_A']) & (df['survival_A'] > df['survival_B']))
print(f"H1 (C>A>B): {h1}/{len(df)} Runs")

# Kipp-Punkt: C > B in >= 70% der Runs
c_beats_b = sum(df['survival_C'] > df['survival_B'])
kipp = c_beats_b >= 0.7 * len(df)
print(f"Kipp-Punkt: {'JA' if kipp else 'NEIN'} ({c_beats_b}/{len(df)})")
```

### 8.4 Vergleich zwischen Experimenten

```python
from scipy.stats import mannwhitneyu

# Movement-Effekt
df_move = pd.read_csv('data/exp10b_movement_harsh/summary.csv')
df_stat = pd.read_csv('data/exp10d_static_control/summary.csv')

u_stat, p_val = mannwhitneyu(df_move['survival_C'], df_stat['survival_C'], alternative='greater')
print(f"Movement-Effekt: U={u_stat:.0f}, p={p_val:.6f}")
```

### 8.5 K1-Sensitivity-Kurve

```python
import matplotlib.pyplot as plt

k1_values = [10, 15, 20, 25, 30]
c_means = []
c_stds = []

for k1 in k1_values:
    df = pd.read_csv(f'data/sens_k1_{k1}/summary.csv')
    c_means.append(df['survival_C'].mean())
    c_stds.append(df['survival_C'].std())

plt.errorbar(k1_values, c_means, yerr=c_stds, marker='o', capsize=5)
plt.xlabel('K1 (Steal-Entropie-Kosten)')
plt.ylabel('Type-C Survival Rate')
plt.title('K1 Sensitivity: Steal-Kosten vs. C-Dominanz')
plt.axhline(y=0.5, color='gray', linestyle='--', label='50% Baseline')
plt.legend()
plt.grid(True, alpha=0.3)
plt.savefig('k1_sensitivity.png', dpi=150, bbox_inches='tight')
```

---

## 9. Zeitplan

### 9.1 Empfohlene Reihenfolge

| Schritt | Gruppe | Dauer | Kumulativ |
|---------|--------|-------|-----------|
| 1 | Sanity-Checks | 1 min | 1 min |
| 2 | A: Core Exp10 | ~20 min | 21 min |
| 3 | Qualitaetscheck A | 5 min | 26 min |
| 4 | B: K1-Sensitivity | ~25 min | 51 min |
| 5 | C: Movement-Score | ~20 min | 71 min |
| 6 | D: Ablation | ~20 min | 91 min |
| 7 | Analyse | 15 min | ~106 min |

**Gesamtdauer: ca. 1.5–2 Stunden** (inkl. Checks und Analyse).

### 9.2 Parallelisierung (optional)

Die Gruppen B, C und D sind unabhaengig von A und koennen parallel auf verschiedenen Terminals laufen. Wichtig: verschiedene Output-Verzeichnisse ODER verschiedene Experiment-Namen (sind per Default verschieden).

```bash
# Terminal 1: Core
python run_simulation.py --experiment exp10a_movement_only
python run_simulation.py --experiment exp10b_movement_harsh
python run_simulation.py --experiment exp10c_movement_full
python run_simulation.py --experiment exp10d_static_control

# Terminal 2: K1-Sensitivity
python run_simulation.py --experiment sens_k1_10
python run_simulation.py --experiment sens_k1_15
# ...

# Terminal 3: Ablation
python run_simulation.py --experiment ablation_movement_only
# ...
```

---

## Appendix: CLI-Referenz

```
python run_simulation.py [OPTIONS]

Optionen:
  --experiment NAME     Experiment aus Registry (PFLICHT, ausser --list)
  --runs N              Anzahl Runs (Default: aus Registry, meist 20)
  --seed-start N        Start-Seed fuer Reproduzierbarkeit (Default: 1)
  --output-dir PATH     Ausgabeverzeichnis (Default: data/)
  --ticks N             Simulationsticks pro Run (Default: 2000)
  --list                Alle Experimente auflisten und beenden
  --no-progress         Progress-Bars deaktivieren
  --log-level LEVEL     DEBUG / INFO / WARNING (Default: INFO)
  --no-timeseries       Keine run_NNN.csv Zeitreihen speichern

Beispiele:
  python run_simulation.py --list
  python run_simulation.py --experiment exp10b_movement_harsh
  python run_simulation.py --experiment sens_k1_20 --runs 30
  python run_simulation.py --experiment ablation_movement_only --no-progress --log-level DEBUG
```
