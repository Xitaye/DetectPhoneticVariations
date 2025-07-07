# Detect Phonetic Variations in Accented Speech with HuBERT  
*A Technion B.Sc. cap‑stone project*

Quantify how far an L2 (accented‑English) pronunciation drifts from an L1 (native‑English) reference by chaining  

* **SoX** – silence trimming & loudness normalisation  
* **Montreal Forced Aligner (MFA)** – word/phone alignment  
* **HuBERT** embeddings + Dynamic‑Time‑Warping  
* A cost‑projection + correlation routine we call a **Distance Signature**

---

## 📑 Table of Contents
1. [Quick‑Start Workflow](#quick-start-workflow)  
2. [Installation](#installation)  
3. [Required Data Layout](#required-data-layout)  
4. [Step‑by‑Step Usage](#step-by-step-usage)  
5. [Helper Scripts](#helper-scripts)  
6. [Batch Processing](#batch-processing)  
7. [Project Tree](#project-tree)  
8. [Dependencies](#dependencies)  
9. [Licence](#licence)

---

## Quick‑Start Workflow <a name="quick-start-workflow"></a>

```mermaid
flowchart LR
    A[Step 1 • Remove silence<br>(SoX)] --> B[Step 2 • Run MFA<br>(wav + txt → TextGrid)]
    B --> C[Step 3 • Trajectory analysis<br>(pair‑wise distance + metrics)]
```

---

## Installation <a name="installation"></a>

```bash
git clone https://github.com/Xitaye/DetectPhoneticVariations.git
cd DetectPhoneticVariations && git checkout refactor/reorder-code

# Conda (recommended)
conda env create -n phon_var -f environment.yml
conda activate phon_var

# Or pip + venv
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

> **External tools** (must be on your `PATH`):  
> • [SoX](http://sox.sourceforge.net/) • [MFA ≥ 3.2](https://montreal-forced-aligner.readthedocs.io)

---

## Required Data Layout <a name="required-data-layout"></a>

```
data/
└── raw/                               # --base_data_path
    ├── MidlandFemale/                 # L1 reference speakers
    │   ├── 01-01.wav
    │   ├── 01-01.TextGrid             # produced in Step 2
    │   └── …
    ├── SpanishFemale/                 # L2 comparison accents
    ├── GermanFemale/
    └── …
results/                               # created automatically
```

Every **`.wav`** must have a **matching `.TextGrid`** (same basename) before Step 3.

---

## Step‑by‑Step Usage <a name="step-by-step-usage"></a>

| # | Command | Description |
|---|---------|-------------|
| **1 Remove silence** | `python scripts/Silence_Remover.py --input data/raw --output data/clean` | Trim leading/trailing silence with SoX |
| **2 Run MFA** | `python scripts/MFA_Alignment_script.py --wav_dir data/clean --trans_dir transcripts_txt --out_dir data/clean` | Generate `.TextGrid` alignments |
| **3 Analyse one pair** | `python scripts/trajectory_analysis_4_HF_Changes.py data/clean/MidlandFemale/01-01.wav data/clean/SpanishFemale/01-01.wav results/01-01_pair.csv` | DTW → projection → Pearson/Spearman correlations |

Use the flags inside `trajectory_analysis_4_HF_Changes.py` (`plot_graph`, `plot_results`, …) to toggle interactive plots.

---

## Helper Scripts <a name="helper-scripts"></a>

| Script | Purpose |
|--------|---------|
| `scripts/Transcriptions_script.py` | Convert an IPA‑annotated CSV/XLSX into MFA‑ready `.txt` files |
| `scripts/Check_for_spn_script.py`  | Scan `.TextGrid`s after MFA and list segments labelled `spn` |
| `scripts/run_pairwise_over_all_data.py`* | Loop over **every** Midland × Other pair and call the trajectory‑analysis script |
| `scripts/workflows/statistics_analysis.py` | Aggregate many pair‑wise CSVs into a master `correlations.csv` |

\* The wrapper may appear as `Trajectory_analysis_for_all_scripts.py` in older commits—use whichever exists.

---

## Batch Processing <a name="batch-processing"></a>

```bash
python scripts/run_pairwise_over_all_data.py     --base_path   data/clean     --results_dir results     --feat_cache  features            # optional: pre‑extracted HuBERT .npy files
```

Creates:  
* `results/<speaker>/<sentence>_results.csv` – per‑pair metrics  
* `results/correlations.csv` – master sheet  

---

## Project Tree <a name="project-tree"></a>

```
DetectPhoneticVariations/
├── data/              # WAV + TextGrid live here
├── results/           # outputs land here
├── scripts/
│   ├── Silence_Remover.py
│   ├── Transcriptions_script.py
│   ├── MFA_Alignment_script.py
│   ├── Check_for_spn_script.py
│   ├── trajectory_analysis_4_HF_Changes.py
│   ├── run_pairwise_over_all_data.py
│   └── workflows/
│       └── statistics_analysis.py
├── environment.yml
├── requirements.txt
└── README.md
```

---

## Dependencies <a name="dependencies"></a>

* **Python ≥ 3.9** – libraries listed in `requirements.txt` (`torch`, `numpy`, `pandas`, `librosa`, `plotly`, `tgt`, …)  
* **SoX** & **Montreal Forced Aligner** (3.x)  
* GPU optional – scripts automatically fall back to CPU.

---

## Licence <a name="licence"></a>

MIT.  
© 2025 Reut Vitzner & Itay Efrat – feel free to open issues or pull requests!
